import asyncio
import json
import os
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import List

from config import config
from confluent_kafka import Consumer, KafkaError
from elasticsearch import Elasticsearch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from loguru import logger

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title='Crypto Monitoring Service',
    description='Monitors model performance and serves real-time dashboard',
    version='1.0.0',
)

# ── ElasticSearch ──────────────────────────────────────────────────────────────
es = Elasticsearch(config.elasticsearch_url)
logger.info(f'ElasticSearch client created → {config.elasticsearch_url}')


# ── Shared in-memory state ─────────────────────────────────────────────────────
# Updated by Kafka consumer thread, read by FastAPI WebSocket broadcaster.
# Python dicts are thread-safe for simple reads/writes (GIL protects us here).

# Latest price per pair from candles topic
# {'BTC/USD': {'price': 61338.0, 'timestamp_ms': 123, 'timestamp_iso': '...'}}
latest_prices: dict = {}

# Price history per pair — last 60 candle close prices
# Used for the price history chart on the dashboard
price_history: dict = defaultdict(lambda: deque(maxlen=60))

# Error history per pair — last 100 computed errors
# Used for the model error chart on the dashboard
error_history: dict = defaultdict(lambda: deque(maxlen=100))


# ── ElasticSearch helper — create model_errors index ──────────────────────────
def ensure_model_errors_index():
    """
    Creates the model_errors ElasticSearch index if it does not exist.
    Called once at startup.
    """
    if not es.indices.exists(index=config.model_errors_index):
        es.indices.create(
            index=config.model_errors_index,
            body={
                'mappings': {
                    'properties': {
                        'pair': {'type': 'keyword'},
                        'predicted': {'type': 'float'},
                        'actual': {'type': 'float'},
                        'error': {'type': 'float'},
                        'abs_error': {'type': 'float'},
                        'pct_error': {'type': 'float'},
                        'prediction_made_at_ms': {'type': 'long'},
                        'prediction_made_at_iso': {'type': 'keyword'},
                        'actual_timestamp_ms': {'type': 'long'},
                        'actual_timestamp_iso': {'type': 'keyword'},
                    }
                }
            },
        )
        logger.info(f'Created ElasticSearch index: {config.model_errors_index}')
    else:
        logger.info(f'ElasticSearch index already exists: {config.model_errors_index}')


# ── Error computation ──────────────────────────────────────────────────────────
def check_and_compute_error(pair: str, actual_price: float, window_end_ms: int):
    """
    For a given candle, checks if there is a prediction that targeted this
    candle's timestamp. If found, computes the prediction error and writes
    it to the model_errors ElasticSearch index.

    Matching logic:
        The prediction service sets predicted_timestamp_ms = timestamp_ms + 300000
        (i.e. 5 minutes after the candle that triggered the prediction).
        So we look for predictions where:
            predicted_timestamp_ms ≈ candle.window_end_ms (±60 seconds tolerance)

    Args:
        pair: e.g. 'BTC/USD'
        actual_price: the candle's close price (what actually happened)
        window_end_ms: when this candle's window closed
    """
    tolerance_ms = 60_000  # ±60 seconds

    try:
        result = es.search(
            index=config.predictions_index,
            body={
                'query': {
                    'bool': {
                        'must': [
                            {'term': {'pair.keyword': pair}},
                            {
                                'range': {
                                    'predicted_timestamp_ms': {
                                        'gte': window_end_ms - tolerance_ms,
                                        'lte': window_end_ms + tolerance_ms,
                                    }
                                }
                            },
                        ]
                    }
                },
                'sort': [{'timestamp_ms': {'order': 'desc'}}],
                'size': 1,
            },
        )

        hits = result['hits']['hits']
        if not hits:
            # No prediction targeted this timestamp — normal, happens often
            return

        prediction_doc = hits[0]['_source']
        predicted_price = prediction_doc['prediction']

        # Core error computation
        error = actual_price - predicted_price
        abs_error = abs(error)
        pct_error = abs(error / actual_price) * 100 if actual_price != 0 else 0

        actual_iso = datetime.fromtimestamp(
            window_end_ms / 1000, tz=timezone.utc
        ).isoformat()

        error_doc = {
            'pair': pair,
            'predicted': predicted_price,
            'actual': actual_price,
            'error': error,
            'abs_error': abs_error,
            'pct_error': pct_error,
            'prediction_made_at_ms': prediction_doc['timestamp_ms'],
            'prediction_made_at_iso': prediction_doc['timestamp_iso'],
            'actual_timestamp_ms': window_end_ms,
            'actual_timestamp_iso': actual_iso,
        }

        # Write to ElasticSearch
        es.index(index=config.model_errors_index, document=error_doc)

        # Update in-memory history for the dashboard
        error_history[pair].append(error_doc)

        logger.info(
            f'Error computed for {pair}: '
            f'predicted={predicted_price:.2f}, '
            f'actual={actual_price:.2f}, '
            f'error={error:+.2f} ({pct_error:.2f}%)'
        )

    except Exception as e:
        logger.error(f'Failed to compute error for {pair}: {e}')


# ── Process incoming candle ────────────────────────────────────────────────────
def process_candle(candle: dict):
    """
    Called for every candle message from Kafka.

    1. Updates latest_prices so the dashboard shows the current price
    2. Adds to price_history for the chart
    3. Checks if any prediction targeted this candle → compute error
    """
    pair = candle.get('pair')
    close = candle.get('close')
    window_end_ms = candle.get('window_end_ms')

    if not pair or close is None or window_end_ms is None:
        return

    # Only process pairs we are monitoring
    if pair not in config.pairs_to_monitor:
        return

    timestamp_iso = datetime.fromtimestamp(
        window_end_ms / 1000, tz=timezone.utc
    ).isoformat()

    # Update current price
    latest_prices[pair] = {
        'price': close,
        'timestamp_ms': window_end_ms,
        'timestamp_iso': timestamp_iso,
    }

    # Add to price history for the chart
    price_history[pair].append(
        {
            'time': timestamp_iso,
            'price': close,
        }
    )

    logger.debug(f'Candle processed: {pair} close={close} at {timestamp_iso}')

    # Check if a prediction targeted this candle and compute error
    check_and_compute_error(pair, close, window_end_ms)


# ── Kafka consumer background thread ──────────────────────────────────────────
def kafka_consumer_thread():
    """
    Runs in a background thread — reads candles from Kafka continuously.

    Why a thread and not asyncio?
    Kafka's consumer.poll() is a BLOCKING call.
    Blocking calls freeze asyncio's event loop.
    Running in a thread keeps FastAPI responsive.

    The thread updates the shared in-memory dicts (latest_prices,
    price_history, error_history) which FastAPI reads for the dashboard.
    """
    logger.info(
        f'Kafka consumer thread starting. '
        f'Topic: {config.kafka_input_topic}, '
        f'Broker: {config.kafka_broker_address}'
    )

    consumer = Consumer(
        {
            'bootstrap.servers': config.kafka_broker_address,
            'group.id': config.kafka_consumer_group,
            'auto.offset.reset': 'latest',
            'enable.auto.commit': True,
        }
    )
    consumer.subscribe([config.kafka_input_topic])

    logger.info('Kafka consumer subscribed to candles topic')

    try:
        while True:
            # poll() blocks for up to 1 second waiting for a message
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                # No message in the last 1 second — normal, keep polling
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # Reached end of partition — not an error
                    continue
                logger.error(f'Kafka consumer error: {msg.error()}')
                continue

            try:
                candle = json.loads(msg.value().decode('utf-8'))
                process_candle(candle)
            except json.JSONDecodeError as e:
                logger.error(f'Failed to decode Kafka message: {e}')
            except Exception as e:
                logger.error(f'Failed to process candle: {e}')

    except Exception as e:
        logger.error(f'Kafka consumer thread crashed: {e}')
    finally:
        consumer.close()
        logger.info('Kafka consumer closed')


# ── WebSocket connection manager ───────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            f'Dashboard client connected. ' f'Total: {len(self.active_connections)}'
        )

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            f'Dashboard client disconnected. ' f'Total: {len(self.active_connections)}'
        )

    async def broadcast(self, data: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.active_connections.remove(conn)


manager = ConnectionManager()


# ── Build WebSocket payload ────────────────────────────────────────────────────
def get_latest_predictions_from_es() -> dict:
    """Queries ElasticSearch for the most recent prediction for each pair."""
    predictions = {}
    for pair in config.pairs_to_monitor:
        try:
            result = es.search(
                index=config.predictions_index,
                body={
                    'query': {'term': {'pair.keyword': pair}},
                    'sort': [{'timestamp_ms': {'order': 'desc'}}],
                    'size': 1,
                },
            )
            hits = result['hits']['hits']
            if hits:
                predictions[pair] = hits[0]['_source']
        except Exception as e:
            logger.error(f'Failed to get prediction for {pair}: {e}')
    return predictions


def build_payload(msg_type: str = 'update') -> dict:
    """
    Builds the complete JSON payload sent over WebSocket to the dashboard.

    Contains:
        current price per pair (from Kafka candles — live)
        latest prediction per pair (from ElasticSearch)
        price history per pair (last 60 candles, for the chart)
        error history per pair (last 100 errors, for the error chart)
    """
    predictions = get_latest_predictions_from_es()

    return {
        'type': msg_type,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'latest_prices': latest_prices,
        'predictions': predictions,
        'price_history': {
            pair: list(history) for pair, history in price_history.items()
        },
        'error_history': {
            pair: list(history) for pair, history in error_history.items()
        },
    }


# ── Background broadcast loop ──────────────────────────────────────────────────
async def broadcast_loop():
    """Sends updates to all connected dashboard browsers every 30 seconds."""
    while True:
        try:
            if manager.active_connections:
                payload = build_payload('update')
                await manager.broadcast(payload)
                logger.info(
                    f'Broadcast sent to {len(manager.active_connections)} client(s)'
                )
        except Exception as e:
            logger.error(f'Broadcast error: {e}')
        await asyncio.sleep(30)


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event('startup')
async def startup_event():
    """
    On startup:
    1. Create ElasticSearch model_errors index if needed
    2. Start Kafka consumer in background thread
    3. Start WebSocket broadcast loop
    """
    # Create ElasticSearch index for errors
    ensure_model_errors_index()

    # Start Kafka consumer in a daemon thread
    # daemon=True means it dies when the main process dies
    thread = threading.Thread(target=kafka_consumer_thread, daemon=True)
    thread.start()
    logger.info('Kafka consumer thread started')

    # Start WebSocket broadcast loop
    asyncio.create_task(broadcast_loop())
    logger.info('WebSocket broadcast loop started')


# ── Dashboard endpoint ─────────────────────────────────────────────────────────
@app.get('/', response_class=HTMLResponse)
async def dashboard():
    """
    Serves the real-time monitoring dashboard.
    Open http://localhost:8082/ in your browser.
    """
    html_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'dashboard.html'
    )
    with open(html_path, 'r', encoding='utf-8') as f:
        return f.read()


# ── Health endpoint ────────────────────────────────────────────────────────────
@app.get('/health')
def health():
    try:
        es_healthy = es.ping()
    except Exception:
        es_healthy = False

    return {
        'status': 'ok' if es_healthy else 'degraded',
        'elasticsearch': 'connected' if es_healthy else 'unreachable',
        'pairs_monitored': config.pairs_to_monitor,
        'kafka_topic': config.kafka_input_topic,
        'latest_prices': {pair: data['price'] for pair, data in latest_prices.items()},
    }


# ── WebSocket endpoint ─────────────────────────────────────────────────────────
@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time dashboard updates.
    Sends initial data immediately on connect.
    Background loop sends updates every 30 seconds.
    """
    await manager.connect(websocket)
    try:
        # Send initial data immediately so dashboard shows something right away
        initial_payload = build_payload('initial')
        await websocket.send_json(initial_payload)
        logger.info('Initial payload sent to new dashboard client')

        # Keep connection alive
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
            except asyncio.TimeoutError:
                await websocket.send_json({'type': 'ping'})

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f'WebSocket error: {e}')
        manager.disconnect(websocket)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=config.port, log_level='info')
