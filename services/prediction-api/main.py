from typing import Optional

from elasticsearch import Elasticsearch
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from loguru import logger
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Config ─────────────────────────────────────────────────────────────────────
class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='settings.env', env_file_encoding='utf-8'
    )
    elasticsearch_url: str = 'http://localhost:9200'
    elasticsearch_index: str = 'price_prediction'
    api_key: str = ''
    port: int = 8081


config = Config()

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title='Crypto Price Prediction API',
    description=(
        'REST API for serving crypto price predictions. '
        'For the monitoring dashboard, see the monitoring-service on port 8082.'
    ),
    version='1.0.0',
)

# ── ElasticSearch connection ───────────────────────────────────────────────────
ELASTICSEARCH_URL = config.elasticsearch_url
ELASTICSEARCH_INDEX = config.elasticsearch_index

es = Elasticsearch(ELASTICSEARCH_URL)
logger.info(f'ElasticSearch client created → {ELASTICSEARCH_URL}')
logger.info(f'Reading predictions from index → {ELASTICSEARCH_INDEX}')

# ── API key authentication ─────────────────────────────────────────────────────
API_KEY = config.api_key

if API_KEY:
    logger.info('API key authentication ENABLED')
else:
    logger.warning(
        'API key authentication DISABLED — set API_KEY in settings.env for production'
    )


def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    """
    Validates the X-API-Key header on protected endpoints.
    If API_KEY is empty in settings.env, authentication is disabled.
    Missing key  → HTTP 401
    Wrong key    → HTTP 403
    Correct key  → request proceeds
    """
    if not API_KEY:
        return
    if x_api_key is None:
        logger.warning('Request rejected — missing X-API-Key header')
        raise HTTPException(
            status_code=401,
            detail='Missing X-API-Key header. Include: X-API-Key: your-key',
        )
    if x_api_key != API_KEY:
        logger.warning('Request rejected — invalid API key')
        raise HTTPException(status_code=403, detail='Invalid API key')


# ── Response model ─────────────────────────────────────────────────────────────
class PredictionOutput(BaseModel):
    """
    Shape of the JSON response from GET /predict.
    All fields written to ElasticSearch by the inference service.
    """

    pair: str  # e.g. "BTC/USD"
    candle_seconds: int  # e.g. 60 (1-minute candles)
    prediction_seconds: int  # e.g. 300 (predicting 5 min ahead)
    prediction: float  # the predicted close price
    timestamp_ms: int  # when prediction was made (UTC epoch ms)
    timestamp_iso: str  # same, ISO format
    predicted_timestamp_ms: int  # the future moment being predicted
    predicted_timestamp_iso: str  # same, ISO format


# ── Health endpoint ────────────────────────────────────────────────────────────
@app.get('/health')
def health():
    """
    Health check endpoint — no authentication required.
    Returns HTTP 200 if healthy, HTTP 503 if ElasticSearch is unreachable.
    """
    try:
        es_healthy = es.ping()
    except Exception as e:
        logger.error(f'ElasticSearch ping failed: {e}')
        es_healthy = False

    if not es_healthy:
        raise HTTPException(status_code=503, detail='ElasticSearch is not reachable')

    return {
        'status': 'ok',
        'elasticsearch': 'connected',
        'index': ELASTICSEARCH_INDEX,
        'authentication': 'enabled' if API_KEY else 'disabled',
    }


# ── Predict endpoint ───────────────────────────────────────────────────────────
@app.get(
    '/predict', response_model=PredictionOutput, dependencies=[Depends(verify_api_key)]
)
def predict(
    pair: str = Query(
        ...,
        description='Crypto trading pair. Example: BTC/USD',
        examples=['BTC/USD'],
    ),
):
    """
    Returns the most recent price prediction for the requested crypto pair.

    Queries ElasticSearch for the latest prediction document,
    sorted by timestamp_ms descending (newest first).

    Requires X-API-Key header when API_KEY is set in settings.env.

    Returns HTTP 404 if no predictions exist for this pair.
    Returns HTTP 500 if ElasticSearch query fails.
    """
    logger.info(f'Prediction requested for pair: {pair}')

    try:
        result = es.search(
            index=ELASTICSEARCH_INDEX,
            body={
                'query': {'term': {'pair.keyword': pair}},
                'sort': [{'timestamp_ms': {'order': 'desc'}}],
                'size': 1,
            },
        )
    except Exception as e:
        logger.error(f'ElasticSearch query failed for pair {pair}: {e}')
        raise HTTPException(
            status_code=500, detail=f'ElasticSearch query failed: {str(e)}'
        ) from e

    hits = result['hits']['hits']
    if not hits:
        logger.warning(f'No predictions found in ElasticSearch for pair: {pair}')
        raise HTTPException(
            status_code=404,
            detail=(
                f'No predictions found for pair "{pair}". '
                f'Make sure the inference service is running.'
            ),
        )

    prediction_doc = hits[0]['_source']
    logger.info(
        f'Returning prediction for {pair}: '
        f'prediction={prediction_doc.get("prediction")}, '
        f'timestamp={prediction_doc.get("timestamp_iso")}'
    )
    return PredictionOutput(**prediction_doc)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=config.port, log_level='info')
