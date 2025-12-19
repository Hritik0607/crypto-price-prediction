from typing import List
from websocket import create_connection
import json
from loguru import logger
from .trade import Trade
from datetime import datetime


class KrakenWebsocketAPI:
    URL = 'wss://ws.kraken.com/v2'

    def __init__(self, pairs: List[str]):
        self.pairs = pairs

        self._ws_client = create_connection(self.URL)

        self._subscribe()

    def get_trades(self) -> List[Trade]:
        """
        Fetches the trades from the Kraken Websocket APIs and returns them as a list of Trade objects.

        Returns:
            List[Trade]: A list of Trade objects.
        """
        # receive the data from the websocket
        data = self._ws_client.recv()

        if 'heartbeat' in data:
            logger.info('Heartbeat received')
            return []

        # transform raw string into a JSON object
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            logger.error(f'Error decoding JSON: {e}')
            return []

        try:
            trades_data = data['data']
        except KeyError as e:
            logger.error(f'No `data` field with trades in the message {e}')
            return []

        trades = [
            Trade(
                pair=trade['symbol'],
                price=trade['price'],
                volume=trade['qty'],
                timestamp=trade['timestamp'],
                timestamp_ms=_datestr2milliseconds(trade['timestamp']),
            )
            for trade in trades_data
        ]
        # breakpoint()
        return trades


    def _subscribe(self):
        """
        Subscribes to the websocket and waits for the initial snapshot.
        """
        # send a subscribe message to the websocket
        self._ws_client.send(
            json.dumps(
                {
                    'method': 'subscribe',
                    'params': {
                        'channel': 'trade',
                        'symbol': self.pairs,
                        'snapshot': False,
                    },
                }
            )
        )

        for _ in self.pairs:
            _ = self._ws_client.recv()
            _ = self._ws_client.recv()

        for _ in self.pairs:
            _ = self._ws_client.recv()
            _ = self._ws_client.recv()


def _datestr2milliseconds(datestr: str) -> int:
    return int(
        datetime.strptime(datestr, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp() * 1000
    )