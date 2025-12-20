from datetime import datetime
from time import sleep
from typing import List

from .trade import Trade


class KrakenMockAPI:
    def __init__(self, pair: str):
        self.pair = pair

    def get_trades(self) -> List[Trade]:
        """
        Returns a list of mock trades.
        """
        mock_trades = [
            Trade(
                pair=self.pair,
                price=0.5117,
                volume=40.0,
                timestamp=datetime(2023, 9, 25, 7, 49, 37, 708706),
                timestamp_ms=172719357708706,
            ),
            Trade(
                pair=self.pair,
                price=0.5317,
                volume=40.0,
                timestamp=datetime(2023, 9, 25, 7, 49, 37, 708706),
                timestamp_ms=172719357708706,
            ),
        ]

        sleep(1)

        return mock_trades
