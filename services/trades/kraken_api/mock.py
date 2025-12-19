from datetime import datetime
from typing import List
from pydantic import BaseModel
from time import sleep

from .trade import Trade


class Trade(BaseModel):
    """
    A trade from the Kraken API.
    """

    pair: str
    price: float
    volume: float
    timestamp: datetime
    timestamp_ms: int

    def to_dict(self) -> dict:
        return {
            'pair': self.pair,
            'price': self.price,
            'volume': self.volume,
            'timestamp_ms': self.timestamp_ms}


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