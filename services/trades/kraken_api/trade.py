from datetime import datetime

from pydantic import BaseModel


class Trade(BaseModel):
    """
    A trade from the Kraken API.
    """

    pair: str
    price: float
    volume: float
    timestamp: datetime
    timestamp_ms: int

    @property
    def timestamp_ms(self) -> int:
        return int(self.timestamp.timestamp() * 1000)

    def to_dict(self) -> dict:
        return self.model_dump_json()

    # def to_dict(self) -> dict:
    #     return {
    #         'pair': self.pair,
    #         'price': self.price,
    #         'volume': self.volume,
    #         'timestamp_ms': self.timestamp_ms}
