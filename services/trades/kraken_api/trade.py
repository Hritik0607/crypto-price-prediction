from datetime import datetime

from pydantic import BaseModel


class Trade(BaseModel):
    """
    A trade from the Kraken API.
    """

    pair: str
    price: float
    volume: float
    timestamp: str
    timestamp_ms: int

    @classmethod
    def from_kraken_rest_api_response(
        cls,
        pair: str,
        price: float,
        volume: float,
        timestamp_sec: float,
    ) -> 'Trade':
        """
        Returns a Trade object from the Kraken REST API response.

        E.g response:
            ['76395.00000', '0.01305597', 1731155565.4159515, 's', 'm', '', 75468573]

            price: float
            volume: float
            timestamp_sec: float
        """
        # convert the timestamp_sec from float to str
        timestamp_ms = int(float(timestamp_sec) * 1000)

        return cls(
            pair=pair,
            price=price,
            volume=volume,
            timestamp=cls._milliseconds2datestr(timestamp_ms),
            timestamp_ms=timestamp_ms,
        )

    @classmethod
    def from_kraken_websocket_api_response(
        cls,
        pair: str,
        price: float,
        volume: float,
        timestamp: str,
    ) -> 'Trade':
        return cls(
            pair=pair,
            price=price,
            volume=volume,
            timestamp=timestamp,
            timestamp_ms=cls._datestr2milliseconds(timestamp),
        )

    @staticmethod
    def _milliseconds2datestr(milliseconds: int) -> str:
        return datetime.fromtimestamp(milliseconds / 1000).strftime(
            '%Y-%m-%dT%H:%M:%S.%fZ'
        )

    @staticmethod
    def _datestr2milliseconds(datestr: str) -> int:
        return int(
            datetime.strptime(datestr, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp() * 1000
        )

    def to_str(self) -> str:
        # pydantic method to convert the model to a dict
        return self.model_dump_json()

    def to_dict(self) -> dict:
        return self.model_dump()
