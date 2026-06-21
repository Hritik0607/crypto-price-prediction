from time import sleep
from typing import List

from .base import TradesAPI
from .trade import Trade


class KrakenMockAPI(TradesAPI):
    def __init__(self, pairs: List[str]):
        self.pairs = pairs

    def get_trades(self) -> List[Trade]:
        """
        Returns a list of mock trades.
        """
        mock_trades = []

        # Generate one fake trade per pair so the mock covers all configured pairs
        for pair in self.pairs:
            mock_trades.append(
                Trade(
                    pair=pair,
                    price=0.5117,
                    volume=40.0,
                    timestamp='2023-09-25T07:49:37.708706Z',
                    timestamp_ms=172719357708706,
                )
            )
            mock_trades.append(
                Trade(
                    pair=pair,
                    price=0.5317,
                    volume=40.0,
                    timestamp='2023-09-25T07:49:37.708706Z',
                    timestamp_ms=172719357708706,
                )
            )
        sleep(1)

        return mock_trades

    def is_done(self) -> bool:
        """
        Mock always returns False — runs forever like live mode.
        The test mode has no concept of "finished" — it keeps producing
        fake trades until the service is manually stopped.
        (Bug 2 fix — this method was completely missing)
        """
        return False
