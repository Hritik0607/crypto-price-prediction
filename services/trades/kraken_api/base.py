from abc import ABC, abstractmethod
from typing import List

from .trade import Trade


class TradesAPI(ABC):
    @abstractmethod
    def get_trades(self) -> List[Trade]:
        pass

    @abstractmethod
    def is_done(self) -> bool:
        pass
