from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field


class NewsSignalOneCoin(BaseModel):
    coin: Literal[
        'BTC',
        'ETH',
        'SOL',
        'XRP',
        'DOGE',
        'ADA',
        'XLM',
        'LTC',
        'BCH',
        'DOT',
        'XMR',
        'EOS',
        'XEM',
        'ZEC',
        'ETC',
        'XLM',
        'LTC',
        'BCH',
        'DOT',
        'XMR',
        'EOS',
        'XEM',
        'ZEC',
        'ETC',
    ] = Field(description='The coin that the news is about')
    signal: Literal[1, 0, -1] = Field(
        description="""
    The signal of the news on the coin price.
    1 if the price is expected to go up
    -1 if it is expected to go down.
    0 if the news is not related to the coin.

    If the news is not related to the coin, no need to create a NewsSignal.
    """
    )


class NewsSignal(BaseModel):
    news_signals: list[NewsSignalOneCoin]

    def to_dict(self) -> dict:
        """
        Return a dictionary representation of the NewsSignal.
        """
        return {
            'btc_signal': self.btc_signal,
            'eth_signal': self.eth_signal,
            'reasoning': self.reasoning,
        }
        # raise NotImplementedError()


class BaseNewsSignalExtractor(ABC):
    @abstractmethod
    def get_signal(
        self, text: str, output_format: Literal['dict', 'NewsSignal'] = 'dict'
    ) -> dict | NewsSignal:
        pass

    # @property
    # def model_name(self) -> str:
    #     return self.model_name
