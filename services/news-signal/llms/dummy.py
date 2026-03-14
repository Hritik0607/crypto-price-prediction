from typing import Literal

from .base import BaseNewsSignalExtractor, NewsSignal


class DummyNewsSignalExtractor(BaseNewsSignalExtractor):
    """
    A dummy news signal that is super stupid, but works fast.
    I added this so I can run the backfill pipeline without having to wait for the LLM to respond.
    """

    def __init__(self):
        self.model_name = 'dummy'

    def get_signal(
        self,
        text: str,
        output_format: Literal['dict', 'NewsSignal'] = 'NewsSignal',
    ) -> dict | NewsSignal:
        """
        Always returns a NewsSignal with a signal of 1 for BTC and 0 for ETH

        Args:
            text: The news article to get the signal from
            output_format: The format of the output

        Returns:
            The news signal
        """
        if output_format == 'list':
            return [
                {
                    'coin': 'BTC',
                    'signal': 1,
                },
                {
                    'coin': 'ETH',
                    'signal': -1,
                },
                {
                    'coin': 'XRP',
                    'signal': 0,
                },
                {
                    'coin': 'SOL',
                    'signal': 0,
                },
            ]
        else:
            raise NotImplementedError(
                'Only list output format is supported for DummyNewsSignalExtractor'
            )
