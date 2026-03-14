from typing import Literal, Optional

from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama

from .base import BaseNewsSignalExtractor, NewsSignal


class OllamaNewsSignalExtractor(BaseNewsSignalExtractor):
    def __init__(
        self,
        model_name: str,
        base_url: str,
        temperature: Optional[float] = 0,
    ):
        self.llm = Ollama(
            model=model_name,
            temperature=temperature,
            base_url=base_url,
            additional_kwargs={
                'num_ctx': 1024,  # 🔑 FIX: limits context memory
                'num_predict': 256,  # 🔑 FIX: limits output tokens
            },
        )

        self.prompt_template = PromptTemplate(
            template="""
            You are an expert crypto financial analyst with deep knowledge of market dynamics and sentiment analysis.
            Analyze the following news story and determine its potential impact on crypto asset prices.
            Focus on both direct mentions and indirect implications for each asset.

            Do not output data for a given coin if the news is not relevant to it.

            ## Example input
            "Goldman Sachs wants to invest in Bitcoin and Ethereum, but not in XRP"

            ## Example output
            [
                {"coin": "BTC", "signal": 1},
                {"coin": "ETH", "signal": 1},
                {"coin": "XRP", "signal": -1},
            ]

            News story to analyze:
            {news_story}
            """
        )

        self.model_name = model_name

    def get_signal(
        self,
        text: str,
        output_format: Literal['list', 'NewsSignal'] = 'NewsSignal',
    ) -> list[dict] | NewsSignal:
        """
        Get the news signal from the given `text`

        Args:
            text: The news article to get the signal from
            output_format: The format of the output

        Returns:
            The news signal
        """
        response: NewsSignal = self.llm.structured_predict(
            NewsSignal,
            prompt=self.prompt_template,
            news_story=text,
        )

        # keep only news signals with non-zero signal
        response.news_signals = [
            news_signal
            for news_signal in response.news_signals
            if news_signal.signal != 0
        ]

        if output_format == 'list':
            return response.model_dump()['news_signals']
        else:
            return response


if __name__ == '__main__':
    from .config import OllamaConfig

    config = OllamaConfig()

    llm = OllamaNewsSignalExtractor(
        model_name=config.model_name, base_url=config.ollama_base_url
    )

    examples = [
        'Extracting news signal from European crypto ETN and ETP report: Wednesday, June 8',
        # 'Bitcoin ETF ads spotted on China’s Alipay payment app',
        # 'U.S. Supreme Court Lets Nvidia’s Crypto Lawsuit Move Forward',
        # 'Trump’s World Liberty Acquires ETH, LINK, and AAVE in $12M Crypto Shopping Spree',
    ]

    for example in examples:
        print(f'Example: {example}')
        response = llm.get_signal(example)
        print(response)

    """
    Example: Bitcoin ETF ads spotted on China’s Alipay payment app
    {
        "btc_signal": 1,
        "eth_signal": 0,
        'reasoning': "The news of Bitcoin ETF ads being spotted on China's Alipay payment
        app suggests a growing interest in Bitcoin and other cryptocurrencies among Chinese
        investors. This could lead to increased demand for BTC, causing its price to rise."
    }

    Example: U.S. Supreme Court Lets Nvidia’s Crypto Lawsuit Move Forward
    {
        'btc_signal': -1,
        'eth_signal': -1,
        'reasoning': "The US Supreme Court's decision allows Nvidia to pursue its crypto
        lawsuit, which could lead to increased regulatory uncertainty and potential
        restrictions on cryptocurrency mining. This could negatively impact the prices
        of both BTC and ETH."
    }

    Example: Trump’s World Liberty Acquires ETH, LINK, and AAVE in $12M Crypto Shopping Spree
    {
        'btc_signal': 0,
        'eth_signal': 1,
        'reasoning': "The acquisition of ETH by a major company like
        Trump's World Liberty suggests that there is increased demand for
        Ethereum, which could lead to an increase in its price."
    }
    """
