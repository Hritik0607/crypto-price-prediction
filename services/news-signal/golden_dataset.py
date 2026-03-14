import json
import random
from typing import Literal

import pandas as pd

instruction = """
You are an expert crypto financial analyst with deep knowledge of market dynamics and sentiment analysis.
Analyze the following news story and determine its potential impact on crypto asset prices.
Focus on both direct mentions and indirect implications for each asset.

Do not output data for a given coin if the news is not relevant to it.

## Example input news story
"Goldman Sachs wants to invest in Bitcoin and Ethereum, but not in XRP"

## Example output
[
    {"coin": "BTC", "signal": 1},
    {"coin": "ETH", "signal": 1},
    {"coin": "XRP", "signal": -1},
]
"""


def generate_dataset(
    model_provider: Literal['claude', 'ollama'],
    n: int,
    input_file: str,
    output_file: str,
):
    """
    Generate a dataset of (instruction, input, output) tuples to do
    Supervised Fine Tuning.

    Args:
        model_provider: The model provider to use.
        n: The number of news stories to generate.
        input_file: The file to read the news stories from.
        output_file: The file to write the dataset to.

    Returns:
        None
    """

    # load dataset
    df = pd.read_csv(input_file)
    news = df['title'].tolist()

    # random sample of n news
    news = random.sample(news, n)

    # llm
    from llms.factory import get_llm

    llm = get_llm(model_provider=model_provider)

    print('model_name:', llm.model_name)

    from tqdm import tqdm

    for news_item in tqdm(news):
        try:
            signals = llm.get_signal(news_item)
            output = {
                'instruction': instruction,
                'input': news_item,
                'output': signals.model_dump_json(),
                'teacher_model_name': llm.model_name,
            }

            # append to file
            with open(output_file, 'a') as f:
                f.write(json.dumps(output) + '\n')

        except Exception as e:
            print(f'Error: {e}')
            continue


if __name__ == '__main__':
    from fire import Fire

    Fire(generate_dataset)
