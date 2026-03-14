from typing import Literal

from .base import BaseNewsSignalExtractor


def get_llm(
    model_provider: Literal['anthropic', 'ollama', 'dummy'],
) -> BaseNewsSignalExtractor:
    """
    Returns the LLM we want for the news signal extractor

    Args:
        model_provider: The model provider to use

    Returns:
        The LLM we want for the news signal extractor
    """
    if model_provider == 'anthropic':
        from .claude import ClaudeNewsSignalExtractor
        from .config import AnthropicConfig

        config = AnthropicConfig()

        return ClaudeNewsSignalExtractor(
            model_name=config.model_name,
            api_key=config.api_key,
        )

    elif model_provider == 'ollama':
        from .config import OllamaConfig
        from .ollama import OllamaNewsSignalExtractor

        config = OllamaConfig()

        return OllamaNewsSignalExtractor(
            model_name=config.model_name,
            base_url=config.ollama_base_url,
        )

    elif model_provider == 'dummy':
        from .dummy import DummyNewsSignalExtractor

        return DummyNewsSignalExtractor()

    else:
        raise ValueError(f'Unsupported model provider: {model_provider}')
