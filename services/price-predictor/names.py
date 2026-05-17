def get_model_name(
    pair_to_predict: str,
    candle_seconds: int,
    prediction_seconds: int,
) -> str:
    """
    Returns the name of the model to save to the model registry
    """
    return f'price_predictor_pair_{pair_to_predict.replace("/", "_")}_candle_seconds_{candle_seconds}_prediction_seconds_{prediction_seconds}'
