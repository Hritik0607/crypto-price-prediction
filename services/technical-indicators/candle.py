from config import config
from loguru import logger
from quixstreams import State

MAX_CANDLES_IN_STATE = config.max_candles_in_state


def update_candles(candle: dict, state: State) -> dict:
    """
    Updates the list of candles we have in our state using the latest candle

    If the latest candle corresponds to a new window, and the total number
    of candles in the state is less than the number of candles we want to keep,
    we just append it to the list.

    If it corresponds to the last window, we replace the last candle in the list.

    Args:
        candle: The latest candle
        state: The state of our application
        max_candles_in_state: The maximum number of candles to keep in the state
    Returns:
        None
    """
    # Get the list of candles from our state
    candles = state.get('candles', default=[])

    if not candles:
        # If the state is empty, we just append the latest candle to the list
        candles.append(candle)
    elif same_window(candle, candles[-1]):
        # Replace the last candle in the list with the latest candle
        candles[-1] = candle
    else:
        # Append the latest candle to the list
        candles.append(candle)

    # If the total number of candles in the state is greater than the maximum number of
    # candles we want to keep, we remove the oldest candle from the list
    if len(candles) > MAX_CANDLES_IN_STATE:
        candles.pop(0)

    # TODO: we should check the candles have no missing windows
    # This can happen for low volume pairs. In this case, we could interpoalte the missing windows

    logger.debug(f'Number of candles in state for {candle["pair"]}: {len(candles)}')

    # Update the state with the new list of candles
    state.set('candles', candles)

    return candle


def same_window(candle_1: dict, candle_2: dict) -> bool:
    """
    Check if the candle 1 and candle 2 are in the same window.

    Args:
        candle_1: The first candle
        candle_2: The second candle
    Returns:
        True if the candles are in the same window, False otherwise
    """
    return (
        candle_1['window_start_ms'] == candle_2['window_start_ms']
        and candle_1['window_end_ms'] == candle_2['window_end_ms']
        and candle_1['pair'] == candle_2['pair']
    )
