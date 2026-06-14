from config import config
from loguru import logger
from quixstreams import State

MAX_CANDLES_IN_STATE = config.max_candles_in_state


def fill_gaps(candles: list, new_candle: dict) -> list:
    """
    Detects gaps between the last candle in state and the incoming candle.
    Fills any gaps with synthetic candles so indicators have continuous data.

    Why this matters:
        Low-volume pairs (XRP/EUR, SOL/EUR) can have minutes with no trades.
        The candles service produces no candle for those empty windows.
        Technical indicators (RSI, MACD, Ichimoku) assume equal time spacing.
        Gaps distort every subsequent indicator calculation.

    Synthetic candle values:
        open = high = low = close = previous candle's close price
        volume = 0 (no trading occurred)

    Args:
        candles: Current list of candles in state
        new_candle: The incoming candle that may have a gap before it

    Returns:
        Updated candles list with gaps filled by synthetic candles
    """
    if not candles:
        return candles

    last_candle = candles[-1]
    candle_duration_ms = last_candle['window_end_ms'] - last_candle['window_start_ms']

    if candle_duration_ms <= 0:
        # Safety check — malformed candle, skip gap detection
        return candles

    # Check if there is a gap between last candle and new candle
    expected_next_start_ms = last_candle['window_end_ms']
    actual_next_start_ms = new_candle['window_start_ms']

    if actual_next_start_ms <= expected_next_start_ms:
        # No gap — candles are consecutive or overlapping (same window update)
        return candles

    # Calculate how many candles are missing
    gap_ms = actual_next_start_ms - expected_next_start_ms
    missing_count = int(gap_ms / candle_duration_ms)

    if missing_count <= 0:
        return candles

    # Cap synthetic candles to avoid filling state with only fake data.
    # Example: XRP/EUR might be inactive for hours (360+ missing candles).
    # We do not want 360 synthetic candles — cap at MAX_CANDLES_IN_STATE.
    if missing_count > MAX_CANDLES_IN_STATE:
        logger.warning(
            f'Large gap of {missing_count} missing candles for '
            f'{last_candle["pair"]}. '
            f'Capping at {MAX_CANDLES_IN_STATE} synthetic candles.'
        )
        missing_count = MAX_CANDLES_IN_STATE

    logger.debug(
        f'Filling {missing_count} missing candle(s) for {last_candle["pair"]}. '
        f'Gap: {last_candle["window_end_ms"]} → {actual_next_start_ms} '
        f'({gap_ms / 1000:.1f} seconds)'
    )

    # Create and insert synthetic candles
    for i in range(missing_count):
        synthetic_start_ms = expected_next_start_ms + (i * candle_duration_ms)
        synthetic_candle = {
            'pair': last_candle['pair'],
            'open': last_candle['close'],  # price stays flat
            'high': last_candle['close'],  # no movement
            'low': last_candle['close'],  # no movement
            'close': last_candle['close'],  # closes at same price
            'volume': 0.0,  # no trades occurred
            'window_start_ms': synthetic_start_ms,
            'window_end_ms': synthetic_start_ms + candle_duration_ms,
            'timestamp_ms': synthetic_start_ms + candle_duration_ms,
            'candle_seconds': last_candle.get('candle_seconds', 60),
        }
        candles.append(synthetic_candle)

        # Keep state within max size
        if len(candles) > MAX_CANDLES_IN_STATE:
            candles.pop(0)

    return candles


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
        # New window — fill any gaps before appending
        candles = fill_gaps(candles, candle)
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
