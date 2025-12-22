import numpy as np
from quixstreams import State
from talib import stream


def compute_indicators(
    candle: dict,
    state: State,
) -> dict:
    """
    Computes the technical indicators from the candles in the state
    """
    candles = state.get('candles', [])

    # extract open, high, low, close from the candles
    # open = np.array([candle['open'] for candle in candles])
    high = np.array([candle['high'] for candle in candles])
    low = np.array([candle['low'] for candle in candles])
    close = np.array([candle['close'] for candle in candles])
    volume = np.array([candle['volume'] for candle in candles])

    indicators = {}

    # Compute the technical indicators
    # I got these after talking to Claude.
    # However, this is where you should really spend time and experiment to build
    # a good set of indicators. This is the FEATURE ENGINEERING part of the project,
    # which is what makes or breaks the performance of the model.
    # 1. RSI (Relative Strength Index) at 9, 14 and 21
    indicators['rsi_9'] = stream.RSI(close, timeperiod=9)
    indicators['rsi_14'] = stream.RSI(close, timeperiod=14)
    indicators['rsi_21'] = stream.RSI(close, timeperiod=21)

    # 2. MACD (Moving Average Convergence Divergence)
    # Standard settings: fast=12, slow=26, signal=9
    # Crypto often benefits from slightly faster settings
    indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = (
        stream.MACD(close, fastperiod=10, slowperiod=24, signalperiod=9)
    )

    # 3. Bollinger Bands
    # Timeperiod: 20 with 2 standard deviations is standard
    # Crypto markets often benefit from slightly tighter bands
    (
        indicators['bbands_upper'],
        indicators['bbands_middle'],
        indicators['bbands_lower'],
    ) = stream.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    # 4. Stochastic RSI
    # More sensitive than regular RSI, good for volatile crypto markets
    # Timeperiod: 14 is standard, but 10 can be more responsive
    indicators['stochrsi_fastk'], indicators['stochrsi_fastd'] = stream.STOCHRSI(
        close, timeperiod=10, fastk_period=5, fastd_period=3, fastd_matype=0
    )

    # 5. ADX (Average Directional Index)
    # Measures trend strength regardless of direction
    # Timeperiod: 14 is standard
    indicators['adx'] = stream.ADX(high, low, close, timeperiod=14)

    # 6. Volume Profile
    # EMA of volume can help confirm price movements
    # Shorter periods for crypto due to 24/7 trading
    indicators['volume_ema'] = stream.EMA(volume, timeperiod=10)

    # 7. Ichimoku Cloud
    # Modified settings for crypto (traditionally 9, 26, 52)
    conversion = stream.EMA(close, timeperiod=9)
    base = stream.EMA(close, timeperiod=20)
    leading_span_a = (conversion + base) / 2
    leading_span_b = stream.EMA(close, timeperiod=40)
    indicators['ichimoku_conv'] = conversion
    indicators['ichimoku_base'] = base
    indicators['ichimoku_span_a'] = leading_span_a
    indicators['ichimoku_span_b'] = leading_span_b

    # 8. MFI (Money Flow Index)
    # Volume-weighted RSI, good for crypto due to volume importance
    # Timeperiod: 14 is standard, but 10 more responsive for crypto
    indicators['mfi'] = stream.MFI(high, low, close, volume, timeperiod=10)

    # 9. ATR (Average True Range)
    # Volatility indicator, crucial for crypto
    # Shorter timeperiod due to crypto volatility
    indicators['atr'] = stream.ATR(high, low, close, timeperiod=10)

    # 10. Price ROC (Rate of Change)
    # Momentum indicator showing velocity of price changes
    # Shorter period for crypto markets
    indicators['price_roc'] = stream.ROC(close, timeperiod=6)

    # SMA at 7, 14 and 21
    indicators['sma_7'] = stream.SMA(close, timeperiod=7)
    indicators['sma_14'] = stream.SMA(close, timeperiod=14)
    indicators['sma_21'] = stream.SMA(close, timeperiod=21)

    # we want to emit a message with the latest candle and the indicators we just computed
    final_message = {
        **candle,
        **indicators,
    }

    # breakpoint()

    return final_message
