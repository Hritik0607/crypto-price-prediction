"""
backfill_news_signals.py

Creates news_signals_v1.parquet with signal=0 for all coins
at all timestamps that exist in technical_indicators_v1.parquet.

This replicates the news_signals Hopsworks feature group
so we can do the same JOIN locally without needing Spark.

Run once before training:
  uv run python backfill_news_signals.py
"""

import os

os.environ.setdefault('HOPSWORKS_CERT_DIR', 'D:/tmp')

from pathlib import Path

import pandas as pd
from loguru import logger

# ── Config ─────────────────────────────────────────────────────────────────────
COINS = ['BTC', 'ETH', 'XRP', 'SOL']
MODEL_NAME = 'dummy'

TECH_PARQUET = (
    Path(__file__).parent.parent
    / 'to-feature-store'
    / 'data'
    / 'technical_indicators_v1.parquet'
)
NEWS_PARQUET = (
    Path(__file__).parent.parent
    / 'to-feature-store'
    / 'data'
    / 'news_signals_v1.parquet'
)

# ── Step 1: Read timestamps from technical_indicators Parquet ──────────────────
logger.info(f'Reading technical_indicators from: {TECH_PARQUET}')
tech_df = pd.read_parquet(TECH_PARQUET)
logger.info(f'Loaded {len(tech_df)} rows')

unique_timestamps = tech_df['timestamp_ms'].unique()
# DEBUG
print(f'Sample raw timestamp: {unique_timestamps[0]}')
print(f'Type: {type(unique_timestamps[0])}')
print(f'int() result: {int(unique_timestamps[0])}')

logger.info(f'Found {len(unique_timestamps)} unique timestamps')
logger.info(
    f'Date range: '
    f'{pd.to_datetime(unique_timestamps.min(), unit="ms")} to '
    f'{pd.to_datetime(unique_timestamps.max(), unit="ms")}'
)

del tech_df  # free memory

# ── Step 2: Build news_signals rows ───────────────────────────────────────────
logger.info(f'Building news_signals rows for coins: {COINS}')

rows = []
for timestamp_ms in unique_timestamps:
    for coin in COINS:
        rows.append(
            {
                'coin': coin,
                'signal': 0,
                'model_name': MODEL_NAME,
                'timestamp_ms': timestamp_ms.item(),  # convert numpy int64 to Python int
            }
        )

news_df = pd.DataFrame(rows)
news_df['signal'] = news_df['signal'].astype('int8')
news_df['timestamp_ms'] = news_df['timestamp_ms'].astype('int64')  # ← explicit int64

logger.info(f'Built {len(news_df)} rows')
logger.info(f'Sample:\n{news_df.head(8)}')

# ── Step 3: Save to Parquet ────────────────────────────────────────────────────
NEWS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
news_df.to_parquet(NEWS_PARQUET, index=False)

logger.info(f'Saved news_signals_v1.parquet to: {NEWS_PARQUET}')
logger.info(f'Total rows: {len(news_df)}')
logger.info(f'Coins: {news_df["coin"].unique()}')
logger.info(f'Signals: {news_df["signal"].unique()} (all 0 = neutral baseline)')
