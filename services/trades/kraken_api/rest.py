import json
import os
import time
from typing import List, Optional

import requests
from loguru import logger

from .base import TradesAPI
from .trade import Trade


class KrakenRestAPI(TradesAPI):
    def __init__(
        self,
        pairs: List[str],
        last_n_days: int,
        max_retries: int = 5,
        initial_delay_seconds: int = 1,
        cursor_dir: str = 'cursors',
    ):
        self.pairs = pairs
        self.last_n_days = last_n_days

        self.apis = [
            KrakenRestAPISinglePair(
                pair=pair,
                last_n_days=last_n_days,
                max_retries=max_retries,
                initial_delay_seconds=initial_delay_seconds,
                cursor_dir=cursor_dir,
            )
            for pair in self.pairs
        ]

    def get_trades(self) -> List[Trade]:
        """
        Get trades for each pair, sort them by timestamp and return the trades.
        """
        trades = []
        for api in self.apis:
            if not api.is_done():
                trades += api.get_trades()

        # sort the trades by timestamp
        trades.sort(key=lambda x: x.timestamp_ms)

        # breakpoint()

        return trades

    def is_done(self) -> bool:
        """
        We are done when all the APIs are done.
        """
        for api in self.apis:
            if not api.is_done():
                return False
        return True


class KrakenRestAPISinglePair(TradesAPI):
    URL = 'https://api.kraken.com/0/public/Trades'

    def __init__(
        self,
        pair: str,
        last_n_days: int,
        max_retries: int = 5,
        initial_delay_seconds: int = 1,
        cursor_dir: str = 'cursors',
    ):
        self.pair = pair
        self.last_n_days = last_n_days
        self.max_retries = max_retries
        self.initial_delay_seconds = initial_delay_seconds
        self.cursor_dir = cursor_dir
        self._is_done = False

        # Cursor file is pair-specific — one file per pair
        # BTC/USD → cursors/cursor_BTC_USD.json
        safe_pair = pair.replace('/', '_')
        self.cursor_file = os.path.join(cursor_dir, f'cursor_{safe_pair}.json')

        # ── Try to resume from saved cursor ──────────────────────────────
        # If a cursor file exists from a previous run, resume from there.
        # This prevents reprocessing already-fetched trades after a crash.
        saved_cursor = self._load_cursor()

        if saved_cursor is not None:
            self.since_timestamp_ns = saved_cursor
            logger.info(
                f'Resuming {pair} from saved cursor: {saved_cursor} '
                f'(skipping already-fetched trades)'
            )
        else:
            # No saved cursor — start fresh from last_n_days ago
            # get current timestamp in nanoseconds
            self.since_timestamp_ns = int(
                time.time_ns() - last_n_days * 24 * 60 * 60 * 1000000000
            )

            logger.info(
                f'No saved cursor for {pair}. '
                f'Starting fresh from {last_n_days} days ago: '
                f'{self.since_timestamp_ns}'
            )

    def _load_cursor(self) -> Optional[int]:
        """
        Loads the saved pagination cursor from disk.
        Returns the saved since_timestamp_ns or None if no cursor exists.
        Handles file corruption gracefully — returns None and starts fresh.
        """
        if not os.path.exists(self.cursor_file):
            return None

        try:
            with open(self.cursor_file, 'r') as f:
                data = json.load(f)
            cursor = data['since_timestamp_ns']
            saved_at = data.get('saved_at', 'unknown')
            logger.info(
                f'Loaded cursor for {self.pair}: {cursor} ' f'(saved at {saved_at})'
            )
            return int(cursor)
        except Exception as e:
            logger.warning(
                f'Failed to load cursor for {self.pair} '
                f'from {self.cursor_file}: {e}. '
                f'Starting fresh.'
            )
            return None

    def _save_cursor(self):
        """
        Saves the current pagination cursor to disk.
        Called after every successful API page fetch.
        If the service crashes after this call, the next restart
        will resume from this exact position — no duplicates.
        """
        try:
            os.makedirs(self.cursor_dir, exist_ok=True)
            with open(self.cursor_file, 'w') as f:
                json.dump(
                    {
                        'since_timestamp_ns': self.since_timestamp_ns,
                        'pair': self.pair,
                        'saved_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    },
                    f,
                    indent=2,
                )
            logger.debug(f'Cursor saved for {self.pair}: {self.since_timestamp_ns}')
        except Exception as e:
            # Log but do not crash — cursor saving is best-effort
            # The service still works without it, just loses resume ability
            logger.warning(f'Failed to save cursor for {self.pair}: {e}')

    def get_trades(self) -> List[Trade]:
        """
        Sends a request to the Kraken API and returns the trades for the pair.
        """
        retry_delay = (
            self.initial_delay_seconds
        )  # doubles each retry: 1s, 2s, 4s, 8s, 16s

        headers = {'Accept': 'application/json'}
        params = {
            'pair': self.pair,
            'since': self.since_timestamp_ns,
        }

        # Make HTTP request with exponential backoff retry
        response = None
        retry_delay = self.initial_delay_seconds

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.request(
                    'GET', self.URL, headers=headers, params=params
                )
                break  # If the request is successful, exit the retry loop
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries:
                    logger.error(
                        f'Request failed for {self.pair} '
                        f'(attempt {attempt}/{self.max_retries}): {e}. '
                        f'Retrying in {retry_delay}s...'
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # exponential backoff: 1s, 2s, 4s, 8s, 16s
                else:
                    logger.error(
                        f'All {self.max_retries} attempts failed for {self.pair}: {e}. '
                        f'Returning empty trades list.'
                    )
                    return []
        if response is None:
            logger.error(f'Response is None after retries for {self.pair}')
            return []

        # Parse JSON response
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError as e:
            logger.error(f'Failed to parse response as json: {e}')
            return []

        # Extract trades from response
        try:
            trades = data['result'][self.pair]
        except KeyError as e:
            logger.error(f'Failed to get trades for pair {self.pair}: {e}')
            return []

        # convert the trades to Trade objects
        trades = [
            Trade.from_kraken_rest_api_response(
                pair=self.pair,
                price=trade[0],
                volume=trade[1],
                timestamp_sec=trade[2],
            )
            for trade in trades
        ]

        # update the since_timestamp_ns
        self.since_timestamp_ns = int(float(data['result']['last']))

        # Step 6: Save cursor to disk for crash recovery
        self._save_cursor()

        # check if we are done
        # TODO: check if this stopping conditions really work
        if self.since_timestamp_ns > int(time.time_ns() - 1000000000):
            self._is_done = True
        if self.since_timestamp_ns == 0:
            self._is_done = True

        return trades

    def is_done(self) -> bool:
        return self._is_done
