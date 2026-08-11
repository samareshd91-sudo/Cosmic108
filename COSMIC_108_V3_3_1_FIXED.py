import time
import sqlite3
from pathlib import Path
import requests
import pandas as pd
import numpy as np
import streamlit as st

from datetime import datetime, timezone
from collections import deque

import plotly.graph_objects as go



# ============================================================
# COSMIC 108 V3.0
# PART 1/5 — FOUNDATION + LIVE DATA + DATA QUALITY
# ============================================================

APP_NAME = "COSMIC 108 V3.2 ADVANCED SMC"
VERSION = "3.2"

# ------------------------------------------------------------
# GLOBAL CONFIGURATION
# ------------------------------------------------------------

CONFIG = {
    # SQLite persistence
    "database_path": str(Path(__file__).with_name("cosmic_108.db")),
    "exchange": "KUCOIN",

    # Symbols to scan
    "symbols": [
        "SOL-USDT",
        "ETH-USDT",
        "BNB-USDT",
        "XRP-USDT",
    ],

    # Timeframes
    "htf_1d": "1day",
    "htf_4h": "4hour",
    "htf_1h": "1hour",
    "poi_tf": "15min",
    "entry_tf": "5min",

    # Candle history
    "history_1d": 100,
    "history_4h": 150,
    "history_1h": 150,
    "history_15m": 200,
    "history_5m": 250,

    # Live refresh
    "refresh_seconds": 30,

    # Data protection
    "request_timeout": 8,
    "max_retries": 3,

    # SMC
    "swing_window": 3,

    # Setup expiry
    "sweep_expiry": 6,
    "mss_expiry": 4,
    "displacement_expiry": 3,
    "bos_expiry": 3,

    # POI
    "poi_proximity_percent": 1.5,

    # Advanced confirmations
    "adx_period": 14,
    "adx_min": 20.0,
    "fvg_max_age": 40,
    "ob_max_age": 60,
    "premium_discount_lookback": 50,
    "cvd_lookback": 20,
    "signal_min_score": 85,

    # Risk
    "account_risk_usd": 10.0,

    # Execution assumptions
    "maker_fee": 0.0006,
    "taker_fee": 0.0008,
    "slippage": 0.0005,

    # Signal protection
    "cooldown_seconds": 1800,
    "max_audit_records": 50,
}


# ============================================================
# KUCOIN LIVE DATA FETCHER
# ============================================================

class KuCoinLive:

    BASE_URL = "https://api.kucoin.com"

    @staticmethod
    def get_klines(
        symbol: str,
        interval: str,
        limit: int = 100
    ) -> pd.DataFrame:

        endpoint = (
            f"{KuCoinLive.BASE_URL}"
            f"/api/v1/market/candles"
        )

        params = {
            "symbol": symbol,
            "type": interval
        }

        last_error = None

        for attempt in range(
            1,
            CONFIG["max_retries"] + 1
        ):

            try:

                response = requests.get(
                    endpoint,
                    params=params,
                    timeout=CONFIG["request_timeout"]
                )

                response.raise_for_status()

                payload = response.json()

                data = payload.get("data", [])

                if not data:
                    return pd.DataFrame()

                # KuCoin candle format:
                #
                # [
                #   timestamp,
                #   open,
                #   close,
                #   high,
                #   low,
                #   volume,
                #   turnover
                # ]

                columns = [
                    "timestamp",
                    "open",
                    "close",
                    "high",
                    "low",
                    "volume",
                    "turnover"
                ]

                df = pd.DataFrame(
                    data,
                    columns=columns
                )

                df = df[
                    [
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume"
                    ]
                ]

                numeric_columns = [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume"
                ]

                for column in numeric_columns:
                    df[column] = pd.to_numeric(
                        df[column],
                        errors="coerce"
                    )

                df = df.dropna()

                df["timestamp"] = (
                    df["timestamp"]
                    .astype(np.int64)
                )

                # KuCoin returns candles in reverse order
                # so always sort chronologically.
                df = df.sort_values(
                    "timestamp"
                ).reset_index(drop=True)

                # Remove duplicate timestamps.
                df = df.drop_duplicates(
                    subset=["timestamp"],
                    keep="last"
                ).reset_index(drop=True)

                return df.tail(limit).reset_index(
                    drop=True
                )

            except Exception as exc:

                last_error = exc

                if attempt < CONFIG["max_retries"]:
                    time.sleep(1)

        print(
            f"KuCoin API Error {symbol} {interval}: {last_error}"
        )

        return pd.DataFrame()


# ============================================================
# DATA QUALITY GUARD
# ============================================================

class DataQualityGuard:

    TIMEFRAME_SECONDS = {
        "1min": 60,
        "5min": 300,
        "15min": 900,
        "1hour": 3600,
        "4hour": 14400,
        "1day": 86400,
    }

    @staticmethod
    def validate(
        df: pd.DataFrame,
        interval: str,
        minimum_candles: int = 50
    ):

        if df is None or df.empty:

            return (
                False,
                "CRITICAL: Empty market data"
            )

        if len(df) < minimum_candles:

            return (
                False,
                f"CRITICAL: Only {len(df)} candles available"
            )

        required_columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        for column in required_columns:

            if column not in df.columns:

                return (
                    False,
                    f"CRITICAL: Missing column {column}"
                )

        # ----------------------------------------------------
        # Duplicate candle protection
        # ----------------------------------------------------

        if df["timestamp"].duplicated().any():

            return (
                False,
                "ERROR: Duplicate timestamps detected"
            )

        # ----------------------------------------------------
        # Timestamp ordering
        # ----------------------------------------------------

        if not df["timestamp"].is_monotonic_increasing:

            return (
                False,
                "ERROR: Candle timestamps not chronological"
            )

        # ----------------------------------------------------
        # Gap detection
        # ----------------------------------------------------

        expected = (
            DataQualityGuard
            .TIMEFRAME_SECONDS
            .get(interval)
        )

        if expected is not None:

            differences = (
                df["timestamp"]
                .diff()
                .dropna()
            )

            max_gap = expected * 1.5

            if (differences > max_gap).any():

                return (
                    False,
                    "WARNING: Candle gap detected"
                )

        # ----------------------------------------------------
        # OHLC sanity checks
        # ----------------------------------------------------

        invalid_ohlc = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"] > df["open"]) |
            (df["low"] > df["close"])
        )

        if invalid_ohlc.any():

            return (
                False,
                "ERROR: Invalid OHLC structure"
            )

        # ----------------------------------------------------
        # Negative / zero price protection
        # ----------------------------------------------------

        if (
            (df["open"] <= 0).any() or
            (df["high"] <= 0).any() or
            (df["low"] <= 0).any() or
            (df["close"] <= 0).any()
        ):

            return (
                False,
                "ERROR: Invalid price detected"
            )

        # ----------------------------------------------------
        # Volume protection
        # ----------------------------------------------------

        if (df["volume"] < 0).any():

            return (
                False,
                "ERROR: Negative volume detected"
            )

        zero_volume_count = (
            df["volume"] == 0
        ).sum()

        if zero_volume_count > 3:

            return (
                False,
                "WARNING: Excessive zero-volume candles"
            )

        # ----------------------------------------------------
        # NaN protection
        # ----------------------------------------------------

        if df[required_columns].isna().any().any():

            return (
                False,
                "ERROR: NaN values detected"
            )

        return True, "DATA OK"


# ============================================================
# STALE DATA PROTECTION
# ============================================================

class StaleDataGuard:

    @staticmethod
    def is_stale(
        df: pd.DataFrame,
        interval: str,
        tolerance_multiplier: float = 2.0
    ):

        if df is None or df.empty:
            return True

        timeframe_seconds = (
            DataQualityGuard
            .TIMEFRAME_SECONDS
            .get(interval)
        )

        if timeframe_seconds is None:
            return False

        latest_timestamp = int(
            df["timestamp"].iloc[-1]
        )

        now_timestamp = int(
            datetime.now(
                timezone.utc
            ).timestamp()
        )

        age = (
            now_timestamp -
            latest_timestamp
        )

        allowed_age = (
            timeframe_seconds *
            tolerance_multiplier
        )

        return age > allowed_age


# ============================================================
# MARKET DATA BUNDLE
# ============================================================

class MarketData:

    def __init__(self, symbol: str):

        self.symbol = symbol

        self.data = {
            "1d": pd.DataFrame(),
            "4h": pd.DataFrame(),
            "1h": pd.DataFrame(),
            "15m": pd.DataFrame(),
            "5m": pd.DataFrame(),
        }

        self.errors = []

    def fetch_all(self):

        self.errors = []

        requests_map = {
            "1d": (
                CONFIG["htf_1d"],
                CONFIG["history_1d"]
            ),
            "4h": (
                CONFIG["htf_4h"],
                CONFIG["history_4h"]
            ),
            "1h": (
                CONFIG["htf_1h"],
                CONFIG["history_1h"]
            ),
            "15m": (
                CONFIG["poi_tf"],
                CONFIG["history_15m"]
            ),
            "5m": (
                CONFIG["entry_tf"],
                CONFIG["history_5m"]
            ),
        }

        for key, (interval, limit) in requests_map.items():

            df = KuCoinLive.get_klines(
                self.symbol,
                interval,
                limit
            )

            valid, message = (
                DataQualityGuard.validate(
                    df,
                    interval
                )
            )

            if not valid:

                self.errors.append(
                    f"{key}: {message}"
                )

                self.data[key] = pd.DataFrame()

                continue

            if StaleDataGuard.is_stale(
                df,
                interval
            ):

                self.errors.append(
                    f"{key}: STALE DATA"
                )

                self.data[key] = pd.DataFrame()

                continue

            self.data[key] = df

        return self.data

    def is_ready(self):

        required = [
            "1d",
            "4h",
            "1h",
            "15m",
            "5m"
        ]

        return all(
            not self.data[key].empty
            for key in required
        )


# ============================================================
# SIMPLE AUDIT LOGGER
# ============================================================

class SQLiteStore:

    def __init__(self, db_path=None):
        self.db_path = db_path or CONFIG["database_path"]
        self._initialize()

    def _connect(self):
        conn = sqlite3.connect(
            self.db_path,
            timeout=10,
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _initialize(self):
        Path(self.db_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    event TEXT NOT NULL,
                    score INTEGER NOT NULL DEFAULT 0,
                    message TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at
                ON audit_logs(created_at DESC)
            """)
            conn.commit()

    def add_audit(self, symbol, event, score, message, created_at):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_logs
                    (created_at, symbol, event, score, message)
                VALUES (?, ?, ?, ?, ?)
                """,
                (created_at, symbol, event, int(score), message)
            )
            conn.commit()

    def latest_audits(self, limit=10):
        limit = max(1, min(int(limit), 500))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT created_at, symbol, event, score, message
                FROM audit_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()

        return [
            {
                "time": row["created_at"],
                "symbol": row["symbol"],
                "event": row["event"],
                "score": row["score"],
                "message": row["message"]
            }
            for row in rows
        ]

    def cleanup_audits(self, keep=5000):
        keep = max(100, int(keep))
        with self._connect() as conn:
            conn.execute(
                """
                DELETE FROM audit_logs
                WHERE id NOT IN (
                    SELECT id
                    FROM audit_logs
                    ORDER BY id DESC
                    LIMIT ?
                )
                """,
                (keep,)
            )
            conn.commit()


# Single SQLite database used by the application.
DB = SQLiteStore()


class AuditLogger:

    def __init__(self, max_records=None):

        if max_records is None:
            max_records = CONFIG[
                "max_audit_records"
            ]

        self.records = deque(
            maxlen=max_records
        )

        # Load recent persisted audit records so a restart does not
        # erase the visible audit history.
        for item in reversed(
            DB.latest_audits(max_records)
        ):
            self.records.appendleft(item)

    def log(
        self,
        symbol: str,
        event: str,
        message: str,
        score: int = 0
    ):

        created_at = datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")

        item = {
            "time": created_at,
            "symbol": symbol,
            "event": event,
            "score": score,
            "message": message
        }

        self.records.appendleft(item)

        try:
            DB.add_audit(
                symbol=symbol,
                event=event,
                score=score,
                message=message,
                created_at=created_at
            )
            # Keep the SQLite file bounded for long-running engines.
            DB.cleanup_audits(5000)
        except Exception as exc:
            print(
                f"SQLite audit persistence warning: {exc}"
            )

    def latest(self, limit=10):

        return list(
            self.records
        )[:limit]



# ============================================================
# BASIC CONNECTION TEST
# ============================================================

def test_market_connection():

    print(
        f"{APP_NAME} {VERSION}\n"
        "Testing KuCoin public market connection..."
    )

    df = KuCoinLive.get_klines(
        "BTC-USDT",
        "5min",
        10
    )

    if df.empty:

        print("❌ KuCoin connection/data test failed")

        return False

    print("✅ KuCoin live market connection OK")

    print(f"BTC-USDT candles received: {len(df)}")

    return True


# ============================================================
# PART 1 TEST
# ============================================================

if __name__ == "__main__":

    test_market_connection()
# ==========================================
# PART 2 — DATA QUALITY + SMC MATH ENGINE
# ==========================================

class DataQualityGuard:

    @staticmethod
    def validate(df, timeframe_minutes=5, min_candles=50):

        if df is None or df.empty:
            return False, "NO DATA"

        if len(df) < min_candles:
            return False, f"INSUFFICIENT DATA ({len(df)}/{min_candles})"

        required = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        for col in required:
            if col not in df.columns:
                return False, f"MISSING COLUMN: {col}"

        if df["timestamp"].duplicated().any():
            return False, "DUPLICATE TIMESTAMP"

        if df[["open", "high", "low", "close"]].isnull().any().any():
            return False, "OHLC NULL DATA"

        if df["volume"].isnull().any():
            return False, "VOLUME NULL DATA"

        if (df["high"] < df["low"]).any():
            return False, "INVALID HIGH/LOW"

        expected = timeframe_minutes * 60

        diffs = df["timestamp"].diff().dropna()

        if (diffs > expected * 1.5).any():
            return False, "CANDLE GAP DETECTED"

        if (df["volume"] < 0).any():
            return False, "INVALID VOLUME"

        return True, "DATA OK"


class SMCMath:

    # --------------------------------------
    # ATR
    # --------------------------------------

    @staticmethod
    def calculate_atr(df, period=14):

        previous_close = df["close"].shift(1)

        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - previous_close).abs()
        tr3 = (df["low"] - previous_close).abs()

        true_range = pd.concat(
            [tr1, tr2, tr3],
            axis=1
        ).max(axis=1)

        atr = true_range.rolling(
            period,
            min_periods=period
        ).mean()

        return atr


    # --------------------------------------
    # EMA
    # --------------------------------------

    @staticmethod
    def ema(df, period):

        return df["close"].ewm(
            span=period,
            adjust=False,
            min_periods=period
        ).mean()


    # --------------------------------------
    # Volume Expansion
    # --------------------------------------

    @staticmethod
    def volume_expansion(df, idx, period=20, multiplier=1.5):

        if idx < period:
            return False

        avg_volume = df["volume"].iloc[
            idx - period:idx
        ].mean()

        if avg_volume <= 0:
            return False

        return (
            df["volume"].iloc[idx]
            >= avg_volume * multiplier
        )


    # --------------------------------------
    # Displacement
    # --------------------------------------

    @staticmethod
    def displacement(
        df,
        idx,
        atr_multiplier=1.2,
        volume_multiplier=1.5,
        body_ratio_min=0.70
    ):

        if idx < 20:
            return False

        atr_series = SMCMath.calculate_atr(df)

        atr = atr_series.iloc[idx]

        if pd.isna(atr) or atr <= 0:
            return False

        candle_open = df["open"].iloc[idx]
        candle_close = df["close"].iloc[idx]
        candle_high = df["high"].iloc[idx]
        candle_low = df["low"].iloc[idx]

        body = abs(
            candle_close - candle_open
        )

        candle_range = (
            candle_high - candle_low
        )

        if candle_range <= 0:
            return False

        body_ratio = body / candle_range

        volume_ok = SMCMath.volume_expansion(
            df,
            idx,
            20,
            volume_multiplier
        )

        atr_ok = (
            body >= atr * atr_multiplier
        )

        body_ok = (
            body_ratio >= body_ratio_min
        )

        return (
            atr_ok
            and body_ok
            and volume_ok
        )


    # --------------------------------------
    # Causal Swing Detection
    # --------------------------------------

    @staticmethod
    def detect_swings(
        df,
        left=3,
        right=3
    ):
        """
        Confirmed swing detection.

        Important:
        A swing is only confirmed AFTER
        the required right-side candles
        have closed.

        This prevents look-ahead bias.
        """

        data = df.copy()

        data["swing_high"] = False
        data["swing_low"] = False

        if len(data) < left + right + 1:
            return data

        highs = data["high"].values
        lows = data["low"].values

        for i in range(
            left,
            len(data) - right
        ):

            left_highs = highs[
                i - left:i
            ]

            right_highs = highs[
                i + 1:i + right + 1
            ]

            left_lows = lows[
                i - left:i
            ]

            right_lows = lows[
                i + 1:i + right + 1
            ]

            if (
                highs[i] > left_highs.max()
                and
                highs[i] > right_highs.max()
            ):
                data.loc[
                    data.index[i],
                    "swing_high"
                ] = True

            if (
                lows[i] < left_lows.min()
                and
                lows[i] < right_lows.min()
            ):
                data.loc[
                    data.index[i],
                    "swing_low"
                ] = True

        return data


    # --------------------------------------
    # Latest Confirmed Swing
    # --------------------------------------

    @staticmethod
    def latest_swing_high(df, before_idx=None):

        if before_idx is None:
            before_idx = len(df) - 1

        candidates = df[
            (df["swing_high"])
            &
            (df.index < df.index[before_idx])
        ]

        if candidates.empty:
            return None

        return candidates.iloc[-1]


    @staticmethod
    def latest_swing_low(df, before_idx=None):

        if before_idx is None:
            before_idx = len(df) - 1

        candidates = df[
            (df["swing_low"])
            &
            (df.index < df.index[before_idx])
        ]

        if candidates.empty:
            return None

        return candidates.iloc[-1]


    # --------------------------------------
    # Liquidity Sweep
    # --------------------------------------

    @staticmethod
    def detect_liquidity_sweep(
        df,
        direction,
        lookback=20
    ):

        if len(df) < lookback + 5:
            return False

        current = df.iloc[-1]

        previous = df.iloc[
            -lookback:-1
        ]

        if direction == "LONG":

            liquidity_level = previous["low"].min()

            swept = (
                current["low"]
                < liquidity_level
            )

            reclaimed = (
                current["close"]
                > liquidity_level
            )

            return swept and reclaimed

        if direction == "SHORT":

            liquidity_level = previous["high"].max()

            swept = (
                current["high"]
                > liquidity_level
            )

            rejected = (
                current["close"]
                < liquidity_level
            )

            return swept and rejected

        return False


    # --------------------------------------
    # Market Structure Shift
    # --------------------------------------

    @staticmethod
    def detect_mss(
        df,
        direction
    ):

        if len(df) < 15:
            return False

        swings = SMCMath.detect_swings(
            df,
            left=3,
            right=3
        )

        current_close = swings["close"].iloc[-1]

        if direction == "LONG":

            swing_highs = swings[
                swings["swing_high"]
            ]

            if swing_highs.empty:
                return False

            latest_high = (
                swing_highs["high"].iloc[-1]
            )

            return (
                current_close
                > latest_high
            )

        if direction == "SHORT":

            swing_lows = swings[
                swings["swing_low"]
            ]

            if swing_lows.empty:
                return False

            latest_low = (
                swing_lows["low"].iloc[-1]
            )

            return (
                current_close
                < latest_low
            )

        return False


    # --------------------------------------
    # BOS
    # --------------------------------------

    @staticmethod
    def detect_bos(
        df,
        direction,
        lookback=10
    ):

        if len(df) < lookback + 2:
            return False

        current = df.iloc[-1]

        previous = df.iloc[
            -lookback:-1
        ]

        if direction == "LONG":

            structure_high = (
                previous["high"].max()
            )

            return (
                current["close"]
                > structure_high
            )

        if direction == "SHORT":

            structure_low = (
                previous["low"].min()
            )

            return (
                current["close"]
                < structure_low
            )

        return False


    # --------------------------------------
    # FVG Detection
    # --------------------------------------

    @staticmethod
    def detect_fvg(df):

        bullish_fvg = []
        bearish_fvg = []

        if len(df) < 3:
            return {
                "bullish": bullish_fvg,
                "bearish": bearish_fvg
            }

        for i in range(2, len(df)):

            c1 = df.iloc[i - 2]
            c3 = df.iloc[i]

            # Bullish FVG
            if c1["high"] < c3["low"]:

                bullish_fvg.append({
                    "index": i,
                    "low": c1["high"],
                    "high": c3["low"],
                    "type": "BULLISH_FVG"
                })

            # Bearish FVG
            if c1["low"] > c3["high"]:

                bearish_fvg.append({
                    "index": i,
                    "low": c3["high"],
                    "high": c1["low"],
                    "type": "BEARISH_FVG"
                })

        return {
            "bullish": bullish_fvg,
            "bearish": bearish_fvg
        }


    # --------------------------------------
    # POI Proximity
    # --------------------------------------

    @staticmethod
    def poi_proximity(
        current_price,
        poi,
        max_distance_pct=1.5
    ):

        if poi is None:
            return False

        distance = (
            abs(
                current_price
                - (
                    poi["high"]
                    + poi["low"]
                ) / 2
            )
            / current_price
        ) * 100

        return (
            distance
            <= max_distance_pct
        )


    # --------------------------------------
    # Candle Direction
    # --------------------------------------

    @staticmethod
    def candle_direction(
        df,
        idx=-1
    ):

        candle = df.iloc[idx]

        if candle["close"] > candle["open"]:
            return "BULLISH"

        if candle["close"] < candle["open"]:
            return "BEARISH"

        return "NEUTRAL"


    # --------------------------------------
    # Volatility Regime
    # --------------------------------------

    @staticmethod
    def volatility_regime(df):

        atr = SMCMath.calculate_atr(
            df,
            14
        )

        if atr.isna().all():
            return "UNKNOWN"

        current_atr = atr.iloc[-1]

        median_atr = (
            atr.dropna()
            .tail(50)
            .median()
        )

        if current_atr > median_atr * 1.5:
            return "HIGH"

        if current_atr < median_atr * 0.7:
            return "LOW"

        return "NORMAL"
    # ==========================================
# PART 3 — MARKET REGIME + HTF + POI ENGINE
# ==========================================

class MarketRegimeEngine:

    @staticmethod
    def get_trend(df, fast=20, slow=50):

        if df is None or len(df) < slow:
            return "UNKNOWN"

        data = df.copy()

        data["ema_fast"] = (
            data["close"]
            .ewm(
                span=fast,
                adjust=False
            )
            .mean()
        )

        data["ema_slow"] = (
            data["close"]
            .ewm(
                span=slow,
                adjust=False
            )
            .mean()
        )

        price = data["close"].iloc[-1]
        fast_ema = data["ema_fast"].iloc[-1]
        slow_ema = data["ema_slow"].iloc[-1]

        if (
            price > fast_ema
            and fast_ema > slow_ema
        ):
            return "BULLISH"

        if (
            price < fast_ema
            and fast_ema < slow_ema
        ):
            return "BEARISH"

        return "CHOP"


    @staticmethod
    def btc_regime(
        df_1d,
        df_4h,
        df_1h
    ):

        trend_1d = MarketRegimeEngine.get_trend(
            df_1d
        )

        trend_4h = MarketRegimeEngine.get_trend(
            df_4h
        )

        trend_1h = MarketRegimeEngine.get_trend(
            df_1h
        )

        bullish_count = [
            trend_1d,
            trend_4h,
            trend_1h
        ].count("BULLISH")

        bearish_count = [
            trend_1d,
            trend_4h,
            trend_1h
        ].count("BEARISH")

        if bullish_count == 3:
            regime = "STRONG_BULLISH"

        elif bearish_count == 3:
            regime = "STRONG_BEARISH"

        elif bullish_count >= 2:
            regime = "BULLISH"

        elif bearish_count >= 2:
            regime = "BEARISH"

        else:
            regime = "CHOP"

        return {
            "regime": regime,
            "1d": trend_1d,
            "4h": trend_4h,
            "1h": trend_1h
        }


    @staticmethod
    def allow_direction(
        regime,
        direction
    ):

        if direction == "LONG":

            return regime in [
                "BULLISH",
                "STRONG_BULLISH"
            ]

        if direction == "SHORT":

            return regime in [
                "BEARISH",
                "STRONG_BEARISH"
            ]

        return False


# ==========================================
# HTF ALIGNMENT ENGINE
# ==========================================

class HTFAlignmentEngine:

    @staticmethod
    def analyze(
        df_1d,
        df_4h,
        df_1h,
        direction
    ):

        trend_1d = MarketRegimeEngine.get_trend(
            df_1d
        )

        trend_4h = MarketRegimeEngine.get_trend(
            df_4h
        )

        trend_1h = MarketRegimeEngine.get_trend(
            df_1h
        )

        trends = [
            trend_1d,
            trend_4h,
            trend_1h
        ]

        if direction == "LONG":

            aligned = all(
                x == "BULLISH"
                for x in trends
            )

        elif direction == "SHORT":

            aligned = all(
                x == "BEARISH"
                for x in trends
            )

        else:
            aligned = False

        return {
            "aligned": aligned,
            "1d": trend_1d,
            "4h": trend_4h,
            "1h": trend_1h
        }


# ==========================================
# 15M POI ENGINE
# ==========================================

class POIEngine:

    def __init__(
        self,
        proximity_pct=1.5
    ):

        self.proximity_pct = (
            proximity_pct
        )


    # --------------------------------------
    # Detect Order Blocks
    # --------------------------------------

    @staticmethod
    def detect_order_blocks(df):

        bullish_obs = []
        bearish_obs = []

        if len(df) < 5:
            return {
                "bullish": bullish_obs,
                "bearish": bearish_obs
            }

        for i in range(
            2,
            len(df) - 1
        ):

            current = df.iloc[i]
            next_candle = df.iloc[i + 1]

            # ----------------------------------
            # Bullish Order Block
            # Last bearish candle before
            # strong bullish expansion
            # ----------------------------------

            if (
                current["close"]
                < current["open"]
            ):

                next_body = abs(
                    next_candle["close"]
                    - next_candle["open"]
                )

                next_range = (
                    next_candle["high"]
                    - next_candle["low"]
                )

                if next_range > 0:

                    body_ratio = (
                        next_body
                        / next_range
                    )

                    if (
                        next_candle["close"]
                        > current["high"]
                        and body_ratio >= 0.60
                    ):

                        bullish_obs.append({
                            "index": i,
                            "low": current["low"],
                            "high": current["high"],
                            "type": "BULLISH_OB"
                        })


            # ----------------------------------
            # Bearish Order Block
            # Last bullish candle before
            # strong bearish expansion
            # ----------------------------------

            if (
                current["close"]
                > current["open"]
            ):

                next_body = abs(
                    next_candle["close"]
                    - next_candle["open"]
                )

                next_range = (
                    next_candle["high"]
                    - next_candle["low"]
                )

                if next_range > 0:

                    body_ratio = (
                        next_body
                        / next_range
                    )

                    if (
                        next_candle["close"]
                        < current["low"]
                        and body_ratio >= 0.60
                    ):

                        bearish_obs.append({
                            "index": i,
                            "low": current["low"],
                            "high": current["high"],
                            "type": "BEARISH_OB"
                        })

        return {
            "bullish": bullish_obs,
            "bearish": bearish_obs
        }


    # --------------------------------------
    # Build Directional POI
    # --------------------------------------

    def get_directional_pois(
        self,
        df_15m,
        direction
    ):

        fvg_data = (
            SMCMath.detect_fvg(
                df_15m
            )
        )

        ob_data = (
            self.detect_order_blocks(
                df_15m
            )
        )

        pois = []

        if direction == "LONG":

            pois.extend(
                fvg_data["bullish"]
            )

            pois.extend(
                ob_data["bullish"]
            )

        elif direction == "SHORT":

            pois.extend(
                fvg_data["bearish"]
            )

            pois.extend(
                ob_data["bearish"]
            )

        return pois


    # --------------------------------------
    # Find Nearest POI
    # --------------------------------------

    def find_nearest_poi(
        self,
        current_price,
        pois
    ):

        if not pois:
            return None

        ranked = []

        for poi in pois:

            midpoint = (
                poi["low"]
                + poi["high"]
            ) / 2

            distance_pct = (
                abs(
                    current_price
                    - midpoint
                )
                / current_price
            ) * 100

            ranked.append({
                "poi": poi,
                "distance_pct": distance_pct
            })

        ranked.sort(
            key=lambda x:
            x["distance_pct"]
        )

        nearest = ranked[0]

        if (
            nearest["distance_pct"]
            <= self.proximity_pct
        ):

            result = nearest["poi"].copy()

            result["distance_pct"] = (
                nearest["distance_pct"]
            )

            return result

        return None


# ==========================================
# DIRECTION ENGINE
# ==========================================

class DirectionEngine:

    @staticmethod
    def determine(
        htf_result,
        btc_regime
    ):

        long_score = 0
        short_score = 0

        # ----------------------------------
        # HTF
        # ----------------------------------

        if htf_result["1d"] == "BULLISH":
            long_score += 2

        if htf_result["4h"] == "BULLISH":
            long_score += 2

        if htf_result["1h"] == "BULLISH":
            long_score += 2

        if htf_result["1d"] == "BEARISH":
            short_score += 2

        if htf_result["4h"] == "BEARISH":
            short_score += 2

        if htf_result["1h"] == "BEARISH":
            short_score += 2

        # ----------------------------------
        # BTC Regime
        # ----------------------------------

        if btc_regime in [
            "BULLISH",
            "STRONG_BULLISH"
        ]:

            long_score += 2

        if btc_regime in [
            "BEARISH",
            "STRONG_BEARISH"
        ]:

            short_score += 2

        # ----------------------------------
        # Final Direction
        # ----------------------------------

        if long_score >= 6 and (
            long_score > short_score
        ):

            return {
                "direction": "LONG",
                "long_score": long_score,
                "short_score": short_score
            }

        if short_score >= 6 and (
            short_score > long_score
        ):

            return {
                "direction": "SHORT",
                "long_score": long_score,
                "short_score": short_score
            }

        return {
            "direction": "NEUTRAL",
            "long_score": long_score,
            "short_score": short_score
        }


# ==========================================
# POI QUALITY ENGINE
# ==========================================

class POIQualityEngine:

    @staticmethod
    def evaluate(
        poi,
        current_price,
        direction
    ):

        if poi is None:
            return {
                "valid": False,
                "quality": 0,
                "reason": "NO DIRECTIONAL POI"
            }

        quality = 0

        # ----------------------------------
        # Correct POI direction
        # ----------------------------------

        if direction == "LONG":

            if poi["type"] in [
                "BULLISH_FVG",
                "BULLISH_OB"
            ]:
                quality += 30

        elif direction == "SHORT":

            if poi["type"] in [
                "BEARISH_FVG",
                "BEARISH_OB"
            ]:
                quality += 30

        # ----------------------------------
        # Proximity
        # ----------------------------------

        distance = poi.get(
            "distance_pct",
            999
        )

        if distance <= 0.50:
            quality += 30

        elif distance <= 1.00:
            quality += 20

        elif distance <= 1.50:
            quality += 10

        # ----------------------------------
        # POI Type Bonus
        # ----------------------------------

        if poi["type"] in [
            "BULLISH_OB",
            "BEARISH_OB"
        ]:
            quality += 20

        if poi["type"] in [
            "BULLISH_FVG",
            "BEARISH_FVG"
        ]:
            quality += 15

        # ----------------------------------
        # Final Quality
        # ----------------------------------

        quality = min(
            quality,
            100
        )

        return {
            "valid": quality >= 50,
            "quality": quality,
            "distance_pct": round(
                distance,
                3
            ),
            "type": poi["type"]
        }
    # ==========================================
# PART 4 — EXECUTION CHAIN + RISK + SCORING
# ==========================================

class SetupState:

    IDLE = "IDLE"
    POI_FOUND = "POI_FOUND"
    SWEEP_DETECTED = "SWEEP_DETECTED"
    MSS_CONFIRMED = "MSS_CONFIRMED"
    DISPLACEMENT = "DISPLACEMENT"
    BOS_CONFIRMED = "BOS_CONFIRMED"
    RETEST_CONFIRMED = "RETEST_CONFIRMED"
    EXPIRED = "EXPIRED"


class SetupTracker:

    def __init__(self, symbol):

        self.symbol = symbol

        self.state = SetupState.IDLE

        self.state_index = None

        self.sweep_index = None

        self.active_poi = None

        self.direction = None

        self.signal_id = None

        self.last_signal_timestamp = 0

        self.cooldown_seconds = 1800

        self.expiry_rules = {
            SetupState.SWEEP_DETECTED: 6,
            SetupState.MSS_CONFIRMED: 4,
            SetupState.DISPLACEMENT: 3,
            SetupState.BOS_CONFIRMED: 3
        }


    # --------------------------------------
    # Reset Setup
    # --------------------------------------

    def reset(self):

        self.state = SetupState.IDLE

        self.state_index = None

        self.sweep_index = None

        self.active_poi = None

        self.direction = None

        self.signal_id = None


    # --------------------------------------
    # Start New Setup
    # --------------------------------------

    def start_setup(
        self,
        direction,
        poi,
        current_index
    ):

        self.direction = direction

        self.active_poi = poi

        self.state = SetupState.POI_FOUND

        self.state_index = current_index

        self.sweep_index = None

        self.signal_id = (
            f"{self.symbol}_"
            f"{direction}_"
            f"{current_index}"
        )


    # --------------------------------------
    # State Transition
    # --------------------------------------

    def transition(
        self,
        new_state,
        current_index
    ):

        self.state = new_state

        self.state_index = current_index

        if new_state == SetupState.SWEEP_DETECTED:

            self.sweep_index = current_index


    # --------------------------------------
    # Dynamic Expiry
    # --------------------------------------

    def check_expiry(
        self,
        current_index
    ):

        if self.state == SetupState.IDLE:
            return False

        if self.state not in self.expiry_rules:
            return False

        if self.state_index is None:
            return False

        max_candles = (
            self.expiry_rules[
                self.state
            ]
        )

        candles_passed = (
            current_index
            - self.state_index
        )

        if candles_passed > max_candles:

            self.state = SetupState.EXPIRED

            return True

        return False


    # --------------------------------------
    # Cooldown
    # --------------------------------------

    def cooldown_active(self):

        return (
            time.time()
            - self.last_signal_timestamp
            < self.cooldown_seconds
        )


    def mark_signal_sent(self):

        self.last_signal_timestamp = (
            time.time()
        )


# ==========================================
# MSS / BOS / RETEST CHAIN
# ==========================================

class ExecutionChain:

    @staticmethod
    def detect_mss_after_sweep(
        df,
        sweep_index,
        direction
    ):

        if sweep_index is None:
            return False

        if sweep_index >= len(df) - 1:
            return False

        post_sweep = df.iloc[
            sweep_index + 1:
        ]

        if len(post_sweep) < 2:
            return False

        structure = df.iloc[
            max(0, sweep_index - 10):
            sweep_index
        ]

        if structure.empty:
            return False

        if direction == "LONG":

            reference_high = (
                structure["high"].max()
            )

            return (
                post_sweep["close"].max()
                > reference_high
            )

        if direction == "SHORT":

            reference_low = (
                structure["low"].min()
            )

            return (
                post_sweep["close"].min()
                < reference_low
            )

        return False


    # --------------------------------------
    # Displacement after MSS
    # --------------------------------------

    @staticmethod
    def detect_displacement_after_mss(
        df,
        mss_index,
        direction
    ):

        if mss_index is None:
            return False

        if mss_index >= len(df):
            return False

        for i in range(
            mss_index,
            len(df)
        ):

            if SMCMath.displacement(
                df,
                i
            ):

                candle = df.iloc[i]

                if direction == "LONG":
                    if (
                        candle["close"]
                        > candle["open"]
                    ):
                        return True

                if direction == "SHORT":
                    if (
                        candle["close"]
                        < candle["open"]
                    ):
                        return True

        return False


    # --------------------------------------
    # BOS after displacement
    # --------------------------------------

    @staticmethod
    def detect_bos_after_displacement(
        df,
        displacement_index,
        direction
    ):

        if displacement_index is None:
            return False

        if displacement_index >= len(df) - 1:
            return False

        structure = df.iloc[
            max(
                0,
                displacement_index - 10
            ):
            displacement_index
        ]

        if structure.empty:
            return False

        current = df.iloc[-1]

        if direction == "LONG":

            swing_high = (
                structure["high"].max()
            )

            return (
                current["close"]
                > swing_high
            )

        if direction == "SHORT":

            swing_low = (
                structure["low"].min()
            )

            return (
                current["close"]
                < swing_low
            )

        return False


    # --------------------------------------
    # POI Retest
    # --------------------------------------

    @staticmethod
    def detect_retest(
        current_price,
        poi,
        direction,
        tolerance_pct=0.30
    ):

        if poi is None:
            return False

        poi_low = poi["low"]
        poi_high = poi["high"]

        tolerance = (
            current_price
            * tolerance_pct
            / 100
        )

        if direction == "LONG":

            return (
                current_price
                >= poi_low - tolerance
                and
                current_price
                <= poi_high + tolerance
            )

        if direction == "SHORT":

            return (
                current_price
                >= poi_low - tolerance
                and
                current_price
                <= poi_high + tolerance
            )

        return False


# ==========================================
# REALISTIC RISK ENGINE
# ==========================================

class RiskEngine:

    def __init__(
        self,
        risk_usd=10.0,
        taker_fee=0.0008,
        maker_fee=0.0006,
        slippage=0.0005
    ):

        self.risk_usd = risk_usd

        self.taker_fee = taker_fee

        self.maker_fee = maker_fee

        self.slippage = slippage


    # --------------------------------------
    # Calculate SL / TP
    # --------------------------------------

    def calculate_levels(
        self,
        df,
        direction
    ):

        atr_series = (
            SMCMath.calculate_atr(df)
        )

        atr = atr_series.iloc[-1]

        if pd.isna(atr) or atr <= 0:
            return None

        price = df["close"].iloc[-1]

        recent_high = (
            df["high"]
            .tail(10)
            .max()
        )

        recent_low = (
            df["low"]
            .tail(10)
            .min()
        )

        if direction == "LONG":

            structure_sl = (
                recent_low
                - atr * 0.20
            )

            sl = min(
                structure_sl,
                price - atr * 1.0
            )

            risk_per_unit = (
                price - sl
            )

            tp = (
                price
                + risk_per_unit * 2.5
            )

        elif direction == "SHORT":

            structure_sl = (
                recent_high
                + atr * 0.20
            )

            sl = max(
                structure_sl,
                price + atr * 1.0
            )

            risk_per_unit = (
                sl - price
            )

            tp = (
                price
                - risk_per_unit * 2.5
            )

        else:
            return None

        if risk_per_unit <= 0:
            return None

        return {
            "entry": price,
            "sl": sl,
            "tp": tp,
            "atr": atr,
            "risk_per_unit": risk_per_unit
        }


    # --------------------------------------
    # Realistic Execution
    # --------------------------------------

    def calculate_execution(
        self,
        levels,
        direction
    ):

        if levels is None:
            return {
                "valid": False,
                "reason": "NO RISK LEVELS"
            }

        entry = levels["entry"]

        sl = levels["sl"]

        tp = levels["tp"]

        if direction == "LONG":

            effective_entry = (
                entry
                * (1 + self.slippage)
            )

            effective_sl = (
                sl
                * (1 - self.slippage)
            )

            risk_per_unit = (
                effective_entry
                - effective_sl
            )

            reward_per_unit = (
                tp
                - effective_entry
            )

        elif direction == "SHORT":

            effective_entry = (
                entry
                * (1 - self.slippage)
            )

            effective_sl = (
                sl
                * (1 + self.slippage)
            )

            risk_per_unit = (
                effective_sl
                - effective_entry
            )

            reward_per_unit = (
                effective_entry
                - tp
            )

        else:

            return {
                "valid": False,
                "reason": "INVALID DIRECTION"
            }

        if risk_per_unit <= 0:
            return {
                "valid": False,
                "reason": "INVALID RISK"
            }

        position_size = (
            self.risk_usd
            / risk_per_unit
        )

        entry_fee = (
            effective_entry
            * position_size
            * self.taker_fee
        )

        exit_fee = (
            tp
            * position_size
            * self.maker_fee
        )

        total_fee = (
            entry_fee
            + exit_fee
        )

        gross_reward = (
            reward_per_unit
            * position_size
        )

        net_reward = (
            gross_reward
            - total_fee
        )

        net_risk = (
            self.risk_usd
            + total_fee
        )

        if net_risk <= 0:
            return {
                "valid": False,
                "reason": "INVALID NET RISK"
            }

        real_rr = (
            net_reward
            / net_risk
        )

        return {
            "valid": real_rr >= 2.0,
            "entry": round(
                effective_entry,
                6
            ),
            "sl": round(
                effective_sl,
                6
            ),
            "tp": round(
                tp,
                6
            ),
            "position_size": round(
                position_size,
                6
            ),
            "fees": round(
                total_fee,
                6
            ),
            "risk_usd": round(
                net_risk,
                2
            ),
            "net_reward": round(
                net_reward,
                2
            ),
            "real_rr": round(
                real_rr,
                2
            )
        }


# ==========================================
# COMPOSITE SCORING ENGINE
# ==========================================

class SignalScorer:

    WEIGHTS = {

        "btc_regime": 10,

        "htf_alignment": 15,

        "directional_poi": 10,

        "poi_proximity": 5,

        "liquidity_sweep": 15,

        "mss_choch": 15,

        "displacement": 10,

        "bos_confirmed": 10,

        "retest_confirmed": 5,

        "realistic_rr": 5
    }


    @classmethod
    def calculate(
        cls,
        checks
    ):

        score = 0

        for key, weight in cls.WEIGHTS.items():

            if checks.get(key, False):
                score += weight

        score = min(
            score,
            100
        )

        if score >= 90:

            grade = "A+ SETUP"

        elif score >= 80:

            grade = "A SETUP"

        elif score >= 70:

            grade = "B WATCH"

        elif score >= 60:

            grade = "C WATCH"

        else:

            grade = "NO TRADE"

        return score, grade


# ==========================================
# FINAL EXECUTION GATE
# ==========================================

class FinalExecutionGate:

    REQUIRED = [

        "btc_regime",

        "htf_alignment",

        "directional_poi",

        "poi_proximity",

        "liquidity_sweep",

        "mss_choch",

        "displacement",

        "bos_confirmed",

        "retest_confirmed",

        "realistic_rr"
    ]


    @classmethod
    def validate(
        cls,
        checks
    ):

        missing = []

        for key in cls.REQUIRED:

            if not checks.get(
                key,
                False
            ):

                missing.append(key)

        return {
            "approved": len(missing) == 0,
            "missing": missing
        }
    # ==========================================
# PART 5/5 — LIVE RADAR + SIGNAL ENGINE
# ==========================================

import time
from datetime import datetime, timezone
import plotly.graph_objects as go



# ==========================================
# ADVANCED CONFIRMATION ENGINE
# ==========================================

class AdvancedConfirmationEngine:
    """Non-lookahead technical confirmations used by the final signal gate."""

    @staticmethod
    def adx(df, period=14):
        if df is None or len(df) < period + 5:
            return pd.Series(dtype=float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        up = high.diff()
        down = -low.diff()
        plus_dm = up.where((up > down) & (up > 0), 0.0)
        minus_dm = down.where((down > up) & (down > 0), 0.0)
        atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        plus = plus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        minus = minus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        plus_di = 100 * plus / atr.replace(0, np.nan)
        minus_di = 100 * minus / atr.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        return dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    @staticmethod
    def adx_confirmation(df, direction, period=14, minimum=20.0):
        adx = AdvancedConfirmationEngine.adx(df, period)
        if adx.empty or pd.isna(adx.iloc[-1]):
            return False, np.nan
        value = float(adx.iloc[-1])
        return value >= minimum, value

    @staticmethod
    def cvd(df):
        if df is None or len(df) < 3:
            return pd.Series(dtype=float)
        delta = df["close"].diff().fillna(0.0)
        sign = np.sign(delta)
        signed_volume = df["volume"].astype(float) * sign
        return signed_volume.cumsum()

    @staticmethod
    def cvd_confirmation(df, direction, lookback=20):
        cvd = AdvancedConfirmationEngine.cvd(df)
        if len(cvd) < max(3, lookback + 1):
            return False, 0.0
        slope = float(cvd.iloc[-1] - cvd.iloc[-1-lookback])
        if direction == "LONG":
            return slope > 0, slope
        if direction == "SHORT":
            return slope < 0, slope
        return False, slope

    @staticmethod
    def premium_discount(df, direction, lookback=50):
        if df is None or len(df) < 10:
            return False, "UNKNOWN", np.nan
        window = df.iloc[-min(lookback, len(df)):]
        hi = float(window["high"].max())
        lo = float(window["low"].min())
        if hi <= lo:
            return False, "UNKNOWN", np.nan
        price = float(df["close"].iloc[-1])
        midpoint = (hi + lo) / 2.0
        if price < midpoint:
            zone = "DISCOUNT"
            ok = direction == "LONG"
        elif price > midpoint:
            zone = "PREMIUM"
            ok = direction == "SHORT"
        else:
            zone = "EQUILIBRIUM"
            ok = False
        return ok, zone, midpoint

    @staticmethod
    def active_fvg_confirmation(df, direction, price, max_age=40, proximity_pct=1.5):
        closed = df.iloc[:-1].copy() if len(df) > 3 else df.copy()
        data = SMCMath.detect_fvg(closed)
        pois = data["bullish"] if direction == "LONG" else data["bearish"]
        current_index = len(closed) - 1
        active = []
        for p in pois:
            age = current_index - int(p["index"])
            if age <= max_age:
                mid = (float(p["low"]) + float(p["high"])) / 2.0
                dist = abs(price-mid) / price * 100.0
                if dist <= proximity_pct or (p["low"] <= price <= p["high"]):
                    active.append((dist, p))
        if not active:
            return False, None
        active.sort(key=lambda x: x[0])
        return True, active[0][1]

    @staticmethod
    def active_ob_confirmation(df, direction, price, max_age=60, proximity_pct=1.5):
        closed = df.iloc[:-1].copy() if len(df) > 5 else df.copy()
        obs = POIEngine.detect_order_blocks(closed)
        pois = obs["bullish"] if direction == "LONG" else obs["bearish"]
        current_index = len(closed) - 1
        active = []
        for p in pois:
            age = current_index - int(p["index"])
            if age <= max_age:
                mid = (float(p["low"]) + float(p["high"])) / 2.0
                dist = abs(price-mid) / price * 100.0
                if dist <= proximity_pct or (p["low"] <= price <= p["high"]):
                    active.append((dist, p))
        if not active:
            return False, None
        active.sort(key=lambda x: x[0])
        return True, active[0][1]


def get_btc_regime():
    """Compatibility-safe BTC regime function; fixes the previous NameError."""
    try:
        df_1d = KuCoinLive.get_klines("BTC-USDT", "1day", 100)
        df_4h = KuCoinLive.get_klines("BTC-USDT", "4hour", 100)
        df_1h = KuCoinLive.get_klines("BTC-USDT", "1hour", 100)
        if any(x is None or x.empty for x in (df_1d, df_4h, df_1h)):
            return "CHOP"
        return MarketRegimeEngine.btc_regime(df_1d, df_4h, df_1h)["regime"]
    except Exception:
        return "CHOP"


# ==========================================
# 1. SIGNAL DECISION ENGINE
# ==========================================

class SignalEngine:
    # 15 confirmations = exactly 100 points.
    WEIGHTS = {
        "BTC REGIME": 5,
        "HTF ALIGNMENT": 10,
        "15M POI": 5,
        "POI PROXIMITY": 5,
        "LIQUIDITY SWEEP": 10,
        "MSS / CHOCH": 10,
        "DISPLACEMENT": 10,
        "BOS": 10,
        "POI RETEST": 5,
        "REALISTIC RR >= 2": 5,
        "ADX": 5,
        "FAIR VALUE GAP": 5,
        "ORDER BLOCK": 5,
        "PREMIUM / DISCOUNT": 5,
        "CVD": 5,
    }

    @classmethod
    def evaluate(cls, checks, direction):
        score = sum(weight for key, weight in cls.WEIGHTS.items() if checks.get(key, False))
        score = min(100, int(score))

        # Core market-structure gates. Advanced confirmations add score rather than
        # making every market condition mandatory, which prevents a permanently-zero engine.
        hard_gate = all(checks.get(k, False) for k in [
            "BTC REGIME", "HTF ALIGNMENT", "15M POI", "POI PROXIMITY",
            "LIQUIDITY SWEEP", "MSS / CHOCH", "BOS", "REALISTIC RR >= 2"
        ])

        if hard_gate and score >= 90:
            grade, status = "A+ SETUP", "VALID SIGNAL"
        elif hard_gate and score >= 85:
            grade, status = "A SETUP", "VALID SIGNAL"
        elif checks.get("LIQUIDITY SWEEP") and (
            checks.get("MSS / CHOCH") or checks.get("DISPLACEMENT") or checks.get("BOS")
        ):
            grade, status = "WATCH", "SETUP FORMING"
        else:
            grade, status = "NO TRADE", "NO TRADE"

        return {
            "score": score,
            "grade": grade,
            "status": status,
            "direction": direction,
            "checks": checks,
            "hard_gate": hard_gate,
        }


# ==========================================
# 2. LIVE RADAR
# ==========================================

class CosmicLiveRadar:

    def __init__(self, symbol="SOL-USDT"):

        self.symbol = symbol

        self.last_signal_id = None
        self.last_signal_time = 0

        self.cooldown_seconds = 30 * 60

        self.setup_state = "IDLE"
        self.setup_start_index = None

        self.audit_log = []


    # ======================================
    # COOLDOWN
    # ======================================

    def cooldown_active(self):

        if self.last_signal_time == 0:
            return False

        return (
            time.time() - self.last_signal_time
            < self.cooldown_seconds
        )


    # ======================================
    # AUDIT LOG
    # ======================================

    def log(self, event, score, details):

        self.audit_log.insert(
            0,
            {
                "time": datetime.now(
                    timezone.utc
                ).strftime("%H:%M:%S"),

                "event": event,
                "score": score,
                "details": details
            }
        )

        self.audit_log = self.audit_log[:10]
        try:
            DB.add_audit(
                symbol=self.symbol,
                event=event,
                score=int(score),
                message=details,
                created_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            )
            DB.cleanup_audits(5000)
        except Exception as exc:
            print(f"SQLite audit warning: {exc}")


    # ======================================
    # MAIN ANALYSIS
    # ======================================

    def analyze(self):
        # ----------------------------------
        # FETCH MULTI-TIMEFRAME DATA
        # ----------------------------------
        frames = {
            "1d": KuCoinLive.get_klines(self.symbol, "1day", CONFIG["history_1d"]),
            "4h": KuCoinLive.get_klines(self.symbol, "4hour", CONFIG["history_4h"]),
            "1h": KuCoinLive.get_klines(self.symbol, "1hour", CONFIG["history_1h"]),
            "15m": KuCoinLive.get_klines(self.symbol, "15min", CONFIG["history_15m"]),
            "5m": KuCoinLive.get_klines(self.symbol, "5min", CONFIG["history_5m"]),
        }
        if any(df is None or df.empty for df in frames.values()):
            return {"symbol": self.symbol, "status": "NO TRADE", "grade": "NO TRADE", "score": 0,
                    "reason": "DATA FETCH FAILED", "checks": {}}

        df_1d, df_4h, df_1h, df_15m, df_5m = (frames[k] for k in ("1d", "4h", "1h", "15m", "5m"))
        price = float(df_5m["close"].iloc[-1])

        data_ok, data_reason = DataQualityGuard.validate(df_5m, timeframe_minutes=5)
        if not data_ok:
            return {"symbol": self.symbol, "price": price, "status": "NO TRADE", "grade": "NO TRADE",
                    "score": 0, "reason": data_reason, "checks": {}}

        # ----------------------------------
        # BTC REGIME — fixed NameError path
        # ----------------------------------
        btc_regime = get_btc_regime()
        btc_long_ok = btc_regime in ["BULLISH", "STRONG_BULLISH"]
        btc_short_ok = btc_regime in ["BEARISH", "STRONG_BEARISH"]

        # ----------------------------------
        # HTF REGIME + DIRECTION
        # ----------------------------------
        htf_result = {
            "1d": MarketRegimeEngine.get_trend(df_1d),
            "4h": MarketRegimeEngine.get_trend(df_4h),
            "1h": MarketRegimeEngine.get_trend(df_1h),
        }
        direction_result = DirectionEngine.determine(htf_result, btc_regime)
        direction = direction_result["direction"]
        if direction == "NEUTRAL":
            self.log("NO TRADE", 0, "HTF/BTC direction is not aligned")
            return {
                "symbol": self.symbol,
                "price": round(price, 6),
                "btc_regime": btc_regime,
                "direction": direction,
                "status": "NO TRADE",
                "grade": "NO TRADE",
                "score": 0,
                "checks": {
                    "BTC REGIME": btc_regime != "CHOP",
                    "HTF ALIGNMENT": False
                },
                "hard_gate": False,
                "reason": "HTF/BTC direction is not aligned",
                "htf": htf_result,
                "direction_scores": {
                    "LONG": direction_result.get("long_score", 0),
                    "SHORT": direction_result.get("short_score", 0)
                },
                "trade": None,
                "indicators": {},
                "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            }

        htf_alignment = (
            (direction == "LONG" and all(htf_result[x] == "BULLISH" for x in ("1d", "4h", "1h"))) or
            (direction == "SHORT" and all(htf_result[x] == "BEARISH" for x in ("1d", "4h", "1h")))
        )

        # Work only on closed candles for structure/indicator confirmation.
        c15 = df_15m.iloc[:-1].copy() if len(df_15m) > 20 else df_15m.copy()
        c5 = df_5m.iloc[:-1].copy() if len(df_5m) > 30 else df_5m.copy()
        c15 = SMCMath.detect_swings(c15, left=CONFIG["swing_window"], right=CONFIG["swing_window"])
        c5 = SMCMath.detect_swings(c5, left=CONFIG["swing_window"], right=CONFIG["swing_window"])
        current_idx = len(c5) - 1

        # ----------------------------------
        # 15M POI: FVG + Order Block
        # ----------------------------------
        adv_fvg_ok, fvg = AdvancedConfirmationEngine.active_fvg_confirmation(
            df_15m, direction, price, CONFIG["fvg_max_age"], CONFIG["poi_proximity_percent"]
        )
        adv_ob_ok, order_block = AdvancedConfirmationEngine.active_ob_confirmation(
            df_15m, direction, price, CONFIG["ob_max_age"], CONFIG["poi_proximity_percent"]
        )
        poi_valid = adv_fvg_ok or adv_ob_ok

        # Legacy recent structure is retained as a secondary POI fallback.
        recent_high = float(c15["high"].iloc[-20:].max())
        recent_low = float(c15["low"].iloc[-20:].min())
        dist_low = abs(price-recent_low)/price*100.0
        dist_high = abs(recent_high-price)/price*100.0
        poi_proximity = (direction == "LONG" and dist_low <= CONFIG["poi_proximity_percent"]) or \
                        (direction == "SHORT" and dist_high <= CONFIG["poi_proximity_percent"])
        poi_valid = poi_valid or poi_proximity

        # ----------------------------------
        # LIQUIDITY SWEEP / MSS / BOS
        # ----------------------------------
        liquidity_sweep = False
        mss_confirmed = False
        bos_confirmed = False
        if direction == "LONG":
            swings = c5[c5["swing_low"]]
            if not swings.empty:
                prev_low = float(swings["low"].iloc[-1])
                liquidity_sweep = float(c5["low"].iloc[-1]) < prev_low and float(c5["close"].iloc[-1]) > prev_low
            if liquidity_sweep:
                prev_high = float(c5["high"].iloc[-10:-1].max())
                mss_confirmed = price > prev_high
                if mss_confirmed:
                    structure_high = float(c5["high"].iloc[-6:-1].max())
                    bos_confirmed = price > structure_high
        else:
            swings = c5[c5["swing_high"]]
            if not swings.empty:
                prev_high = float(swings["high"].iloc[-1])
                liquidity_sweep = float(c5["high"].iloc[-1]) > prev_high and float(c5["close"].iloc[-1]) < prev_high
            if liquidity_sweep:
                prev_low = float(c5["low"].iloc[-10:-1].min())
                mss_confirmed = price < prev_low
                if mss_confirmed:
                    structure_low = float(c5["low"].iloc[-6:-1].min())
                    bos_confirmed = price < structure_low

        displacement = SMCMath.displacement(c5, current_idx)

        # ----------------------------------
        # RETEST
        # ----------------------------------
        retest_confirmed = False
        atr_series = SMCMath.calculate_atr(c5)
        atr = float(atr_series.iloc[-1]) if not atr_series.empty and pd.notna(atr_series.iloc[-1]) else np.nan
        if bos_confirmed and np.isfinite(atr):
            structure_level = float(c5["high"].iloc[-6:-1].max()) if direction == "LONG" else float(c5["low"].iloc[-6:-1].min())
            retest_confirmed = abs(price-structure_level) <= atr * 0.5

        # ----------------------------------
        # ADVANCED CONFIRMATIONS
        # ----------------------------------
        adx_ok, adx_value = AdvancedConfirmationEngine.adx_confirmation(
            c5, direction, CONFIG["adx_period"], CONFIG["adx_min"]
        )
        pd_ok, pd_zone, midpoint = AdvancedConfirmationEngine.premium_discount(
            c15, direction, CONFIG["premium_discount_lookback"]
        )
        cvd_ok, cvd_slope = AdvancedConfirmationEngine.cvd_confirmation(
            c5, direction, CONFIG["cvd_lookback"]
        )

        # ----------------------------------
        # TRADE LEVELS
        # ----------------------------------
        if direction == "LONG":
            swing_sl = float(c5["low"].iloc[-20:].min())
            sl = min(swing_sl, price * 0.995)
            tp = price + (price-sl) * 2.5
        else:
            swing_sl = float(c5["high"].iloc[-20:].max())
            sl = max(swing_sl, price * 1.005)
            tp = price - (sl-price) * 2.5

        execution = RiskEngine(
            risk_usd=CONFIG["account_risk_usd"],
            taker_fee=CONFIG["taker_fee"],
            maker_fee=CONFIG["maker_fee"],
            slippage=CONFIG["slippage"]
        )
        levels = {"entry": price, "sl": sl, "tp": tp}
        trade = execution.calculate_execution(levels, direction)
        real_rr = float(trade.get("real_rr", 0)) if isinstance(trade, dict) else 0.0

        checks = {
            "BTC REGIME": btc_long_ok if direction == "LONG" else btc_short_ok,
            "HTF ALIGNMENT": htf_alignment,
            "15M POI": poi_valid,
            "POI PROXIMITY": poi_proximity,
            "LIQUIDITY SWEEP": liquidity_sweep,
            "MSS / CHOCH": mss_confirmed,
            "DISPLACEMENT": bool(displacement),
            "BOS": bos_confirmed,
            "POI RETEST": retest_confirmed,
            "REALISTIC RR >= 2": real_rr >= 2.0,
            "ADX": adx_ok,
            "FAIR VALUE GAP": adv_fvg_ok,
            "ORDER BLOCK": adv_ob_ok,
            "PREMIUM / DISCOUNT": pd_ok,
            "CVD": cvd_ok,
        }

        result = SignalEngine.evaluate(checks, direction)
        signal_id = f"{self.symbol}_{direction}_{df_5m['timestamp'].iloc[-1]}_{result['grade']}"

        if result["status"] == "VALID SIGNAL":
            if self.cooldown_active():
                result["status"] = "COOLDOWN"
            elif signal_id == self.last_signal_id:
                result["status"] = "DUPLICATE BLOCKED"
            else:
                self.last_signal_id = signal_id
                self.last_signal_time = time.time()
                self.log("VALID SIGNAL", result["score"], f"{direction} | {result['grade']} | RR 1:{real_rr}")
        elif result["status"] == "SETUP FORMING":
            self.log("SETUP FORMING", result["score"], f"{direction} | advanced confirmations pending")
        else:
            self.log("NO TRADE", result["score"], f"{direction} | core/advanced gate incomplete")

        return {
            "symbol": self.symbol,
            "price": round(price, 6),
            "btc_regime": btc_regime,
            "direction": direction,
            "htf": htf_result,
            "direction_scores": {
                "LONG": direction_result.get("long_score", 0),
                "SHORT": direction_result.get("short_score", 0)
            },
            "status": result["status"],
            "grade": result["grade"],
            "score": result["score"],
            "checks": result["checks"],
            "hard_gate": result["hard_gate"],
            "indicators": {
                "ADX": round(float(adx_value), 2) if np.isfinite(adx_value) else None,
                "CVD_SLOPE": round(float(cvd_slope), 4),
                "PD_ZONE": pd_zone,
                "MIDPOINT": round(float(midpoint), 6) if np.isfinite(midpoint) else None,
                "FVG": fvg,
                "ORDER_BLOCK": order_block,
            },
            "trade": trade if trade.get("valid") else None,
            "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        }


# ==========================================
# STREAMLIT DASHBOARD — V3.3 ADVANCED RADAR
# ==========================================

st.set_page_config(
    page_title="COSMIC 108 V3.3 — Advanced SMC Radar",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: .8rem; padding-bottom: 1rem; max-width: 1600px;}
[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); border-radius: 12px; padding: 10px 12px; background: rgba(128,128,128,.035);}
.cosmic-title {font-size: 2rem; font-weight: 800; letter-spacing: -.03em; margin-bottom: 0;}
.cosmic-sub {opacity:.70; margin-top:0;}
.status-card {border:1px solid rgba(128,128,128,.25); border-radius:14px; padding:16px; min-height:120px;}
.good {color:#39d98a; font-weight:700;}
.bad {color:#ff5c66; font-weight:700;}
.warn {color:#f6c945; font-weight:700;}
.muted {opacity:.65;}
.section {font-size:1.05rem; font-weight:750; margin: .4rem 0 .6rem 0;}
.small {font-size:.82rem; opacity:.72;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_radar(symbol):
    return CosmicLiveRadar(symbol)

# Streamlit source is executed top-to-bottom; keep cached radar instances per symbol.
def run_scan(symbol):
    radar = get_radar(symbol)
    result = radar.analyze()
    return result, radar.audit_log

def badge(ok):
    return "✅ CONFIRMED" if ok else "❌ FAILED"

def pct_score(score):
    return max(0, min(100, int(score or 0)))

def render_check_table(checks):
    weights = SignalEngine.WEIGHTS
    rows = []
    for name, weight in weights.items():
        ok = bool(checks.get(name, False))
        rows.append({
            "Confirmation": name,
            "Weight": f"+{weight}",
            "Status": badge(ok),
            "Points": weight if ok else 0,
        })
    return pd.DataFrame(rows)

def render_chart(df, symbol, direction):
    if df is None or df.empty:
        st.info("No chart data available.")
        return
    view = df.tail(120).copy()
    fig = go.Figure(data=[go.Candlestick(
        x=pd.to_datetime(view["timestamp"], unit="s"),
        open=view["open"], high=view["high"], low=view["low"], close=view["close"],
        name=symbol
    )])
    fig.update_layout(
        height=460,
        margin=dict(l=10,r=10,t=25,b=10),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        title=f"{symbol} • 5M Market Structure • {direction}",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def scan_all_symbols():
    results = []
    for sym in CONFIG["symbols"]:
        try:
            r, _ = run_scan(sym)
            results.append(r)
        except Exception as exc:
            results.append({"symbol": sym, "status":"ENGINE ERROR", "grade":"ERROR", "score":0, "reason":str(exc), "checks":{}})
    return results

st.markdown('<div class="cosmic-title">⚡ COSMIC 108 V3.3 — Advanced SMC Radar</div>', unsafe_allow_html=True)
st.markdown('<div class="cosmic-sub">KuCoin • SQLite • Multi-Timeframe SMC • ADX • FVG • Order Block • Premium/Discount • CVD • Risk Engine</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("🎛️ Radar Control")
    symbol = st.selectbox("Symbol", CONFIG["symbols"], index=0)
    auto = st.checkbox("Auto refresh", value=False)
    refresh = st.slider("Refresh seconds", 15, 120, CONFIG["refresh_seconds"])
    st.divider()
    st.caption("ENGINE SETTINGS")
    st.write(f"ADX minimum: **{CONFIG['adx_min']:.1f}**")
    st.write(f"Signal minimum: **{CONFIG['signal_min_score']}/100**")
    st.write(f"POI proximity: **{CONFIG['poi_proximity_percent']:.2f}%**")
    st.write(f"Cooldown: **{CONFIG['cooldown_seconds']//60} min**")
    st.write(f"Risk / signal: **${CONFIG['account_risk_usd']:.2f}**")
    st.divider()
    st.caption("STORAGE")
    st.code(CONFIG["database_path"], language="text")
    if st.button("🧹 Clear cached radar state"):
        get_radar.clear()
        st.rerun()

control1, control2, control3 = st.columns([1,1,2])
with control1:
    scan_now = st.button("🔄 Scan Now", type="primary", use_container_width=True)
with control2:
    scan_all = st.button("📡 Scan All Coins", use_container_width=True)
with control3:
    st.caption("The engine only evaluates closed candles for structure confirmations.")

# Full market overview
if scan_all:
    st.session_state["overview_results"] = scan_all_symbols()

if "overview_results" not in st.session_state:
    st.session_state["overview_results"] = []

if st.session_state["overview_results"]:
    st.markdown('<div class="section">📊 Market Overview</div>', unsafe_allow_html=True)
    overview_rows = []
    for r in st.session_state["overview_results"]:
        overview_rows.append({
            "Symbol": r.get("symbol","—"),
            "Direction": r.get("direction","—"),
            "BTC": r.get("btc_regime","—"),
            "Score": r.get("score",0),
            "Grade": r.get("grade","—"),
            "Status": r.get("status","—"),
            "Reason": r.get("reason", ""),
        })
    st.dataframe(pd.DataFrame(overview_rows), use_container_width=True, hide_index=True)

scan_key = symbol
needs_scan = (
    scan_now
    or "current_result" not in st.session_state
    or st.session_state.get("current_symbol") != scan_key
    or auto
)

if needs_scan:
    try:
        result, audit_log = run_scan(symbol)
        st.session_state["current_result"] = result
        st.session_state["current_audit"] = audit_log
        st.session_state["current_symbol"] = scan_key
    except Exception as exc:
        st.error(f"ENGINE ERROR: {type(exc).__name__}: {exc}")
        st.stop()
else:
    result = st.session_state["current_result"]
    audit_log = st.session_state.get("current_audit", [])

if not result.get("checks"):
    # Still show useful diagnostic data for neutral/blocked states.
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("PAIR", result.get("symbol", symbol))
    m2.metric("PRICE", result.get("price", "—"))
    m3.metric("BTC REGIME", result.get("btc_regime", "—"))
    m4.metric("LONG SCORE", (result.get("direction_scores") or {}).get("LONG", 0))
    m5.metric("SHORT SCORE", (result.get("direction_scores") or {}).get("SHORT", 0))
    reason = result.get("reason", "No executable setup")
    if "not aligned" in reason.lower():
        st.warning(f"⏸️ NO TRADE — {reason}")
    else:
        st.info(f"ℹ️ {reason}")
    htf = result.get("htf", {})
    st.markdown('<div class="section">🧭 Higher-Timeframe Map</div>', unsafe_allow_html=True)
    hc = st.columns(3)
    for col, tf in zip(hc, ["1d","4h","1h"]):
        col.metric(tf.upper(), htf.get(tf, "UNKNOWN"))
    st.stop()

checks = result.get("checks", {})
ind = result.get("indicators", {})
trade = result.get("trade") or {}
score = pct_score(result.get("score",0))

# Header metrics
m = st.columns(6)
m[0].metric("PAIR", result.get("symbol", symbol))
m[1].metric("PRICE", result.get("price", "—"))
m[2].metric("DIRECTION", result.get("direction", "—"))
m[3].metric("BTC REGIME", result.get("btc_regime", "—"))
m[4].metric("SCORE", f"{score}/100")
m[5].metric("STATUS", result.get("status", "—"))

st.progress(score, text=f"Signal quality: {score}/100 • Minimum executable score: {CONFIG['signal_min_score']}/100")

# Status banner
status = result.get("status", "NO TRADE")
if status == "VALID SIGNAL":
    st.success(f"🚨 {result.get('grade')} — {result.get('direction')} signal passed the execution gate")
elif status == "SETUP FORMING":
    st.warning(f"👀 WATCH — {result.get('direction')} setup is forming; confirmation chain is incomplete")
elif status == "COOLDOWN":
    st.warning("⏳ COOLDOWN — duplicate/new signal temporarily blocked")
else:
    st.info(f"⏸️ {result.get('grade','NO TRADE')} — wait for the missing confirmations")

# Main tabs
tab_overview, tab_structure, tab_advanced, tab_risk, tab_chart, tab_audit = st.tabs([
    "🧠 Overview", "🏗️ SMC Structure", "📐 Advanced Filters", "💰 Risk & Trade", "📈 Chart", "🧾 Audit"
])

with tab_overview:
    left,right = st.columns([1.15,.85])
    with left:
        st.markdown('<div class="section">Confirmation Matrix</div>', unsafe_allow_html=True)
        df_checks = render_check_table(checks)
        st.dataframe(df_checks, use_container_width=True, hide_index=True, height=520)
    with right:
        st.markdown('<div class="section">HTF Alignment</div>', unsafe_allow_html=True)
        htf = result.get("htf", {})
        hcols = st.columns(3)
        for col, tf in zip(hcols,["1d","4h","1h"]):
            col.metric(tf.upper(), htf.get(tf,"UNKNOWN"))
        st.markdown('<div class="section">Direction Pressure</div>', unsafe_allow_html=True)
        ds = result.get("direction_scores", {})
        st.metric("LONG", ds.get("LONG",0))
        st.metric("SHORT", ds.get("SHORT",0))
        st.markdown('<div class="section">Decision</div>', unsafe_allow_html=True)
        st.write(f"Hard gate: **{'PASSED' if result.get('hard_gate') else 'NOT PASSED'}**")
        st.write(f"Grade: **{result.get('grade','—')}**")
        st.write(f"Reason: **{result.get('reason','Core/advanced gate evaluation')}**")

with tab_structure:
    s1,s2,s3,s4 = st.columns(4)
    s1.metric("Liquidity Sweep", badge(checks.get("LIQUIDITY SWEEP")))
    s2.metric("MSS / CHOCH", badge(checks.get("MSS / CHOCH")))
    s3.metric("Displacement", badge(checks.get("DISPLACEMENT")))
    s4.metric("BOS", badge(checks.get("BOS")))
    s5,s6,s7,s8 = st.columns(4)
    s5.metric("15M POI", badge(checks.get("15M POI")))
    s6.metric("POI Proximity", badge(checks.get("POI PROXIMITY")))
    s7.metric("POI Retest", badge(checks.get("POI RETEST")))
    s8.metric("RR ≥ 2", badge(checks.get("REALISTIC RR >= 2")))
    st.markdown('<div class="section">POI Details</div>', unsafe_allow_html=True)
    fvg = ind.get("FVG")
    ob = ind.get("ORDER_BLOCK")
    poi_df = pd.DataFrame([
        {"Type":"FVG", "Active": bool(fvg), "Low": (fvg or {}).get("low") if isinstance(fvg,dict) else None, "High": (fvg or {}).get("high") if isinstance(fvg,dict) else None},
        {"Type":"Order Block", "Active": bool(ob), "Low": (ob or {}).get("low") if isinstance(ob,dict) else None, "High": (ob or {}).get("high") if isinstance(ob,dict) else None},
    ])
    st.dataframe(poi_df, use_container_width=True, hide_index=True)

with tab_advanced:
    a1,a2,a3,a4 = st.columns(4)
    a1.metric("ADX", ind.get("ADX", "—"))
    a2.metric("CVD Slope", ind.get("CVD_SLOPE", "—"))
    a3.metric("P/D Zone", ind.get("PD_ZONE", "—"))
    a4.metric("Midpoint", ind.get("MIDPOINT", "—"))
    st.markdown('<div class="section">Advanced Confirmation Status</div>', unsafe_allow_html=True)
    adv_df = pd.DataFrame([
        {"Filter":"ADX", "Value":ind.get("ADX","—"), "Status":badge(checks.get("ADX"))},
        {"Filter":"Fair Value Gap", "Value":"ACTIVE" if ind.get("FVG") else "NONE", "Status":badge(checks.get("FAIR VALUE GAP"))},
        {"Filter":"Order Block", "Value":"ACTIVE" if ind.get("ORDER_BLOCK") else "NONE", "Status":badge(checks.get("ORDER BLOCK"))},
        {"Filter":"Premium / Discount", "Value":ind.get("PD_ZONE","—"), "Status":badge(checks.get("PREMIUM / DISCOUNT"))},
        {"Filter":"CVD", "Value":ind.get("CVD_SLOPE","—"), "Status":badge(checks.get("CVD"))},
    ])
    st.dataframe(adv_df, use_container_width=True, hide_index=True)

with tab_risk:
    if trade:
        r1,r2,r3,r4,r5 = st.columns(5)
        r1.metric("ENTRY", trade.get("entry"))
        r2.metric("STOP LOSS", trade.get("sl"))
        r3.metric("TAKE PROFIT", trade.get("tp"))
        r4.metric("REAL RR", f"1:{trade.get('real_rr','—')}")
        r5.metric("POSITION", trade.get("position_size"))
        st.success(f"Net risk ≈ ${trade.get('risk_usd','—')} • Net reward ≈ ${trade.get('net_reward','—')} • Fees ≈ ${trade.get('fees','—')}")
    else:
        st.info("No executable trade parameters because the final gate is not approved.")
    st.caption("Risk values are calculation outputs, not an instruction to place a trade.")

with tab_chart:
    # Fetch only the selected symbol's 5M chart for visualization.
    try:
        chart_df = KuCoinLive.get_klines(symbol, "5min", CONFIG["history_5m"])
        render_chart(chart_df, symbol, result.get("direction","NEUTRAL"))
    except Exception as exc:
        st.error(f"Chart error: {exc}")

with tab_audit:
    if audit_log:
        st.dataframe(pd.DataFrame(audit_log), use_container_width=True, hide_index=True, height=420)
    else:
        st.info("No audit events yet.")
    st.caption(f"Last scan: {result.get('time','—')} • SQLite persistence enabled")

if auto:
    time.sleep(max(15, int(refresh)))
    st.rerun()
