import time
import sqlite3
from pathlib import Path
import requests
import pandas as pd
import numpy as np

from datetime import datetime, timezone
from collections import deque

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================
# COSMIC 108 V3.0
# PART 1/5 — FOUNDATION + LIVE DATA + DATA QUALITY
# ============================================================

APP_NAME = "COSMIC 108 V3.0"
VERSION = "3.0"

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

        console.print(
            f"[red]KuCoin API Error "
            f"{symbol} {interval}: "
            f"{last_error}[/red]"
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
            console.print(
                f"[yellow]SQLite audit persistence warning: {exc}[/yellow]"
            )

    def latest(self, limit=10):

        return list(
            self.records
        )[:limit]



# ============================================================
# BASIC CONNECTION TEST
# ============================================================

def test_market_connection():

    console.print(
        Panel(
            "[bold cyan]"
            f"{APP_NAME} {VERSION}"
            "[/bold cyan]\n"
            "Testing KuCoin public market connection..."
        )
    )

    df = KuCoinLive.get_klines(
        "BTC-USDT",
        "5min",
        10
    )

    if df.empty:

        console.print(
            "[bold red]"
            "❌ KuCoin connection/data test failed"
            "[/bold red]"
        )

        return False

    console.print(
        "[bold green]"
        "✅ KuCoin live market connection OK"
        "[/bold green]"
    )

    console.print(
        f"BTC-USDT candles received: {len(df)}"
    )

    return True


# ============================================================
# PART 1 TEST
# ============================================================
# Startup connectivity is executed only from the final entrypoint.
# ==========================================
# PART 2 — DATA QUALITY + SMC MATH ENGINE
# ==========================================

class DataQualityGuard:
    """Compatibility-safe data quality validator."""

    TIMEFRAME_MINUTES = {
        "1min": 1,
        "5min": 5,
        "15min": 15,
        "1hour": 60,
        "4hour": 240,
        "1day": 1440,
    }

    @staticmethod
    def validate(
        df,
        interval=None,
        timeframe_minutes=None,
        time_frame_min=None,
        min_candles=50
    ):
        # Accept every legacy spelling used by this codebase.
        if timeframe_minutes is None and time_frame_min is not None:
            timeframe_minutes = time_frame_min
        if timeframe_minutes is None and isinstance(interval, str):
            timeframe_minutes = DataQualityGuard.TIMEFRAME_MINUTES.get(interval)
        if timeframe_minutes is None:
            timeframe_minutes = 5
        try:
            timeframe_minutes = int(timeframe_minutes)
        except (TypeError, ValueError):
            return False, f"INVALID TIMEFRAME: {timeframe_minutes!r}"
        if timeframe_minutes <= 0:
            return False, "INVALID TIMEFRAME: must be positive"

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
# GLOBAL BTC REGIME HELPER
# ==========================================
# The radar calls get_btc_regime() without arguments.
# Keep BTC regime calculation centralized and always use BTC/USDT
# independent of the altcoin currently being scanned.
# A short cache prevents 3 BTC API requests for every symbol on every loop.
_BTC_REGIME_CACHE = {
    "regime": "CHOP",
    "timestamp": 0.0,
}
_BTC_REGIME_CACHE_SECONDS = 30


def get_btc_regime():
    now = time.time()
    if now - _BTC_REGIME_CACHE["timestamp"] < _BTC_REGIME_CACHE_SECONDS:
        return _BTC_REGIME_CACHE["regime"]

    try:
        btc_1d = KuCoinLive.get_klines("BTC-USDT", "1day", 100)
        btc_4h = KuCoinLive.get_klines("BTC-USDT", "4hour", 100)
        btc_1h = KuCoinLive.get_klines("BTC-USDT", "1hour", 100)

        if any(df is None or df.empty for df in (btc_1d, btc_4h, btc_1h)):
            return "CHOP"

        result = MarketRegimeEngine.btc_regime(
            btc_1d,
            btc_4h,
            btc_1h
        )

        regime = result.get("regime", "CHOP")
        _BTC_REGIME_CACHE["regime"] = regime
        _BTC_REGIME_CACHE["timestamp"] = now
        return regime
    except Exception as exc:
        console.print(
            f"[yellow]BTC regime unavailable: {exc}[/yellow]"
        )
        # Fail closed: never allow an unavailable BTC regime to approve a trade.
        return "CHOP"


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
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# ==========================================
# 1. SIGNAL DECISION ENGINE
# ==========================================

class SignalEngine:

    @staticmethod
    def evaluate(
        btc_regime,
        htf_alignment,
        poi_valid,
        poi_proximity,
        liquidity_sweep,
        mss_confirmed,
        displacement,
        bos_confirmed,
        retest_confirmed,
        real_rr
    ):

        checks = {
            "BTC REGIME": btc_regime,
            "HTF ALIGNMENT": htf_alignment,
            "15M POI": poi_valid,
            "POI PROXIMITY": poi_proximity,
            "LIQUIDITY SWEEP": liquidity_sweep,
            "MSS / CHOCH": mss_confirmed,
            "DISPLACEMENT": displacement,
            "BOS": bos_confirmed,
            "POI RETEST": retest_confirmed,
            "REALISTIC RR >= 2": real_rr >= 2.0
        }

        weights = {
            "BTC REGIME": 5,
            "HTF ALIGNMENT": 15,
            "15M POI": 10,
            "POI PROXIMITY": 5,
            "LIQUIDITY SWEEP": 15,
            "MSS / CHOCH": 15,
            "DISPLACEMENT": 15,
            "BOS": 10,
            "POI RETEST": 5,
            "REALISTIC RR >= 2": 5
        }

        score = 0

        for key, passed in checks.items():
            if passed:
                score += weights[key]

        # ----------------------------------
        # STRICT EXECUTION GATE
        # ----------------------------------

        mandatory = all(checks.values())

        if mandatory and score >= 90:
            grade = "A+ SETUP"
            status = "VALID SIGNAL"

        elif mandatory and score >= 85:
            grade = "A SETUP"
            status = "VALID SIGNAL"

        elif liquidity_sweep and (
            mss_confirmed or displacement or bos_confirmed
        ):
            grade = "WATCH"
            status = "SETUP FORMING"

        else:
            grade = "NO TRADE"
            status = "NO TRADE"

        return {
            "score": score,
            "grade": grade,
            "status": status,
            "checks": checks
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


    # ======================================
    # MAIN ANALYSIS
    # ======================================

    def analyze(self):

        # ----------------------------------
        # FETCH MULTI TIMEFRAME DATA
        # ----------------------------------

        df_1d = KuCoinLive.get_klines(
            self.symbol,
            "1day",
            100
        )

        df_4h = KuCoinLive.get_klines(
            self.symbol,
            "4hour",
            100
        )

        df_1h = KuCoinLive.get_klines(
            self.symbol,
            "1hour",
            100
        )

        df_15m = KuCoinLive.get_klines(
            self.symbol,
            "15min",
            150
        )

        df_5m = KuCoinLive.get_klines(
            self.symbol,
            "5min",
            200
        )


        # ----------------------------------
        # DATA VALIDATION
        # ----------------------------------

        if any(
            df.empty
            for df in [
                df_1d,
                df_4h,
                df_1h,
                df_15m,
                df_5m
            ]
        ):

            return {
                "status": "NO TRADE",
                "reason": "DATA FETCH FAILED"
            }


        # ----------------------------------
        # CURRENT PRICE
        # ----------------------------------

        price = float(
            df_5m["close"].iloc[-1]
        )


        # ----------------------------------
        # DATA QUALITY
        # ----------------------------------

        data_ok, data_reason = (
            DataQualityGuard.validate(
                df_5m,
                timeframe_minutes=5
            )
        )

        if not data_ok:

            return {
                "symbol": self.symbol,
                "price": price,
                "status": "NO TRADE",
                "grade": "NO TRADE",
                "score": 0,
                "reason": data_reason,
                "checks": {}
            }


        # ----------------------------------
        # BTC REGIME
        # ----------------------------------

        btc_regime = get_btc_regime()

        btc_long_ok = btc_regime in [
            "STRONG_BULLISH"
        ]


        # ----------------------------------
        # HTF ALIGNMENT
        # ----------------------------------

        htf_1d = (
            df_1d["close"].iloc[-1]
            >
            df_1d["close"]
            .ewm(span=20)
            .mean()
            .iloc[-1]
        )

        htf_4h = (
            df_4h["close"].iloc[-1]
            >
            df_4h["close"]
            .ewm(span=20)
            .mean()
            .iloc[-1]
        )

        htf_1h = (
            df_1h["close"].iloc[-1]
            >
            df_1h["close"]
            .ewm(span=20)
            .mean()
            .iloc[-1]
        )

        htf_alignment = (
            htf_1d
            and htf_4h
            and htf_1h
        )


        # ----------------------------------
        # 15M POI
        # ----------------------------------

        df_15m = SMCMath.get_swings(
            df_15m.copy()
        )

        recent_high = (
            df_15m["high"]
            .iloc[-20:-1]
            .max()
        )

        recent_low = (
            df_15m["low"]
            .iloc[-20:-1]
            .min()
        )


        # Basic directional POI proximity

        distance_to_low = (
            abs(price - recent_low)
            / price
            * 100
        )

        distance_to_high = (
            abs(recent_high - price)
            / price
            * 100
        )

        poi_proximity = (
            distance_to_low <= 1.5
        )

        poi_valid = (
            price >= recent_low
        )


        # ----------------------------------
        # 5M STRUCTURE
        # ----------------------------------

        df_5m = SMCMath.get_swings(
            df_5m.copy()
        )

        current_idx = len(df_5m) - 1


        # ----------------------------------
        # LIQUIDITY SWEEP
        # ----------------------------------

        swing_lows = df_5m[
            df_5m["swing_low"]
        ]["low"]

        liquidity_sweep = False

        if not swing_lows.empty:

            previous_swing_low = (
                swing_lows.iloc[-1]
            )

            current_low = (
                df_5m["low"].iloc[-1]
            )

            current_close = (
                df_5m["close"].iloc[-1]
            )

            liquidity_sweep = (
                current_low < previous_swing_low
                and
                current_close > previous_swing_low
            )


        # ----------------------------------
        # DISPLACEMENT
        # ----------------------------------

        displacement = (
            SMCMath.check_displacement(
                df_5m,
                current_idx
            )
        )


        # ----------------------------------
        # CAUSAL MSS
        # ----------------------------------

        mss_confirmed = False

        if liquidity_sweep:

            previous_high = (
                df_5m["high"]
                .iloc[-10:-1]
                .max()
            )

            mss_confirmed = (
                price > previous_high
            )


        # ----------------------------------
        # BOS
        # ----------------------------------

        bos_confirmed = False

        if mss_confirmed:

            structure_high = (
                df_5m["high"]
                .iloc[-6:-1]
                .max()
            )

            bos_confirmed = (
                price > structure_high
            )


        # ----------------------------------
        # RETEST
        # ----------------------------------

        retest_confirmed = False

        if bos_confirmed:

            atr_series = (
                SMCMath.calculate_atr(
                    df_5m
                )
            )

            atr = atr_series.iloc[-1]

            if pd.notna(atr):

                retest_distance = abs(
                    price - structure_high
                )

                retest_confirmed = (
                    retest_distance
                    <= atr * 0.5
                )


        # ----------------------------------
        # TRADE PARAMETERS
        # ----------------------------------

        entry = price

        sl = min(
            recent_low,
            price * 0.995
        )

        tp = price + (
            (price - sl) * 2.5
        )


        execution = ExecutionEngine(
            account_risk_usd=10.0
        )

        trade = execution.calculate_trade_parameters(
            entry,
            sl,
            tp,
            "LONG"
        )

        real_rr = trade.get(
            "real_rr",
            0
        )


        # ----------------------------------
        # FINAL SIGNAL SCORING
        # ----------------------------------

        result = SignalEngine.evaluate(

            btc_regime=btc_long_ok,

            htf_alignment=htf_alignment,

            poi_valid=poi_valid,

            poi_proximity=poi_proximity,

            liquidity_sweep=liquidity_sweep,

            mss_confirmed=mss_confirmed,

            displacement=displacement,

            bos_confirmed=bos_confirmed,

            retest_confirmed=retest_confirmed,

            real_rr=real_rr
        )


        # ----------------------------------
        # SIGNAL ID
        # ----------------------------------

        signal_id = (
            f"{self.symbol}_"
            f"{df_5m['timestamp'].iloc[-1]}_"
            f"{result['grade']}"
        )


        # ----------------------------------
        # DUPLICATE PROTECTION
        # ----------------------------------

        valid_signal = (
            result["status"]
            == "VALID SIGNAL"
        )

        if valid_signal:

            if self.cooldown_active():

                result["status"] = (
                    "COOLDOWN"
                )

            elif (
                signal_id
                == self.last_signal_id
            ):

                result["status"] = (
                    "DUPLICATE BLOCKED"
                )

            else:

                self.last_signal_id = (
                    signal_id
                )

                self.last_signal_time = (
                    time.time()
                )

                self.log(
                    "VALID SIGNAL",
                    result["score"],
                    f"{result['grade']} | "
                    f"RR 1:{real_rr}"
                )

        elif result["grade"] == "WATCH":

            self.log(
                "SETUP FORMING",
                result["score"],
                "SMC chain not completed"
            )

        else:

            self.log(
                "NO TRADE",
                result["score"],
                result.get(
                    "reason",
                    "Mandatory gate failed"
                )
            )


        # ----------------------------------
        # FINAL RESULT
        # ----------------------------------

        return {

            "symbol": self.symbol,

            "price": round(
                price,
                4
            ),

            "btc_regime": btc_regime,

            "status": result[
                "status"
            ],

            "grade": result[
                "grade"
            ],

            "score": result[
                "score"
            ],

            "checks": result[
                "checks"
            ],

            "trade": (
                trade
                if trade.get("valid")
                else None
            ),

            "time": datetime.now(
                timezone.utc
            ).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
        }


# ==========================================
# 3. TERMINAL DASHBOARD
# ==========================================

def render_radar(
    result,
    audit_log
):

    console.clear()

    if "checks" not in result:

        console.print(
            Panel(
                str(result),
                title="COSMIC 108 V3.0"
            )
        )

        return


    checks = result["checks"]

    table = Table(
        title=(
            "COSMIC 108 V3.0 "
            "— LIVE SMC RADAR"
        )
    )

    table.add_column(
        "Validation",
        style="cyan"
    )

    table.add_column(
        "Status",
        justify="center"
    )


    for key, passed in checks.items():

        if passed:

            table.add_row(
                key,
                "[green]✅ CONFIRMED[/green]"
            )

        else:

            table.add_row(
                key,
                "[red]❌ FAILED[/red]"
            )


    trade = result.get(
        "trade"
    )

    trade_text = ""

    if trade:

        trade_text = (
            f"\nENTRY: {trade['entry']}"
            f"\nSL: {trade['sl']}"
            f"\nTP: {trade['tp']}"
            f"\nREAL RR: 1:{trade['real_rr']}"
            f"\nPOSITION: {trade['position_size']}"
        )


    main_text = (
        f"[bold cyan]PAIR:[/bold cyan] "
        f"{result['symbol']}\n"

        f"[bold yellow]PRICE:[/bold yellow] "
        f"{result['price']}\n"

        f"[bold cyan]BTC REGIME:[/bold cyan] "
        f"{result['btc_regime']}\n"

        f"[bold white]SCORE:[/bold white] "
        f"{result['score']}/100\n"

        f"[bold white]GRADE:[/bold white] "
        f"{result['grade']}\n"

        f"[bold white]STATUS:[/bold white] "
        f"{result['status']}\n"

        f"{trade_text}"
    )


    console.print(
        Panel(
            main_text,
            title=(
                "[bold cyan]"
                "COSMIC 108 V3.0"
                "[/bold cyan]"
            ),
            expand=False
        )
    )

    console.print(table)


    # --------------------------------------
    # AUDIT LOG
    # --------------------------------------

    log_table = Table(
        title="LIVE AUDIT LOG"
    )

    log_table.add_column("TIME")
    log_table.add_column("EVENT")
    log_table.add_column("SCORE")
    log_table.add_column("DETAILS")


    for item in audit_log:

        log_table.add_row(
            item["time"],
            item["event"],
            str(item["score"]),
            item["details"]
        )


    console.print(log_table)


# ==========================================
# 4. LIVE RUNNER
# ==========================================

def run_cosmic_radar():

    radar = CosmicLiveRadar(
        "SOL-USDT"
    )

    console.print(
        Panel(
            "[bold cyan]"
            "COSMIC 108 V3.0"
            "[/bold cyan]\n"
            "Starting LIVE SMC Radar...\n"
            "Connecting to KuCoin...",
            title="SYSTEM START"
        )
    )

    time.sleep(2)


    while True:

        try:

            result = radar.analyze()

            render_radar(
                result,
                radar.audit_log
            )

            console.print(
                "\n[dim]"
                "Next scan in 30 seconds..."
                "[/dim]"
            )

            time.sleep(30)


        except KeyboardInterrupt:

            console.print(
                "\n[bold red]"
                "COSMIC RADAR STOPPED"
                "[/bold red]"
            )

            break


        except Exception as e:

            console.print(
                Panel(
                    f"[red]"
                    f"ENGINE ERROR: {e}"
                    f"[/red]",
                    title="ERROR"
                )
            )

            time.sleep(10)


# ==========================================
# PART 5 — RADAR DEFINITION COMPLETE
# ==========================================
# Start the engine only after ALL parts/helpers are defined.

# ==========================================
# PART 6 — FVG + ORDER BLOCK + POI ENGINE
# ==========================================

class POIEngine:

    def __init__(
        self,
        proximity_pct=1.5,
        max_fvg_age=40,
        max_ob_age=60
    ):
        self.proximity_pct = proximity_pct
        self.max_fvg_age = max_fvg_age
        self.max_ob_age = max_ob_age


    # ======================================
    # FAIR VALUE GAP DETECTION
    # ======================================

    @staticmethod
    def detect_fvg(df):

        fvgs = []

        if len(df) < 5:
            return fvgs

        for i in range(2, len(df)):

            c1 = df.iloc[i - 2]
            c2 = df.iloc[i - 1]
            c3 = df.iloc[i]

            # --------------------------------
            # BULLISH FVG
            # Candle 3 LOW > Candle 1 HIGH
            # --------------------------------

            if c3["low"] > c1["high"]:

                fvgs.append({
                    "type": "BULLISH_FVG",
                    "index": i,
                    "low": float(c1["high"]),
                    "high": float(c3["low"]),
                    "timestamp": int(c3["timestamp"])
                })


            # --------------------------------
            # BEARISH FVG
            # Candle 3 HIGH < Candle 1 LOW
            # --------------------------------

            if c3["high"] < c1["low"]:

                fvgs.append({
                    "type": "BEARISH_FVG",
                    "index": i,
                    "low": float(c3["high"]),
                    "high": float(c1["low"]),
                    "timestamp": int(c3["timestamp"])
                })

        return fvgs


    # ======================================
    # ORDER BLOCK DETECTION
    # ======================================

    @staticmethod
    def detect_order_blocks(df):

        order_blocks = []

        if len(df) < 10:
            return order_blocks

        for i in range(2, len(df) - 2):

            candle = df.iloc[i]

            next_1 = df.iloc[i + 1]
            next_2 = df.iloc[i + 2]


            # --------------------------------
            # BULLISH ORDER BLOCK
            #
            # Last bearish candle before
            # strong bullish expansion
            # --------------------------------

            if (
                candle["close"] < candle["open"]
                and
                next_1["close"] > next_1["open"]
                and
                next_2["close"] > next_2["open"]
            ):

                bullish_impulse = (
                    next_2["close"]
                    >
                    candle["high"]
                )

                if bullish_impulse:

                    order_blocks.append({
                        "type": "BULLISH_OB",
                        "index": i,
                        "low": float(candle["low"]),
                        "high": float(candle["high"]),
                        "timestamp": int(
                            candle["timestamp"]
                        )
                    })


            # --------------------------------
            # BEARISH ORDER BLOCK
            #
            # Last bullish candle before
            # strong bearish expansion
            # --------------------------------

            if (
                candle["close"] > candle["open"]
                and
                next_1["close"] < next_1["open"]
                and
                next_2["close"] < next_2["open"]
            ):

                bearish_impulse = (
                    next_2["close"]
                    <
                    candle["low"]
                )

                if bearish_impulse:

                    order_blocks.append({
                        "type": "BEARISH_OB",
                        "index": i,
                        "low": float(candle["low"]),
                        "high": float(candle["high"]),
                        "timestamp": int(
                            candle["timestamp"]
                        )
                    })

        return order_blocks


    # ======================================
    # REMOVE OLD / INVALID POIs
    # ======================================

    def filter_active_pois(
        self,
        pois,
        current_index
    ):

        active = []

        for poi in pois:

            age = (
                current_index
                -
                poi["index"]
            )

            if poi["type"].endswith("_FVG"):

                if age <= self.max_fvg_age:
                    active.append(poi)

            elif poi["type"].endswith("_OB"):

                if age <= self.max_ob_age:
                    active.append(poi)

        return active


    # ======================================
    # DIRECTIONAL POI FILTER
    # ======================================

    @staticmethod
    def directional_pois(
        pois,
        direction
    ):

        if direction == "LONG":

            return [
                p for p in pois
                if p["type"] in [
                    "BULLISH_FVG",
                    "BULLISH_OB"
                ]
            ]

        if direction == "SHORT":

            return [
                p for p in pois
                if p["type"] in [
                    "BEARISH_FVG",
                    "BEARISH_OB"
                ]
            ]

        return []


    # ======================================
    # PRICE PROXIMITY
    # ======================================

    def check_proximity(
        self,
        price,
        pois
    ):

        nearby = []

        if not pois:
            return False, nearby


        for poi in pois:

            poi_low = poi["low"]
            poi_high = poi["high"]


            # Price already inside POI

            inside = (
                poi_low
                <= price
                <= poi_high
            )


            # Distance to POI

            if price > poi_high:

                distance = (
                    (price - poi_high)
                    / price
                    * 100
                )

            elif price < poi_low:

                distance = (
                    (poi_low - price)
                    / price
                    * 100
                )

            else:

                distance = 0.0


            if (
                inside
                or
                distance <= self.proximity_pct
            ):

                poi_copy = poi.copy()

                poi_copy["distance_pct"] = (
                    round(distance, 4)
                )

                poi_copy["inside"] = inside

                nearby.append(
                    poi_copy
                )


        return len(nearby) > 0, nearby


    # ======================================
    # BEST POI SELECTION
    # ======================================

    @staticmethod
    def select_best_poi(
        pois
    ):

        if not pois:
            return None


        # Prefer POI where price is already
        # inside the zone.

        inside_pois = [
            p for p in pois
            if p.get("inside", False)
        ]

        if inside_pois:

            return sorted(
                inside_pois,
                key=lambda x: x.get(
                    "distance_pct",
                    999
                )
            )[0]


        return sorted(
            pois,
            key=lambda x: x.get(
                "distance_pct",
                999
            )
        )[0]


    # ======================================
    # COMPLETE POI ANALYSIS
    # ======================================

    def analyze(
        self,
        df,
        price,
        direction
    ):

        if df is None or df.empty:

            return {
                "valid": False,
                "proximity": False,
                "best_poi": None,
                "pois": []
            }


        current_index = (
            len(df) - 1
        )


        # ----------------------------------
        # Detect FVG
        # ----------------------------------

        fvgs = self.detect_fvg(
            df
        )


        # ----------------------------------
        # Detect Order Blocks
        # ----------------------------------

        obs = self.detect_order_blocks(
            df
        )


        # ----------------------------------
        # Combine
        # ----------------------------------

        all_pois = (
            fvgs + obs
        )


        # ----------------------------------
        # Remove expired POIs
        # ----------------------------------

        active_pois = (
            self.filter_active_pois(
                all_pois,
                current_index
            )
        )


        # ----------------------------------
        # Directional filter
        # ----------------------------------

        directional = (
            self.directional_pois(
                active_pois,
                direction
            )
        )


        # ----------------------------------
        # Proximity
        # ----------------------------------

        proximity, nearby = (
            self.check_proximity(
                price,
                directional
            )
        )


        # ----------------------------------
        # Best POI
        # ----------------------------------

        best = (
            self.select_best_poi(
                nearby
            )
        )


        return {

            "valid": (
                len(directional) > 0
            ),

            "proximity": proximity,

            "best_poi": best,

            "pois": nearby,

            "total_active": len(
                active_pois
            ),

            "directional_count": len(
                directional
            )
        }


# ==========================================
# POI ENGINE HELPER
# ==========================================

def get_directional_poi(
    df_15m,
    price,
    direction="LONG"
):

    engine = POIEngine(
        proximity_pct=1.5,
        max_fvg_age=40,
        max_ob_age=60
    )

    return engine.analyze(
        df_15m,
        price,
        direction
            )

# ==========================================
# FINAL SINGLE ENTRYPOINT
# ==========================================

if __name__ == "__main__":
    try:
        test_market_connection()
    except Exception as exc:
        console.print(
            Panel(
                f"[yellow]STARTUP CHECK WARNING: {exc}[/yellow]",
                title="KUCOIN CONNECTION"
            )
        )

    run_cosmic_radar()
