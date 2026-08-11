import os
import time
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from collections import deque

import numpy as np
import pandas as pd
import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


# ============================================================
# COSMIC 108 V3.1 — CLEAN SQLITE BUILD
# ============================================================
# This is a single-flow rebuild of the uploaded V3 source.
# Important fixes:
#   - ONE DataQualityGuard definition
#   - ONE main entry point
#   - no undefined get_btc_regime()
#   - no undefined SMCMath.get_swings()
#   - no undefined SMCMath.check_displacement()
#   - no undefined ExecutionEngine / calculate_trade_parameters()
#   - SQLite only; no PostgreSQL dependency
#   - all classes/functions are defined before the runner starts
# ============================================================

APP_NAME = "COSMIC 108 V3.1"
VERSION = "3.1"
console = Console()

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "cosmic_108.db"

CONFIG = {
    "database_path": str(DB_PATH),
    "exchange": "KUCOIN",
    "symbols": ["SOL-USDT", "ETH-USDT", "BNB-USDT", "XRP-USDT"],
    "btc_symbol": "BTC-USDT",
    "intervals": {
        "1d": "1day",
        "4h": "4hour",
        "1h": "1hour",
        "15m": "15min",
        "5m": "5min",
    },
    "history": {
        "1d": 100,
        "4h": 150,
        "1h": 150,
        "15m": 200,
        "5m": 250,
    },
    "btc_history": {"1d": 100, "4h": 150, "1h": 150},
    "refresh_seconds": 30,
    "request_timeout": 10,
    "max_retries": 3,
    "retry_delay": 1.0,
    "swing_left": 3,
    "swing_right": 3,
    "poi_proximity_percent": 1.5,
    "risk_usd": 10.0,
    "maker_fee": 0.0006,
    "taker_fee": 0.0008,
    "slippage": 0.0005,
    "cooldown_seconds": 1800,
    "audit_keep": 5000,
}


# ============================================================
# SQLITE
# ============================================================

class SQLiteStore:
    def __init__(self, db_path=None):
        self.db_path = str(db_path or CONFIG["database_path"])
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        conn = sqlite3.connect(
            self.db_path,
            timeout=15,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _initialize(self):
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
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    signal_id TEXT NOT NULL UNIQUE,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    status TEXT NOT NULL,
                    grade TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    entry REAL,
                    sl REAL,
                    tp REAL,
                    real_rr REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs(created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_signal_symbol_created ON signals(symbol, created_at DESC)")
            conn.commit()

    def add_audit(self, symbol, event, score, message):
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO audit_logs(created_at,symbol,event,score,message) VALUES(?,?,?,?,?)",
                (created_at, symbol, event, int(score), str(message)),
            )
            conn.commit()

    def add_signal(self, signal):
        created_at = signal.get("time") or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        trade = signal.get("trade") or {}
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO signals
                (created_at,signal_id,symbol,direction,status,grade,score,entry,sl,tp,real_rr)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    created_at,
                    signal["signal_id"],
                    signal["symbol"],
                    signal["direction"],
                    signal["status"],
                    signal["grade"],
                    int(signal["score"]),
                    trade.get("entry"),
                    trade.get("sl"),
                    trade.get("tp"),
                    trade.get("real_rr"),
                ),
            )
            conn.commit()

    def latest_audits(self, limit=20):
        limit = max(1, min(int(limit), 500))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT created_at,symbol,event,score,message FROM audit_logs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def latest_signal(self, symbol):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM signals WHERE symbol=? ORDER BY id DESC LIMIT 1",
                (symbol,),
            ).fetchone()
        return dict(row) if row else None

    def cleanup(self):
        keep = max(100, int(CONFIG["audit_keep"]))
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM audit_logs WHERE id NOT IN (SELECT id FROM audit_logs ORDER BY id DESC LIMIT ?)",
                (keep,),
            )
            conn.commit()


DB = SQLiteStore()


class AuditLogger:
    def __init__(self, max_records=50):
        self.records = deque(maxlen=max_records)
        for row in reversed(DB.latest_audits(max_records)):
            self.records.appendleft({
                "time": row["created_at"],
                "event": row["event"],
                "score": row["score"],
                "details": f'{row["symbol"]}: {row["message"]}',
            })

    def log(self, symbol, event, score, details):
        item = {
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S UTC"),
            "event": event,
            "score": int(score),
            "details": str(details),
        }
        self.records.appendleft(item)
        try:
            DB.add_audit(symbol, event, score, details)
            DB.cleanup()
        except Exception as exc:
            console.print(f"[yellow]SQLite audit warning: {exc}[/yellow]")

    def latest(self):
        return list(self.records)


# ============================================================
# KUCOIN PUBLIC DATA
# ============================================================

class KuCoinLive:
    BASE_URL = "https://api.kucoin.com"

    @classmethod
    def get_klines(cls, symbol, interval, limit=100):
        endpoint = f"{cls.BASE_URL}/api/v1/market/candles"
        params = {"symbol": symbol, "type": interval}
        last_error = None

        for attempt in range(1, CONFIG["max_retries"] + 1):
            try:
                response = requests.get(
                    endpoint,
                    params=params,
                    timeout=CONFIG["request_timeout"],
                    headers={"User-Agent": "COSMIC-108/3.1"},
                )
                response.raise_for_status()
                payload = response.json()

                if payload.get("code") not in (None, "200000"):
                    raise RuntimeError(f'KuCoin API code: {payload.get("code")}')

                data = payload.get("data") or []
                if not data:
                    return pd.DataFrame()

                columns = ["timestamp", "open", "close", "high", "low", "volume", "turnover"]
                df = pd.DataFrame(data, columns=columns)
                df = df[["timestamp", "open", "high", "low", "close", "volume"]]

                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                df = df.dropna().copy()
                if df.empty:
                    return pd.DataFrame()

                df["timestamp"] = df["timestamp"].astype("int64")
                df = df.drop_duplicates("timestamp", keep="last")
                df = df.sort_values("timestamp").reset_index(drop=True)
                return df.tail(int(limit)).reset_index(drop=True)

            except Exception as exc:
                last_error = exc
                if attempt < CONFIG["max_retries"]:
                    time.sleep(CONFIG["retry_delay"] * attempt)

        console.print(f"[red]KuCoin API error {symbol} {interval}: {last_error}[/red]")
        return pd.DataFrame()


# ============================================================
# DATA QUALITY
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

    TIMEFRAME_MINUTES = {
        "1min": 1,
        "5min": 5,
        "15min": 15,
        "1hour": 60,
        "4hour": 240,
        "1day": 1440,
    }

    @classmethod
    def validate(cls, df, interval=None, minimum_candles=50, timeframe_minutes=None, time_frame_min=None, min_candles=None):
        if min_candles is not None:
            minimum_candles = min_candles
        if timeframe_minutes is None and time_frame_min is not None:
            timeframe_minutes = time_frame_min
        if timeframe_minutes is None and interval in cls.TIMEFRAME_MINUTES:
            timeframe_minutes = cls.TIMEFRAME_MINUTES[interval]

        if df is None or df.empty:
            return False, "NO DATA"
        if len(df) < int(minimum_candles):
            return False, f"INSUFFICIENT DATA ({len(df)}/{minimum_candles})"

        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return False, f"MISSING COLUMN: {missing[0]}"

        if df[required].isna().any().any():
            return False, "NULL DATA"
        if df["timestamp"].duplicated().any():
            return False, "DUPLICATE TIMESTAMP"
        if not df["timestamp"].is_monotonic_increasing:
            return False, "TIMESTAMP ORDER ERROR"
        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            return False, "INVALID PRICE"
        if (df["volume"] < 0).any():
            return False, "INVALID VOLUME"
        if (df["high"] < df["low"]).any():
            return False, "INVALID HIGH/LOW"
        if (df["high"] < df[["open", "close"]].max(axis=1)).any():
            return False, "INVALID HIGH"
        if (df["low"] > df[["open", "close"]].min(axis=1)).any():
            return False, "INVALID LOW"

        if timeframe_minutes:
            expected = int(timeframe_minutes) * 60
            diffs = df["timestamp"].diff().dropna()
            if not diffs.empty and (diffs > expected * 1.5).any():
                return False, "CANDLE GAP DETECTED"

        return True, "DATA OK"


class StaleDataGuard:
    @classmethod
    def is_stale(cls, df, interval, tolerance_multiplier=2.5):
        if df is None or df.empty:
            return True
        seconds = DataQualityGuard.TIMEFRAME_SECONDS.get(interval)
        if not seconds:
            return False
        latest = int(df["timestamp"].iloc[-1])
        now = int(datetime.now(timezone.utc).timestamp())
        return (now - latest) > seconds * tolerance_multiplier


# ============================================================
# SMC MATH
# ============================================================

class SMCMath:
    @staticmethod
    def calculate_atr(df, period=14):
        prev = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=period).mean()

    @staticmethod
    def ema(df, period):
        return df["close"].ewm(span=period, adjust=False, min_periods=period).mean()

    @staticmethod
    def volume_expansion(df, idx, period=20, multiplier=1.5):
        if idx < period:
            return False
        avg = float(df["volume"].iloc[idx-period:idx].mean())
        return avg > 0 and float(df["volume"].iloc[idx]) >= avg * multiplier

    @staticmethod
    def displacement(df, idx, atr_multiplier=1.2, volume_multiplier=1.3, body_ratio_min=0.65):
        if idx < 20:
            return False
        atr = SMCMath.calculate_atr(df).iloc[idx]
        if pd.isna(atr) or atr <= 0:
            return False
        o = float(df["open"].iloc[idx]); c = float(df["close"].iloc[idx])
        h = float(df["high"].iloc[idx]); l = float(df["low"].iloc[idx])
        rng = h - l
        if rng <= 0:
            return False
        body = abs(c - o)
        return body >= atr * atr_multiplier and body / rng >= body_ratio_min and SMCMath.volume_expansion(df, idx, 20, volume_multiplier)

    @staticmethod
    def check_displacement(df, idx):
        return SMCMath.displacement(df, idx)

    @staticmethod
    def detect_swings(df, left=3, right=3):
        data = df.copy()
        data["swing_high"] = False
        data["swing_low"] = False
        if len(data) < left + right + 1:
            return data
        highs = data["high"].to_numpy()
        lows = data["low"].to_numpy()
        for i in range(left, len(data) - right):
            if highs[i] >= np.max(highs[i-left:i]) and highs[i] >= np.max(highs[i+1:i+right+1]):
                data.loc[data.index[i], "swing_high"] = True
            if lows[i] <= np.min(lows[i-left:i]) and lows[i] <= np.min(lows[i+1:i+right+1]):
                data.loc[data.index[i], "swing_low"] = True
        return data

    @staticmethod
    def get_swings(df, left=3, right=3):
        return SMCMath.detect_swings(df, left, right)

    @staticmethod
    def detect_liquidity_sweep(df, direction="LONG"):
        data = SMCMath.detect_swings(df)
        if len(data) < 10:
            return False
        current = data.iloc[-1]
        if direction == "LONG":
            swings = data[data["swing_low"]]
            if swings.empty:
                return False
            level = float(swings["low"].iloc[-1])
            return float(current["low"]) < level and float(current["close"]) > level
        swings = data[data["swing_high"]]
        if swings.empty:
            return False
        level = float(swings["high"].iloc[-1])
        return float(current["high"]) > level and float(current["close"]) < level

    @staticmethod
    def detect_mss(df, direction="LONG"):
        data = SMCMath.detect_swings(df)
        if len(data) < 8:
            return False
        price = float(data["close"].iloc[-1])
        if direction == "LONG":
            highs = data[data["swing_high"]]["high"]
            return not highs.empty and price > float(highs.iloc[-1])
        highs = data[data["swing_low"]]["low"]
        return not highs.empty and price < float(highs.iloc[-1])

    @staticmethod
    def detect_bos(df, direction="LONG"):
        return SMCMath.detect_mss(df, direction)

    @staticmethod
    def detect_fvg(df):
        data = df.copy()
        data["bullish_fvg"] = False
        data["bearish_fvg"] = False
        if len(data) < 3:
            return data
        for i in range(2, len(data)):
            if float(data["low"].iloc[i]) > float(data["high"].iloc[i-2]):
                data.loc[data.index[i], "bullish_fvg"] = True
            if float(data["high"].iloc[i]) < float(data["low"].iloc[i-2]):
                data.loc[data.index[i], "bearish_fvg"] = True
        return data


# ============================================================
# MARKET REGIME / HTF
# ============================================================

class MarketRegimeEngine:
    @staticmethod
    def get_trend(df, fast=20, slow=50):
        if df is None or len(df) < slow:
            return "UNKNOWN"
        ema_fast = SMCMath.ema(df, fast).iloc[-1]
        ema_slow = SMCMath.ema(df, slow).iloc[-1]
        price = float(df["close"].iloc[-1])
        if pd.isna(ema_fast) or pd.isna(ema_slow):
            return "UNKNOWN"
        if price > ema_fast > ema_slow:
            return "BULLISH"
        if price < ema_fast < ema_slow:
            return "BEARISH"
        return "CHOP"

    @classmethod
    def btc_regime(cls, df_1d, df_4h, df_1h):
        trends = {
            "1d": cls.get_trend(df_1d),
            "4h": cls.get_trend(df_4h),
            "1h": cls.get_trend(df_1h),
        }
        bullish = list(trends.values()).count("BULLISH")
        bearish = list(trends.values()).count("BEARISH")
        if bullish == 3:
            regime = "STRONG_BULLISH"
        elif bearish == 3:
            regime = "STRONG_BEARISH"
        elif bullish >= 2:
            regime = "BULLISH"
        elif bearish >= 2:
            regime = "BEARISH"
        else:
            regime = "CHOP"
        return {"regime": regime, **trends}

    @staticmethod
    def allow_direction(regime, direction):
        if direction == "LONG":
            return regime in ("BULLISH", "STRONG_BULLISH")
        if direction == "SHORT":
            return regime in ("BEARISH", "STRONG_BEARISH")
        return False


# Compatibility wrapper: always defined, but internally uses the real engine.
def get_btc_regime():
    frames = {}
    for key in ("1d", "4h", "1h"):
        interval = CONFIG["intervals"][key]
        frames[key] = KuCoinLive.get_klines(CONFIG["btc_symbol"], interval, CONFIG["btc_history"][key])
    if any(df.empty for df in frames.values()):
        return "UNKNOWN"
    result = MarketRegimeEngine.btc_regime(frames["1d"], frames["4h"], frames["1h"])
    return result["regime"]


class HTFAlignmentEngine:
    @staticmethod
    def analyze(df_1d, df_4h, df_1h, direction):
        trends = [
            MarketRegimeEngine.get_trend(df_1d),
            MarketRegimeEngine.get_trend(df_4h),
            MarketRegimeEngine.get_trend(df_1h),
        ]
        wanted = "BULLISH" if direction == "LONG" else "BEARISH"
        return trends.count(wanted) >= 2


# ============================================================
# POI ENGINE
# ============================================================

class POIEngine:
    @staticmethod
    def detect_order_blocks(df):
        data = df.copy()
        data["bullish_ob"] = False
        data["bearish_ob"] = False
        if len(data) < 3:
            return data
        for i in range(1, len(data)):
            prev = data.iloc[i-1]
            cur = data.iloc[i]
            if float(prev["close"]) < float(prev["open"]) and float(cur["close"]) > float(cur["high"] if "high" in cur else cur["close"]):
                data.loc[data.index[i-1], "bullish_ob"] = True
            if float(prev["close"]) > float(prev["open"]) and float(cur["close"]) < float(cur["low"] if "low" in cur else cur["close"]):
                data.loc[data.index[i-1], "bearish_ob"] = True
        return data

    @staticmethod
    def nearest_poi(df, price, direction, proximity_pct=1.5):
        data = SMCMath.detect_fvg(df)
        levels = []
        if direction == "LONG":
            lows = data["low"].tail(30).tolist()
            levels.extend(lows)
            fvg = data.loc[data["bullish_fvg"], "low"].tolist()
            levels.extend(fvg)
            candidates = [float(x) for x in levels if float(x) <= price]
            if not candidates:
                return False, None, None
            level = max(candidates)
        else:
            highs = data["high"].tail(30).tolist()
            levels.extend(highs)
            fvg = data.loc[data["bearish_fvg"], "high"].tolist()
            levels.extend(fvg)
            candidates = [float(x) for x in levels if float(x) >= price]
            if not candidates:
                return False, None, None
            level = min(candidates)
        distance = abs(price - level) / price * 100
        return distance <= proximity_pct, level, distance


# ============================================================
# RISK / EXECUTION
# ============================================================

class RiskEngine:
    def __init__(self, risk_usd=None, taker_fee=None, maker_fee=None, slippage=None):
        self.risk_usd = float(CONFIG["risk_usd"] if risk_usd is None else risk_usd)
        self.taker_fee = float(CONFIG["taker_fee"] if taker_fee is None else taker_fee)
        self.maker_fee = float(CONFIG["maker_fee"] if maker_fee is None else maker_fee)
        self.slippage = float(CONFIG["slippage"] if slippage is None else slippage)

    def calculate_levels(self, df, direction, poi_level=None):
        atr = SMCMath.calculate_atr(df).iloc[-1]
        if pd.isna(atr) or atr <= 0:
            return None
        price = float(df["close"].iloc[-1])
        recent_high = float(df["high"].tail(15).max())
        recent_low = float(df["low"].tail(15).min())
        if direction == "LONG":
            base = poi_level if poi_level is not None else recent_low
            sl = min(base - atr * 0.20, price - atr)
            risk_per_unit = price - sl
            tp = price + risk_per_unit * 2.5
        elif direction == "SHORT":
            base = poi_level if poi_level is not None else recent_high
            sl = max(base + atr * 0.20, price + atr)
            risk_per_unit = sl - price
            tp = price - risk_per_unit * 2.5
        else:
            return None
        if risk_per_unit <= 0 or tp <= 0:
            return None
        return {"entry": price, "sl": sl, "tp": tp, "atr": float(atr), "risk_per_unit": risk_per_unit}

    def calculate_execution(self, levels, direction):
        if not levels:
            return {"valid": False, "reason": "NO RISK LEVELS"}
        entry, sl, tp = levels["entry"], levels["sl"], levels["tp"]
        if direction == "LONG":
            effective_entry = entry * (1 + self.slippage)
            effective_sl = sl * (1 - self.slippage)
            risk_per_unit = effective_entry - effective_sl
            reward_per_unit = tp - effective_entry
        elif direction == "SHORT":
            effective_entry = entry * (1 - self.slippage)
            effective_sl = sl * (1 + self.slippage)
            risk_per_unit = effective_sl - effective_entry
            reward_per_unit = effective_entry - tp
        else:
            return {"valid": False, "reason": "INVALID DIRECTION"}
        if risk_per_unit <= 0 or reward_per_unit <= 0:
            return {"valid": False, "reason": "INVALID RISK/REWARD"}
        size = self.risk_usd / risk_per_unit
        entry_fee = effective_entry * size * self.taker_fee
        exit_fee = tp * size * self.maker_fee
        fees = entry_fee + exit_fee
        gross_reward = reward_per_unit * size
        net_reward = gross_reward - fees
        net_risk = self.risk_usd + fees
        rr = net_reward / net_risk if net_risk > 0 else 0.0
        return {
            "valid": rr >= 2.0,
            "entry": round(effective_entry, 8),
            "sl": round(effective_sl, 8),
            "tp": round(tp, 8),
            "position_size": round(size, 8),
            "fees": round(fees, 8),
            "risk_usd": round(net_risk, 2),
            "net_reward": round(net_reward, 2),
            "real_rr": round(rr, 2),
        }

    def calculate_trade_parameters(self, entry, sl, tp, direction):
        levels = {"entry": float(entry), "sl": float(sl), "tp": float(tp)}
        return self.calculate_execution(levels, direction)


class ExecutionEngine(RiskEngine):
    """Compatibility alias for the older uploaded V3 naming."""
    pass


# ============================================================
# SIGNAL ENGINE
# ============================================================

class SignalEngine:
    WEIGHTS = {
        "BTC REGIME": 10,
        "HTF ALIGNMENT": 15,
        "15M POI": 10,
        "POI PROXIMITY": 5,
        "LIQUIDITY SWEEP": 15,
        "MSS / CHOCH": 15,
        "DISPLACEMENT": 10,
        "BOS": 10,
        "POI RETEST": 5,
        "REALISTIC RR >= 2": 5,
    }

    @classmethod
    def evaluate(cls, btc_regime, htf_alignment, poi_valid, poi_proximity, liquidity_sweep, mss_confirmed, displacement, bos_confirmed, retest_confirmed, real_rr):
        checks = {
            "BTC REGIME": bool(btc_regime),
            "HTF ALIGNMENT": bool(htf_alignment),
            "15M POI": bool(poi_valid),
            "POI PROXIMITY": bool(poi_proximity),
            "LIQUIDITY SWEEP": bool(liquidity_sweep),
            "MSS / CHOCH": bool(mss_confirmed),
            "DISPLACEMENT": bool(displacement),
            "BOS": bool(bos_confirmed),
            "POI RETEST": bool(retest_confirmed),
            "REALISTIC RR >= 2": float(real_rr) >= 2.0,
        }
        score = sum(cls.WEIGHTS[k] for k, passed in checks.items() if passed)
        mandatory = all(checks.values())
        if mandatory and score >= 90:
            grade, status = "A+ SETUP", "VALID SIGNAL"
        elif mandatory and score >= 85:
            grade, status = "A SETUP", "VALID SIGNAL"
        elif checks["LIQUIDITY SWEEP"] and (checks["MSS / CHOCH"] or checks["DISPLACEMENT"] or checks["BOS"]):
            grade, status = "WATCH", "SETUP FORMING"
        else:
            grade, status = "NO TRADE", "NO TRADE"
        return {"score": score, "grade": grade, "status": status, "checks": checks}


# ============================================================
# RADAR
# ============================================================

class CosmicLiveRadar:
    def __init__(self, symbol="SOL-USDT"):
        self.symbol = symbol
        self.last_signal_id = None
        self.last_signal_time = 0.0
        self.cooldown_seconds = CONFIG["cooldown_seconds"]
        self.audit = AuditLogger()
        self.risk = RiskEngine()
        self._restore_last_signal()

    def _restore_last_signal(self):
        try:
            row = DB.latest_signal(self.symbol)
            if row:
                self.last_signal_id = row["signal_id"]
                created = row["created_at"].replace(" UTC", "")
                dt = datetime.strptime(created, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                self.last_signal_time = dt.timestamp()
        except Exception:
            self.last_signal_id = None
            self.last_signal_time = 0.0

    def cooldown_active(self):
        return self.last_signal_time > 0 and (time.time() - self.last_signal_time) < self.cooldown_seconds

    def log(self, event, score, details):
        self.audit.log(self.symbol, event, score, details)

    def _fetch_bundle(self):
        data = {}
        for key, interval in CONFIG["intervals"].items():
            df = KuCoinLive.get_klines(self.symbol, interval, CONFIG["history"][key])
            ok, reason = DataQualityGuard.validate(df, interval=interval, minimum_candles=50)
            if not ok:
                raise RuntimeError(f"{key}: {reason}")
            if StaleDataGuard.is_stale(df, interval):
                raise RuntimeError(f"{key}: STALE DATA")
            data[key] = df
        return data

    def _fetch_btc_regime(self):
        frames = {}
        for key in ("1d", "4h", "1h"):
            interval = CONFIG["intervals"][key]
            df = KuCoinLive.get_klines(CONFIG["btc_symbol"], interval, CONFIG["btc_history"][key])
            ok, reason = DataQualityGuard.validate(df, interval=interval, minimum_candles=50)
            if not ok:
                raise RuntimeError(f"BTC {key}: {reason}")
            if StaleDataGuard.is_stale(df, interval):
                raise RuntimeError(f"BTC {key}: STALE DATA")
            frames[key] = df
        return MarketRegimeEngine.btc_regime(frames["1d"], frames["4h"], frames["1h"])

    def analyze(self):
        data = self._fetch_bundle()
        btc = self._fetch_btc_regime()

        df_1d, df_4h, df_1h = data["1d"], data["4h"], data["1h"]
        df_15m = SMCMath.detect_swings(data["15m"])
        df_5m = SMCMath.detect_swings(data["5m"])
        price = float(df_5m["close"].iloc[-1])

        # Direction is determined from the 1H/15M structure, not hard-coded LONG.
        htf_long = HTFAlignmentEngine.analyze(df_1d, df_4h, df_1h, "LONG")
        htf_short = HTFAlignmentEngine.analyze(df_1d, df_4h, df_1h, "SHORT")
        direction = "LONG" if htf_long and not htf_short else "SHORT" if htf_short and not htf_long else None

        if direction is None:
            self.log("NO TRADE", 0, "HTF direction is not aligned")
            return self._no_trade(price, btc["regime"], "HTF DIRECTION CHOP")

        btc_ok = MarketRegimeEngine.allow_direction(btc["regime"], direction)
        poi_valid, poi_level, poi_distance = POIEngine.nearest_poi(df_15m, price, direction, CONFIG["poi_proximity_percent"])
        poi_proximity = poi_valid and poi_level is not None

        sweep = SMCMath.detect_liquidity_sweep(df_5m, direction)
        displacement = SMCMath.check_displacement(df_5m, len(df_5m) - 1)
        mss = SMCMath.detect_mss(df_5m, direction)
        bos = SMCMath.detect_bos(df_5m, direction)

        # Retest: price is near the latest structural level within 0.5 ATR.
        atr = SMCMath.calculate_atr(df_5m).iloc[-1]
        retest = False
        if pd.notna(atr) and atr > 0:
            if direction == "LONG":
                recent_structure = float(df_5m["high"].tail(8).max())
            else:
                recent_structure = float(df_5m["low"].tail(8).min())
            retest = abs(price - recent_structure) <= float(atr) * 0.5

        levels = self.risk.calculate_levels(df_5m, direction, poi_level if poi_valid else None)
        trade = self.risk.calculate_execution(levels, direction)
        real_rr = float(trade.get("real_rr", 0.0))

        result = SignalEngine.evaluate(
            btc_regime=btc_ok,
            htf_alignment=True,
            poi_valid=poi_valid,
            poi_proximity=poi_proximity,
            liquidity_sweep=sweep,
            mss_confirmed=mss,
            displacement=displacement,
            bos_confirmed=bos,
            retest_confirmed=retest,
            real_rr=real_rr,
        )

        candle_ts = int(df_5m["timestamp"].iloc[-1])
        signal_id = f"{self.symbol}_{candle_ts}_{direction}_{result['grade']}"
        status = result["status"]

        if status == "VALID SIGNAL":
            if self.cooldown_active():
                status = "COOLDOWN"
            elif signal_id == self.last_signal_id:
                status = "DUPLICATE BLOCKED"
            else:
                self.last_signal_id = signal_id
                self.last_signal_time = time.time()
                self.log("VALID SIGNAL", result["score"], f"{direction} | {result['grade']} | RR 1:{real_rr:.2f}")
        elif status == "SETUP FORMING":
            self.log("SETUP FORMING", result["score"], f"{direction} | POI distance {poi_distance}")
        else:
            self.log("NO TRADE", result["score"], f"{direction} | BTC {btc['regime']}")

        output = {
            "signal_id": signal_id,
            "symbol": self.symbol,
            "price": round(price, 8),
            "direction": direction,
            "btc_regime": btc["regime"],
            "btc_detail": btc,
            "status": status,
            "grade": result["grade"],
            "score": result["score"],
            "checks": result["checks"],
            "poi_level": round(float(poi_level), 8) if poi_level is not None else None,
            "poi_distance_pct": round(float(poi_distance), 3) if poi_distance is not None else None,
            "trade": trade if trade.get("valid") else None,
            "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

        if status in ("VALID SIGNAL", "COOLDOWN", "DUPLICATE BLOCKED"):
            try:
                DB.add_signal(output)
            except Exception as exc:
                console.print(f"[yellow]SQLite signal warning: {exc}[/yellow]")

        return output

    def _no_trade(self, price, btc_regime, reason):
        return {
            "signal_id": f"{self.symbol}_{int(time.time())}_NO_TRADE",
            "symbol": self.symbol,
            "price": round(price, 8),
            "direction": "NONE",
            "btc_regime": btc_regime,
            "status": "NO TRADE",
            "grade": "NO TRADE",
            "score": 0,
            "checks": {},
            "trade": None,
            "reason": reason,
            "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }


# ============================================================
# TERMINAL UI
# ============================================================

def render_radar(result, audit_log):
    console.clear()
    title = f"{APP_NAME} — LIVE SMC RADAR"
    status = result.get("status", "UNKNOWN")
    color = "green" if status == "VALID SIGNAL" else "yellow" if status in ("SETUP FORMING", "COOLDOWN") else "red"

    trade = result.get("trade")
    trade_text = ""
    if trade:
        trade_text = (
            f"\nENTRY: {trade['entry']}"
            f"\nSL: {trade['sl']}"
            f"\nTP: {trade['tp']}"
            f"\nREAL RR: 1:{trade['real_rr']}"
            f"\nPOSITION: {trade['position_size']}"
        )

    text = (
        f"PAIR: {result.get('symbol')}\n"
        f"PRICE: {result.get('price')}\n"
        f"DIRECTION: {result.get('direction', 'NONE')}\n"
        f"BTC REGIME: {result.get('btc_regime')}\n"
        f"SCORE: {result.get('score', 0)}/100\n"
        f"GRADE: {result.get('grade')}\n"
        f"STATUS: [{color}]{status}[/{color}]"
        f"{trade_text}"
    )
    console.print(Panel(text, title=title, expand=False))

    checks = result.get("checks", {})
    if checks:
        table = Table(title="VALIDATION")
        table.add_column("CHECK")
        table.add_column("RESULT", justify="center")
        for key, passed in checks.items():
            table.add_row(key, "[green]PASS[/green]" if passed else "[red]FAIL[/red]")
        console.print(table)

    logs = Table(title="AUDIT LOG")
    logs.add_column("TIME")
    logs.add_column("EVENT")
    logs.add_column("SCORE")
    logs.add_column("DETAILS")
    for item in audit_log[:10]:
        logs.add_row(str(item["time"]), str(item["event"]), str(item["score"]), str(item["details"]))
    console.print(logs)


def test_market_connection():
    console.print(Panel(f"[bold cyan]{APP_NAME}[/bold cyan]\nTesting KuCoin public market data..."))
    df = KuCoinLive.get_klines("BTC-USDT", "5min", 10)
    if df.empty:
        console.print("[bold red]FAILED: KuCoin market data unavailable[/bold red]")
        return False
    console.print(f"[bold green]OK: received {len(df)} BTC-USDT candles[/bold green]")
    return True


def run_cosmic_radar(symbol=None):
    symbol = symbol or os.getenv("COSMIC_SYMBOL", CONFIG["symbols"][0])
    if symbol not in CONFIG["symbols"]:
        console.print(f"[yellow]Unknown symbol {symbol}; using {CONFIG['symbols'][0]}[/yellow]")
        symbol = CONFIG["symbols"][0]

    if not test_market_connection():
        return

    radar = CosmicLiveRadar(symbol)
    console.print(Panel(
        f"[bold cyan]{APP_NAME}[/bold cyan]\n"
        f"Starting LIVE SMC Radar for {symbol}\n"
        f"SQLite: {DB_PATH}\n"
        f"Refresh: {CONFIG['refresh_seconds']}s",
        title="SYSTEM START",
    ))

    while True:
        try:
            result = radar.analyze()
            render_radar(result, radar.audit.latest())
            time.sleep(CONFIG["refresh_seconds"])
        except KeyboardInterrupt:
            console.print("\n[bold red]COSMIC RADAR STOPPED[/bold red]")
            break
        except Exception as exc:
            # One scan failing must NOT kill the engine.
            radar.log("ENGINE ERROR", 0, repr(exc))
            console.print(Panel(f"[red]ENGINE ERROR: {exc}[/red]", title="RECOVERABLE ERROR"))
            time.sleep(10)


if __name__ == "__main__":
    run_cosmic_radar()
