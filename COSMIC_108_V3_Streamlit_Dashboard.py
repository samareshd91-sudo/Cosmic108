import os
import time
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from collections import deque

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ============================================================
# COSMIC 108 V3.1 — STREAMLIT DASHBOARD
# Derived from COSMIC_108_V3_SQLite_v3_fixed.py
# - SQLite only
# - KuCoin public market data
# - SMC / HTF / BTC regime / POI / risk / scoring
# - Browser dashboard (no Rich terminal renderer)
# - Compatibility aliases for get_swings / check_displacement
# ============================================================

APP_NAME = "COSMIC 108 V3.1"
VERSION = "3.1-STREAMLIT"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONFIG = {
    "database_path": str(Path(__file__).with_name("cosmic_108.db")),
    "exchange": "KUCOIN",
    "symbols": ["SOL-USDT", "ETH-USDT", "BNB-USDT", "XRP-USDT"],
    "htf_1d": "1day",
    "htf_4h": "4hour",
    "htf_1h": "1hour",
    "poi_tf": "15min",
    "entry_tf": "5min",
    "history_1d": 100,
    "history_4h": 150,
    "history_1h": 150,
    "history_15m": 200,
    "history_5m": 250,
    "refresh_seconds": 30,
    "request_timeout": 8,
    "max_retries": 3,
    "swing_window": 3,
    "poi_proximity_percent": 1.5,
    "account_risk_usd": 10.0,
    "maker_fee": 0.0006,
    "taker_fee": 0.0008,
    "slippage": 0.0005,
    "cooldown_seconds": 1800,
    "max_audit_records": 50,
}

# -----------------------------
# UI CSS
# -----------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
    .cosmic-title {font-size: 2.0rem; font-weight: 800; letter-spacing: .04em;}
    .cosmic-sub {opacity: .72; margin-bottom: 1rem;}
    .status-card {padding: .9rem 1rem; border: 1px solid rgba(128,128,128,.25); border-radius: 14px;}
    .small-muted {font-size: .82rem; opacity: .65;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# KUCOIN DATA
# ============================================================
class KuCoinLive:
    BASE_URL = "https://api.kucoin.com"

    @staticmethod
    def get_klines(symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        endpoint = f"{KuCoinLive.BASE_URL}/api/v1/market/candles"
        params = {"symbol": symbol, "type": interval}
        last_error = None

        for attempt in range(1, CONFIG["max_retries"] + 1):
            try:
                response = requests.get(
                    endpoint,
                    params=params,
                    timeout=CONFIG["request_timeout"],
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get("code") not in (None, "200000"):
                    raise RuntimeError(f"KuCoin code={payload.get('code')}")

                data = payload.get("data", [])
                if not data:
                    return pd.DataFrame()

                columns = [
                    "timestamp", "open", "close", "high", "low", "volume", "turnover"
                ]
                df = pd.DataFrame(data, columns=columns)
                df = df[["timestamp", "open", "high", "low", "close", "volume"]]

                for column in df.columns:
                    df[column] = pd.to_numeric(df[column], errors="coerce")

                df = df.dropna()
                if df.empty:
                    return pd.DataFrame()

                df["timestamp"] = df["timestamp"].astype(np.int64)
                df = (
                    df.sort_values("timestamp")
                    .drop_duplicates("timestamp", keep="last")
                    .reset_index(drop=True)
                    .tail(int(limit))
                    .reset_index(drop=True)
                )
                return df
            except Exception as exc:
                last_error = exc
                if attempt < CONFIG["max_retries"]:
                    time.sleep(0.8)

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

    @staticmethod
    def validate(df, interval=None, timeframe_minutes=None, time_frame_min=None, min_candles=50):
        if timeframe_minutes is None and time_frame_min is not None:
            timeframe_minutes = time_frame_min
        if timeframe_minutes is None and interval:
            timeframe_minutes = DataQualityGuard.TIMEFRAME_SECONDS.get(interval, 300) // 60
        try:
            timeframe_minutes = int(timeframe_minutes or 5)
        except (TypeError, ValueError):
            return False, "INVALID TIMEFRAME"

        if df is None or df.empty:
            return False, "NO DATA"
        if len(df) < min_candles:
            return False, f"INSUFFICIENT DATA ({len(df)}/{min_candles})"

        required = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                return False, f"MISSING COLUMN: {col}"

        if df["timestamp"].duplicated().any():
            return False, "DUPLICATE TIMESTAMP"
        if not df["timestamp"].is_monotonic_increasing:
            return False, "TIMESTAMP ORDER ERROR"
        if df[required].isna().any().any():
            return False, "NULL DATA"
        if (
            (df["high"] < df["low"]).any()
            or (df["high"] < df["open"]).any()
            or (df["high"] < df["close"]).any()
            or (df["low"] > df["open"]).any()
            or (df["low"] > df["close"]).any()
        ):
            return False, "INVALID OHLC"
        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            return False, "INVALID PRICE"
        if (df["volume"] < 0).any():
            return False, "INVALID VOLUME"

        expected = timeframe_minutes * 60
        diffs = df["timestamp"].diff().dropna()
        if not diffs.empty and (diffs > expected * 1.5).any():
            return False, "CANDLE GAP DETECTED"

        return True, "DATA OK"


class StaleDataGuard:
    @staticmethod
    def is_stale(df, interval, tolerance_multiplier=2.0):
        if df is None or df.empty:
            return True
        timeframe_seconds = DataQualityGuard.TIMEFRAME_SECONDS.get(interval)
        if not timeframe_seconds:
            return False
        latest = int(df["timestamp"].iloc[-1])
        age = int(datetime.now(timezone.utc).timestamp()) - latest
        return age > timeframe_seconds * tolerance_multiplier


# ============================================================
# SQLITE STORE
# ============================================================
class SQLiteStore:
    def __init__(self, db_path=None):
        self.db_path = db_path or CONFIG["database_path"]
        self._initialize()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _initialize(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    event TEXT NOT NULL,
                    score INTEGER NOT NULL DEFAULT 0,
                    message TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at DESC)"
            )
            conn.commit()

    def add_audit(self, symbol, event, score, message, created_at):
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO audit_logs(created_at,symbol,event,score,message) VALUES(?,?,?,?,?)",
                (created_at, symbol, event, int(score), str(message)),
            )
            conn.commit()

    def latest_audits(self, limit=50):
        limit = max(1, min(int(limit), 500))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT created_at,symbol,event,score,message FROM audit_logs ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def cleanup_audits(self, keep=5000):
        keep = max(100, int(keep))
        with self._connect() as conn:
            conn.execute(
                """
                DELETE FROM audit_logs
                WHERE id NOT IN (
                    SELECT id FROM audit_logs ORDER BY id DESC LIMIT ?
                )
                """,
                (keep,),
            )
            conn.commit()

    def count(self):
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM audit_logs").fetchone()[0])


DB = SQLiteStore()


class AuditLogger:
    def __init__(self, max_records=50):
        self.records = deque(maxlen=max_records)
        try:
            for item in reversed(DB.latest_audits(max_records)):
                self.records.appendleft(item)
        except Exception:
            pass

    def log(self, symbol, event, message, score=0):
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        item = {
            "created_at": created_at,
            "symbol": symbol,
            "event": event,
            "score": int(score),
            "message": str(message),
        }
        self.records.appendleft(item)
        try:
            DB.add_audit(symbol, event, score, message, created_at)
            DB.cleanup_audits(5000)
        except Exception:
            pass

    def latest(self, limit=50):
        return list(self.records)[:limit]


# ============================================================
# SMC MATH — compatibility-safe
# ============================================================
class SMCMath:
    @staticmethod
    def calculate_atr(df, period=14):
        prev = df["close"].shift(1)
        tr = pd.concat(
            [df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()],
            axis=1,
        ).max(axis=1)
        return tr.rolling(period, min_periods=period).mean()

    @staticmethod
    def ema(df, period):
        return df["close"].ewm(span=period, adjust=False, min_periods=period).mean()

    @staticmethod
    def volume_expansion(df, idx, period=20, multiplier=1.5):
        if idx < period:
            return False
        avg = df["volume"].iloc[idx - period:idx].mean()
        return bool(avg > 0 and df["volume"].iloc[idx] >= avg * multiplier)

    @staticmethod
    def displacement(df, idx, atr_multiplier=1.2, volume_multiplier=1.5, body_ratio_min=0.70):
        if idx < 20:
            return False
        atr = SMCMath.calculate_atr(df).iloc[idx]
        if pd.isna(atr) or atr <= 0:
            return False
        o, c = df["open"].iloc[idx], df["close"].iloc[idx]
        h, l = df["high"].iloc[idx], df["low"].iloc[idx]
        body = abs(c - o)
        rng = h - l
        if rng <= 0:
            return False
        return bool(
            body >= atr * atr_multiplier
            and body / rng >= body_ratio_min
            and SMCMath.volume_expansion(df, idx, 20, volume_multiplier)
        )

    # Canonical implementation from the source's detect_swings logic.
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
            if highs[i] > highs[i-left:i].max() and highs[i] > highs[i+1:i+right+1].max():
                data.loc[data.index[i], "swing_high"] = True
            if lows[i] < lows[i-left:i].min() and lows[i] < lows[i+1:i+right+1].min():
                data.loc[data.index[i], "swing_low"] = True
        return data

    # Compatibility alias — fixes the source's SMCMath.get_swings crash.
    @staticmethod
    def get_swings(df, left=3, right=3):
        return SMCMath.detect_swings(df, left, right)

    # Compatibility alias — fixes the source's SMCMath.check_displacement crash.
    @staticmethod
    def check_displacement(df, idx=-1):
        if idx < 0:
            idx = len(df) + idx
        return SMCMath.displacement(df, idx)

    @staticmethod
    def detect_liquidity_sweep(df, direction, lookback=20):
        if len(df) < lookback + 5:
            return False
        current = df.iloc[-1]
        previous = df.iloc[-lookback:-1]
        if direction == "LONG":
            level = previous["low"].min()
            return bool(current["low"] < level and current["close"] > level)
        if direction == "SHORT":
            level = previous["high"].max()
            return bool(current["high"] > level and current["close"] < level)
        return False

    @staticmethod
    def detect_mss(df, direction):
        if len(df) < 15:
            return False
        swings = SMCMath.detect_swings(df)
        current_close = swings["close"].iloc[-1]
        if direction == "LONG":
            x = swings[swings["swing_high"]]
            return bool(not x.empty and current_close > x["high"].iloc[-1])
        if direction == "SHORT":
            x = swings[swings["swing_low"]]
            return bool(not x.empty and current_close < x["low"].iloc[-1])
        return False

    @staticmethod
    def detect_bos(df, direction, lookback=10):
        if len(df) < lookback + 2:
            return False
        current = df.iloc[-1]
        previous = df.iloc[-lookback:-1]
        if direction == "LONG":
            return bool(current["close"] > previous["high"].max())
        if direction == "SHORT":
            return bool(current["close"] < previous["low"].min())
        return False

    @staticmethod
    def detect_fvg(df):
        bullish, bearish = [], []
        for i in range(2, len(df)):
            c1, c3 = df.iloc[i - 2], df.iloc[i]
            if c1["high"] < c3["low"]:
                bullish.append({"type": "BULLISH_FVG", "index": i, "low": float(c1["high"]), "high": float(c3["low"])})
            if c1["low"] > c3["high"]:
                bearish.append({"type": "BEARISH_FVG", "index": i, "low": float(c3["high"]), "high": float(c1["low"])})
        return {"bullish": bullish, "bearish": bearish}

    @staticmethod
    def candle_direction(df, idx=-1):
        c = df.iloc[idx]
        if c["close"] > c["open"]:
            return "BULLISH"
        if c["close"] < c["open"]:
            return "BEARISH"
        return "NEUTRAL"

    @staticmethod
    def volatility_regime(df):
        atr = SMCMath.calculate_atr(df)
        if atr.dropna().empty:
            return "UNKNOWN"
        cur = atr.iloc[-1]
        med = atr.dropna().tail(50).median()
        if cur > med * 1.5:
            return "HIGH"
        if cur < med * 0.7:
            return "LOW"
        return "NORMAL"


# ============================================================
# REGIME / HTF / BTC
# ============================================================
class MarketRegimeEngine:
    @staticmethod
    def get_trend(df, fast=20, slow=50):
        if df is None or len(df) < slow:
            return "UNKNOWN"
        ef = df["close"].ewm(span=fast, adjust=False).mean().iloc[-1]
        es = df["close"].ewm(span=slow, adjust=False).mean().iloc[-1]
        price = df["close"].iloc[-1]
        if price > ef > es:
            return "BULLISH"
        if price < ef < es:
            return "BEARISH"
        return "CHOP"

    @staticmethod
    def btc_regime(df_1d, df_4h, df_1h):
        trends = {
            "1d": MarketRegimeEngine.get_trend(df_1d),
            "4h": MarketRegimeEngine.get_trend(df_4h),
            "1h": MarketRegimeEngine.get_trend(df_1h),
        }
        values = list(trends.values())
        bull = values.count("BULLISH")
        bear = values.count("BEARISH")
        if bull == 3:
            regime = "STRONG_BULLISH"
        elif bear == 3:
            regime = "STRONG_BEARISH"
        elif bull >= 2:
            regime = "BULLISH"
        elif bear >= 2:
            regime = "BEARISH"
        else:
            regime = "CHOP"
        return {"regime": regime, **trends}

    @staticmethod
    def allow_direction(regime, direction):
        if direction == "LONG":
            return regime in {"BULLISH", "STRONG_BULLISH"}
        if direction == "SHORT":
            return regime in {"BEARISH", "STRONG_BEARISH"}
        return False


_BTC_CACHE = {"regime": "CHOP", "details": {}, "timestamp": 0.0}
_BTC_CACHE_SECONDS = 30


def get_btc_regime(force=False):
    now = time.time()
    if not force and now - _BTC_CACHE["timestamp"] < _BTC_CACHE_SECONDS:
        return _BTC_CACHE["regime"], _BTC_CACHE["details"]
    try:
        btc_1d = KuCoinLive.get_klines("BTC-USDT", "1day", 100)
        btc_4h = KuCoinLive.get_klines("BTC-USDT", "4hour", 100)
        btc_1h = KuCoinLive.get_klines("BTC-USDT", "1hour", 100)
        if any(x.empty for x in (btc_1d, btc_4h, btc_1h)):
            return "CHOP", {}
        result = MarketRegimeEngine.btc_regime(btc_1d, btc_4h, btc_1h)
        _BTC_CACHE.update({"regime": result["regime"], "details": result, "timestamp": now})
        return result["regime"], result
    except Exception:
        return "CHOP", {}


class HTFAlignmentEngine:
    @staticmethod
    def analyze(df_1d, df_4h, df_1h, direction):
        trends = {
            "1d": MarketRegimeEngine.get_trend(df_1d),
            "4h": MarketRegimeEngine.get_trend(df_4h),
            "1h": MarketRegimeEngine.get_trend(df_1h),
        }
        wanted = "BULLISH" if direction == "LONG" else "BEARISH"
        return {"aligned": all(v == wanted for v in trends.values()), **trends}


# ============================================================
# POI ENGINE
# ============================================================
class POIEngine:
    def __init__(self, proximity_pct=1.5, max_fvg_age=40, max_ob_age=60):
        self.proximity_pct = proximity_pct
        self.max_fvg_age = max_fvg_age
        self.max_ob_age = max_ob_age

    @staticmethod
    def detect_fvg(df):
        return [*SMCMath.detect_fvg(df)["bullish"], *SMCMath.detect_fvg(df)["bearish"]]

    @staticmethod
    def detect_order_blocks(df):
        order_blocks = []
        if len(df) < 10:
            return order_blocks
        for i in range(2, len(df) - 2):
            candle, n1, n2 = df.iloc[i], df.iloc[i + 1], df.iloc[i + 2]
            if candle["close"] < candle["open"] and n1["close"] > n1["open"] and n2["close"] > n2["open"] and n2["close"] > candle["high"]:
                order_blocks.append({"type": "BULLISH_OB", "index": i, "low": float(candle["low"]), "high": float(candle["high"])})
            if candle["close"] > candle["open"] and n1["close"] < n1["open"] and n2["close"] < n2["open"] and n2["close"] < candle["low"]:
                order_blocks.append({"type": "BEARISH_OB", "index": i, "low": float(candle["low"]), "high": float(candle["high"])})
        return order_blocks

    def filter_active_pois(self, pois, current_index):
        active = []
        for poi in pois:
            age = current_index - poi["index"]
            if poi["type"].endswith("_FVG") and age <= self.max_fvg_age:
                active.append(poi)
            elif poi["type"].endswith("_OB") and age <= self.max_ob_age:
                active.append(poi)
        return active

    @staticmethod
    def directional_pois(pois, direction):
        allowed = {"BULLISH_FVG", "BULLISH_OB"} if direction == "LONG" else {"BEARISH_FVG", "BEARISH_OB"}
        return [p for p in pois if p["type"] in allowed]

    def check_proximity(self, price, pois):
        nearby = []
        for poi in pois:
            low, high = poi["low"], poi["high"]
            inside = low <= price <= high
            if inside:
                distance = 0.0
            elif price > high:
                distance = (price - high) / price * 100
            else:
                distance = (low - price) / price * 100
            if inside or distance <= self.proximity_pct:
                x = dict(poi)
                x["distance_pct"] = round(distance, 4)
                x["inside"] = inside
                nearby.append(x)
        return bool(nearby), nearby

    @staticmethod
    def select_best_poi(pois):
        if not pois:
            return None
        return sorted(pois, key=lambda x: (not x.get("inside", False), x.get("distance_pct", 999)))[0]

    def analyze(self, df, price, direction):
        if df is None or df.empty:
            return {"valid": False, "proximity": False, "best_poi": None, "pois": []}
        current_index = len(df) - 1
        all_pois = self.detect_fvg(df) + self.detect_order_blocks(df)
        active = self.filter_active_pois(all_pois, current_index)
        directional = self.directional_pois(active, direction)
        proximity, nearby = self.check_proximity(price, directional)
        best = self.select_best_poi(nearby)
        return {
            "valid": bool(directional),
            "proximity": proximity,
            "best_poi": best,
            "pois": nearby,
            "total_active": len(active),
            "directional_count": len(directional),
        }


# ============================================================
# RISK / SCORING
# ============================================================
class RiskEngine:
    def __init__(self, risk_usd=10.0, taker_fee=0.0008, maker_fee=0.0006, slippage=0.0005):
        self.risk_usd = float(risk_usd)
        self.taker_fee = float(taker_fee)
        self.maker_fee = float(maker_fee)
        self.slippage = float(slippage)

    def calculate_levels(self, df, direction):
        atr = SMCMath.calculate_atr(df).iloc[-1]
        if pd.isna(atr) or atr <= 0:
            return None
        price = float(df["close"].iloc[-1])
        recent_high = float(df["high"].tail(10).max())
        recent_low = float(df["low"].tail(10).min())
        if direction == "LONG":
            sl = min(recent_low - atr * 0.20, price - atr)
            risk_per_unit = price - sl
            tp = price + risk_per_unit * 2.5
        else:
            sl = max(recent_high + atr * 0.20, price + atr)
            risk_per_unit = sl - price
            tp = price - risk_per_unit * 2.5
        if risk_per_unit <= 0:
            return None
        return {"entry": price, "sl": sl, "tp": tp, "atr": float(atr), "risk_per_unit": risk_per_unit}

    def calculate_execution(self, levels, direction):
        if not levels:
            return {"valid": False, "reason": "NO RISK LEVELS"}
        entry, sl, tp = levels["entry"], levels["sl"], levels["tp"]
        if direction == "LONG":
            effective_entry = entry * (1 + self.slippage)
            effective_sl = sl * (1 - self.slippage)
            risk_unit = effective_entry - effective_sl
            reward_unit = tp - effective_entry
        else:
            effective_entry = entry * (1 - self.slippage)
            effective_sl = sl * (1 + self.slippage)
            risk_unit = effective_sl - effective_entry
            reward_unit = effective_entry - tp
        if risk_unit <= 0 or reward_unit <= 0:
            return {"valid": False, "reason": "INVALID RISK/REWARD"}
        size = self.risk_usd / risk_unit
        entry_fee = effective_entry * size * self.taker_fee
        exit_fee = tp * size * self.maker_fee
        fees = entry_fee + exit_fee
        gross_reward = reward_unit * size
        net_reward = gross_reward - fees
        net_risk = self.risk_usd + fees
        rr = net_reward / net_risk if net_risk > 0 else 0
        return {
            "valid": rr >= 2.0,
            "entry": round(effective_entry, 6),
            "sl": round(effective_sl, 6),
            "tp": round(tp, 6),
            "position_size": round(size, 6),
            "fees": round(fees, 6),
            "risk_usd": round(net_risk, 2),
            "net_reward": round(net_reward, 2),
            "real_rr": round(rr, 2),
        }


class SignalEngine:
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
        "realistic_rr": 5,
    }

    @classmethod
    def evaluate(cls, checks):
        score = sum(cls.WEIGHTS[k] for k, v in checks.items() if v)
        score = min(100, score)
        mandatory = all(checks.values())
        if mandatory and score >= 90:
            grade, status = "A+ SETUP", "VALID SIGNAL"
        elif mandatory and score >= 85:
            grade, status = "A SETUP", "VALID SIGNAL"
        elif checks.get("liquidity_sweep") and any(checks.get(k) for k in ("mss_choch", "displacement", "bos_confirmed")):
            grade, status = "WATCH", "SETUP FORMING"
        else:
            grade, status = "NO TRADE", "NO TRADE"
        return {"score": score, "grade": grade, "status": status, "checks": checks}


# ============================================================
# ANALYSIS
# ============================================================
def fetch_bundle(symbol):
    specs = {
        "1d": ("1day", CONFIG["history_1d"]),
        "4h": ("4hour", CONFIG["history_4h"]),
        "1h": ("1hour", CONFIG["history_1h"]),
        "15m": ("15min", CONFIG["history_15m"]),
        "5m": ("5min", CONFIG["history_5m"]),
    }
    data, errors = {}, []
    for key, (interval, limit) in specs.items():
        df = KuCoinLive.get_klines(symbol, interval, limit)
        ok, reason = DataQualityGuard.validate(df, interval=interval)
        if not ok:
            errors.append(f"{key}: {reason}")
            data[key] = pd.DataFrame()
        elif StaleDataGuard.is_stale(df, interval):
            errors.append(f"{key}: STALE DATA")
            data[key] = pd.DataFrame()
        else:
            data[key] = df
    return data, errors


def analyze_direction(symbol, data, errors, direction, btc_regime, btc_details):
    if errors or any(data[k].empty for k in data):
        return {
            "symbol": symbol, "direction": direction, "price": None,
            "btc_regime": btc_regime, "status": "NO TRADE", "grade": "NO TRADE",
            "score": 0, "checks": {}, "reason": "; ".join(errors) or "DATA FETCH FAILED",
            "trade": None, "poi": None, "htf": {}, "volatility": "UNKNOWN",
        }

    df_1d, df_4h, df_1h, df_15m, df_5m = (data[x] for x in ("1d", "4h", "1h", "15m", "5m"))
    price = float(df_5m["close"].iloc[-1])

    htf = HTFAlignmentEngine.analyze(df_1d, df_4h, df_1h, direction)
    poi = POIEngine(CONFIG["poi_proximity_percent"]).analyze(df_15m, price, direction)
    structure = SMCMath.get_swings(df_5m.copy())

    liquidity_sweep = SMCMath.detect_liquidity_sweep(structure, direction, lookback=20)
    mss_confirmed = SMCMath.detect_mss(structure, direction)
    displacement = SMCMath.check_displacement(structure, -1)
    bos_confirmed = SMCMath.detect_bos(structure, direction, lookback=10)

    best_poi = poi.get("best_poi")
    retest_confirmed = False
    if best_poi:
        low, high = best_poi["low"], best_poi["high"]
        tolerance = price * 0.003
        retest_confirmed = bool(low - tolerance <= price <= high + tolerance)

    risk = RiskEngine(
        CONFIG["account_risk_usd"], CONFIG["taker_fee"], CONFIG["maker_fee"], CONFIG["slippage"]
    )
    levels = risk.calculate_levels(df_5m, direction)
    trade = risk.calculate_execution(levels, direction)
    real_rr = float(trade.get("real_rr", 0))

    btc_ok = MarketRegimeEngine.allow_direction(btc_regime, direction)
    checks = {
        "btc_regime": btc_ok,
        "htf_alignment": htf["aligned"],
        "directional_poi": poi["valid"],
        "poi_proximity": poi["proximity"],
        "liquidity_sweep": liquidity_sweep,
        "mss_choch": mss_confirmed,
        "displacement": displacement,
        "bos_confirmed": bos_confirmed,
        "retest_confirmed": retest_confirmed,
        "realistic_rr": real_rr >= 2.0,
    }
    result = SignalEngine.evaluate(checks)

    return {
        "symbol": symbol,
        "direction": direction,
        "price": round(price, 8),
        "btc_regime": btc_regime,
        "btc_details": btc_details,
        "status": result["status"],
        "grade": result["grade"],
        "score": result["score"],
        "checks": result["checks"],
        "trade": trade if trade.get("valid") else None,
        "raw_trade": trade,
        "poi": poi,
        "htf": htf,
        "volatility": SMCMath.volatility_regime(df_5m),
        "candle": SMCMath.candle_direction(df_5m),
        "reason": "All mandatory gates passed" if result["status"] == "VALID SIGNAL" else "Mandatory gate failed",
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


def analyze_symbol(symbol, force_btc=False):
    data, errors = fetch_bundle(symbol)
    btc_regime, btc_details = get_btc_regime(force=force_btc)
    long_result = analyze_direction(symbol, data, errors, "LONG", btc_regime, btc_details)
    short_result = analyze_direction(symbol, data, errors, "SHORT", btc_regime, btc_details)

    # Choose the stronger direction; if both are weak, expose NO TRADE/neutral.
    candidates = [long_result, short_result]
    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]
    best["alternatives"] = {"LONG": long_result, "SHORT": short_result}
    best["data_errors"] = errors
    return best


# ============================================================
# SESSION STATE / AUDIT
# ============================================================
if "audit" not in st.session_state:
    st.session_state.audit = AuditLogger(CONFIG["max_audit_records"])
if "last_results" not in st.session_state:
    st.session_state.last_results = {}
if "last_scan" not in st.session_state:
    st.session_state.last_scan = None
if "scan_counter" not in st.session_state:
    st.session_state.scan_counter = 0


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("⚙️ Engine Controls")
selected_symbols = st.sidebar.multiselect(
    "Symbols",
    CONFIG["symbols"],
    default=CONFIG["symbols"],
)
auto_refresh = st.sidebar.checkbox("Auto refresh", value=True)
refresh_seconds = st.sidebar.slider("Refresh seconds", 15, 120, CONFIG["refresh_seconds"], 5)
force_btc = st.sidebar.checkbox("Force BTC refresh", value=False)
risk_usd = st.sidebar.number_input("Risk / trade (USD)", min_value=1.0, max_value=1000.0, value=CONFIG["account_risk_usd"], step=1.0)
CONFIG["account_risk_usd"] = float(risk_usd)

if st.sidebar.button("🔄 Scan Now", use_container_width=True):
    st.session_state.force_scan = True
else:
    st.session_state.force_scan = False

st.sidebar.divider()
st.sidebar.caption(f"SQLite: `{CONFIG['database_path']}`")
st.sidebar.caption(f"DB audit rows: {DB.count()}")
st.sidebar.caption("Public KuCoin market API only")


# ============================================================
# HEADER
# ============================================================
st.markdown(f'<div class="cosmic-title">🌌 {APP_NAME}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="cosmic-sub">Live multi-timeframe SMC radar • SQLite persistence • KuCoin public market data</div>',
    unsafe_allow_html=True,
)

if not selected_symbols:
    st.warning("Select at least one symbol from the sidebar.")
    st.stop()

# Scan button always executes immediately. Auto refresh uses Streamlit fragment when available.

def perform_scan():
    results = {}
    audit_before = len(st.session_state.audit.records)
    progress = st.progress(0, text="Starting scan...")
    for idx, symbol in enumerate(selected_symbols):
        try:
            result = analyze_symbol(symbol, force_btc=force_btc)
            results[symbol] = result
            st.session_state.audit.log(
                symbol,
                result["status"],
                f"{result['direction']} | {result['grade']} | {result['reason']}",
                result["score"],
            )
        except Exception as exc:
            result = {
                "symbol": symbol, "direction": "NEUTRAL", "price": None,
                "btc_regime": "CHOP", "status": "ENGINE ERROR", "grade": "ERROR",
                "score": 0, "checks": {}, "trade": None,
                "reason": f"{type(exc).__name__}: {exc}", "alternatives": {},
                "data_errors": [],
            }
            results[symbol] = result
            st.session_state.audit.log(symbol, "ENGINE ERROR", result["reason"], 0)
        progress.progress((idx + 1) / len(selected_symbols), text=f"Scanned {symbol}")
    progress.empty()
    st.session_state.last_results = results
    st.session_state.last_scan = datetime.now(timezone.utc)
    st.session_state.scan_counter += 1


# Use fragment auto-run where available. The fragment only contains the scan trigger area.
# Main dashboard remains deterministic and does not use an infinite while-loop.
if hasattr(st, "fragment") and auto_refresh:
    @st.fragment(run_every=refresh_seconds)
    def live_scan_fragment():
        if st.session_state.get("force_scan", False) or not st.session_state.last_results:
            perform_scan()
        else:
            # Refresh scan on every fragment cycle.
            perform_scan()
    live_scan_fragment()
else:
    if st.session_state.get("force_scan", False) or not st.session_state.last_results:
        perform_scan()


results = st.session_state.last_results

# ============================================================
# TOP STATUS
# ============================================================
btc_regime = next(iter(results.values())).get("btc_regime", "CHOP") if results else "CHOP"
valid_count = sum(1 for r in results.values() if r.get("status") == "VALID SIGNAL")
watch_count = sum(1 for r in results.values() if r.get("status") == "SETUP FORMING")
error_count = sum(1 for r in results.values() if r.get("status") == "ENGINE ERROR")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Engine", "ONLINE")
c2.metric("BTC Regime", btc_regime)
c3.metric("Valid Signals", valid_count)
c4.metric("Watch", watch_count)
c5.metric("Errors", error_count)

if st.session_state.last_scan:
    st.caption(
        f"Last scan: {st.session_state.last_scan.strftime('%Y-%m-%d %H:%M:%S UTC')} • "
        f"Scan #{st.session_state.scan_counter} • Auto refresh: {'ON' if auto_refresh else 'OFF'}"
    )

# ============================================================
# RADAR TABLE
# ============================================================
st.subheader("📡 Live Radar")
rows = []
for symbol, r in results.items():
    rows.append({
        "Symbol": symbol,
        "Direction": r.get("direction", "NEUTRAL"),
        "Price": r.get("price"),
        "BTC": r.get("btc_regime", "CHOP"),
        "Score": r.get("score", 0),
        "Grade": r.get("grade", "NO TRADE"),
        "Status": r.get("status", "NO TRADE"),
        "Volatility": r.get("volatility", "UNKNOWN"),
        "Candle": r.get("candle", "UNKNOWN"),
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ============================================================
# DETAIL CARDS
# ============================================================
st.subheader("🔎 Signal Details")
for symbol, r in results.items():
    direction = r.get("direction", "NEUTRAL")
    with st.expander(f"{symbol}  •  {direction}  •  {r.get('grade', 'NO TRADE')}  •  Score {r.get('score', 0)}/100", expanded=False):
        left, mid, right = st.columns([1, 1.2, 1])
        with left:
            st.metric("Price", r.get("price") if r.get("price") is not None else "—")
            st.write(f"**BTC Regime:** {r.get('btc_regime', 'CHOP')}")
            st.write(f"**HTF:** {r.get('htf', {})}")
            st.write(f"**Volatility:** {r.get('volatility', 'UNKNOWN')}")
        with mid:
            st.markdown("**SMC Gate Checks**")
            checks = r.get("checks", {})
            if checks:
                check_df = pd.DataFrame(
                    [{"Check": k.replace("_", " ").upper(), "Result": "✅ PASS" if v else "❌ FAIL"} for k, v in checks.items()]
                )
                st.dataframe(check_df, use_container_width=True, hide_index=True)
            else:
                st.info(r.get("reason", "No check data"))
        with right:
            trade = r.get("trade")
            if trade:
                st.success("VALID TRADE PARAMETERS")
                st.metric("Entry", trade["entry"])
                st.metric("Stop Loss", trade["sl"])
                st.metric("Take Profit", trade["tp"])
                st.write(f"**Real RR:** 1:{trade['real_rr']}")
                st.write(f"**Position Size:** {trade['position_size']}")
                st.write(f"**Estimated Fees:** ${trade['fees']}")
            else:
                st.warning(r.get("reason", "No executable trade"))

        poi = r.get("poi") or {}
        best_poi = poi.get("best_poi")
        if best_poi:
            st.markdown(
                f"**Best POI:** `{best_poi['type']}` • "
                f"Zone `{best_poi['low']:.6f} – {best_poi['high']:.6f}` • "
                f"Distance `{best_poi.get('distance_pct', 0):.3f}%`"
            )

        alternatives = r.get("alternatives", {})
        if alternatives:
            st.markdown("**Direction comparison**")
            st.dataframe(
                pd.DataFrame([
                    {
                        "Direction": d,
                        "Score": x.get("score", 0),
                        "Grade": x.get("grade", "NO TRADE"),
                        "Status": x.get("status", "NO TRADE"),
                    }
                    for d, x in alternatives.items()
                ]),
                use_container_width=True,
                hide_index=True,
            )

        if r.get("data_errors"):
            st.error("Data: " + " | ".join(r["data_errors"]))

# ============================================================
# BTC REGIME DETAILS
# ============================================================
st.subheader("₿ BTC Regime")
btc_regime_value, btc_details = get_btc_regime(force=False)
btc_cols = st.columns(4)
btc_cols[0].metric("Regime", btc_regime_value)
for idx, tf in enumerate(("1d", "4h", "1h"), start=1):
    btc_cols[idx].metric(tf.upper(), btc_details.get(tf, "UNKNOWN"))

# ============================================================
# AUDIT LOG
# ============================================================
st.subheader("🧾 SQLite Audit Log")
audit_rows = st.session_state.audit.latest(50)
if audit_rows:
    audit_df = pd.DataFrame(audit_rows)
    st.dataframe(audit_df, use_container_width=True, hide_index=True)
else:
    st.info("No audit records yet.")

# ============================================================
# FOOTER / AUTO REFRESH NOTE
# ============================================================
st.divider()
st.caption(
    "COSMIC 108 V3.1 • Dashboard mode. The engine is analysis-only and does not place orders. "
    "BTC regime fails closed to CHOP when unavailable."
)
