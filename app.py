import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import ccxt
import sqlite3
import os
from datetime import datetime, timedelta, timezone
import time

# ==========================================
# 🚀 COSMIC 108 — SMART MARKET RADAR V1.7.4
# Institutional Upgrade: Active FVG/OB, Trigger Persistence & ATR SL
# ==========================================

st.set_page_config(page_title="COSMIC 108 | Institutional Radar V1.7.4", layout="wide")

DB_FILE = "cosmic108.db"
TARGET_COINS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
TIMEFRAMES = ["1d", "4h", "1h", "15m", "5m"]
ANALYSIS_VERSION = "V1.7.4_INST"

# ==========================================
# 1. HARDENED SQLITE CONNECTION & SCHEMA
# ==========================================
def get_db_connection():
    """ Safe connection with concurrency timeout and multi-thread allowance """
    conn = sqlite3.connect(DB_FILE, timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS candles (
                    symbol TEXT, timeframe TEXT, timestamp DATETIME,
                    open REAL, high REAL, low REAL, close REAL, volume REAL,
                    UNIQUE(symbol, timeframe, timestamp)
                 )''')
                 
    c.execute('''CREATE TABLE IF NOT EXISTS analysis (
                    symbol TEXT, timeframe TEXT, timestamp DATETIME,
                    open REAL, high REAL, low REAL, close REAL, volume REAL,
                    ema_50 REAL, ema_200 REAL, rsi_14 REAL, atr_14 REAL, vol_sma_20 REAL,
                    ema_bias TEXT, structure_state TEXT, structure_event TEXT,
                    break_distance REAL, last_swing_high REAL, last_swing_low REAL,
                    analysis_version TEXT, calc_timestamp DATETIME,
                    UNIQUE(symbol, timeframe, timestamp)
                 )''')
                 
    c.execute('''CREATE TABLE IF NOT EXISTS phase5_features (
                    symbol TEXT, timeframe TEXT, timestamp DATETIME,
                    liquidity_type TEXT, liquidity_level REAL, liquidity_swept INTEGER,
                    fvg_type TEXT, fvg_high REAL, fvg_low REAL, fvg_status TEXT,
                    ob_type TEXT, ob_high REAL, ob_low REAL, ob_status TEXT,
                    phase5_score INTEGER, phase5_confluence INTEGER,
                    analysis_version TEXT, calc_timestamp DATETIME,
                    UNIQUE(symbol, timeframe, timestamp)
                 )''')
                 
    c.execute('''CREATE TABLE IF NOT EXISTS signals (
                    symbol TEXT, timeframe TEXT, timestamp DATETIME, direction TEXT,
                    score INTEGER, grade TEXT, macro_bias TEXT, htf_bias TEXT,
                    ltf_bias TEXT, mtf_alignment TEXT, bos_choch TEXT,
                    phase5_score INTEGER, momentum TEXT, volatility TEXT,
                    volume_quality TEXT, entry_price REAL, stop_loss REAL,
                    take_profit_1 REAL, take_profit_2 REAL, risk_reward REAL,
                    trigger INTEGER, signal_status TEXT, 
                    analysis_version TEXT, created_at DATETIME,
                    UNIQUE(symbol, timeframe, timestamp)
                 )''')
    conn.commit()
    conn.close()

init_db()

# ==========================================
# 2. LIVE CCXT DATA INGESTION ENGINE
# ==========================================
@st.cache_resource
def get_exchange_instance():
    return ccxt.kucoin({'enableRateLimit': True})

def fetch_and_store_live_candles(progress_bar, status_text):
    exchange = get_exchange_instance()
    conn = get_db_connection()
    c = conn.cursor()
    
    total_tasks = len(TARGET_COINS) * len(TIMEFRAMES)
    task_count = 0
    new_candles_count = 0
    
    for coin in TARGET_COINS:
        for tf in TIMEFRAMES:
            task_count += 1
            status_text.text(f"Ingesting: {coin} [{tf}] ({task_count}/{total_tasks})")
            progress_bar.progress(task_count / total_tasks)
            
            try:
                c.execute("SELECT MAX(timestamp) FROM candles WHERE symbol = ? AND timeframe = ?", (coin, tf))
                last_ts_str = c.fetchone()[0]
                
                since = None
                limit = 400
                if last_ts_str:
                    last_dt = datetime.fromisoformat(last_ts_str)
                    since = int(last_dt.timestamp() * 1000) + 1 # Avoid duplicate boundary overlap
                    limit = 100
                
                ohlcv = exchange.fetch_ohlcv(coin, tf, since=since, limit=limit)
                
                for row in ohlcv:
                    timestamp = datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc).isoformat()
                    c.execute("""INSERT OR IGNORE INTO candles 
                                 (symbol, timeframe, timestamp, open, high, low, close, volume) 
                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", 
                              (coin, tf, timestamp, row[1], row[2], row[3], row[4], row[5]))
                    if c.rowcount > 0:
                        new_candles_count += 1
                conn.commit()
            except Exception as e:
                print(f"[Ingestion Error] {coin} {tf}: {e}")
                time.sleep(1)
                
    conn.close()
    return new_candles_count

def tf_to_seconds(tf):
    units = {'m': 60, 'h': 3600, 'd': 86400}
    return int(tf[:-1]) * units[tf[-1]]

def filter_closed_candles(df, tf):
    if df.empty: return df
    tf_secs = tf_to_seconds(tf)
    current_utc = datetime.now(timezone.utc)
    df['close_time'] = df['timestamp'] + pd.to_timedelta(tf_secs, unit='s')
    return df[df['close_time'] <= current_utc].drop(columns=['close_time']).copy()

def load_raw_candles(symbol, timeframe, limit=500):
    conn = get_db_connection()
    query = """
        SELECT * FROM (
            SELECT * FROM candles 
            WHERE symbol = ? AND timeframe = ? 
            ORDER BY timestamp DESC LIMIT ?
        ) ORDER BY timestamp ASC
    """
    df = pd.read_sql_query(query, conn, params=(symbol, timeframe, limit))
    conn.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return filter_closed_candles(df, timeframe)

# ==========================================
# 3. V1.7.4 CAUSAL ANALYSIS & LIFECYCLE ENGINE
# ==========================================
def run_v174_analysis(df):
    # Minimum Guard: At least 250 candles required for accurate EMA 200 & structure
    if df.empty or len(df) < 250:
        return df, []
    df = df.copy()
    
    df['ema_50'] = ta.ema(df['close'], length=50)
    df['ema_200'] = ta.ema(df['close'], length=200) 
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    df['vol_sma_20'] = ta.sma(df['volume'], length=20)
    
    df['ema_bias'] = np.where(pd.notna(df['ema_200']) & (df['close'] > df['ema_200']), 'Bullish',
                     np.where(pd.notna(df['ema_200']) & (df['close'] < df['ema_200']), 'Bearish', 'Neutral'))
    
    df['rolling_max'] = df['high'].rolling(window=5, center=True).max()
    df['rolling_min'] = df['low'].rolling(window=5, center=True).min()
    df['swing_high'] = np.where(df['high'] == df['rolling_max'], df['high'], np.nan)
    df['swing_low'] = np.where(df['low'] == df['rolling_min'], df['low'], np.nan)
    df['swing_high'] = df['swing_high'].shift(2)
    df['swing_low'] = df['swing_low'].shift(2)
    
    states, events, break_dists, last_sh, last_sl = [], [], [], [], []
    current_state = 'NEUTRAL'
    current_sh, current_sl = np.nan, np.nan

    for row in df.itertuples():
        event = 'NONE'
        brk_dist = 0.0
        
        if pd.notna(row.swing_high): current_sh = row.swing_high
        if pd.notna(row.swing_low): current_sl = row.swing_low
        
        range_size = row.high - row.low
        body_size = abs(row.close - row.open)
        body_ratio = body_size / range_size if range_size > 0 else 0
        close_pos = (row.close - row.low) / range_size if range_size > 0 else 0
        atr_buffer = (row.atr_14 * 0.15) if pd.notna(row.atr_14) else 0.0
        
        if pd.notna(current_sh) and row.close > current_sh:
            dist = row.close - current_sh
            min_dist = min(max(current_sh * 0.001, atr_buffer), current_sh * 0.005)
            if dist >= min_dist and body_ratio >= 0.55 and close_pos >= 0.70:
                event = 'BOS_BULLISH' if current_state in ['BULLISH', 'NEUTRAL'] else 'CHOCH_BULLISH'
                current_state = 'BULLISH'
                brk_dist = dist / current_sh
                current_sh = np.nan
                
        elif pd.notna(current_sl) and row.close < current_sl:
            dist = current_sl - row.close
            min_dist = min(max(current_sl * 0.001, atr_buffer), current_sl * 0.005)
            if dist >= min_dist and body_ratio >= 0.55 and close_pos <= 0.30:
                event = 'BOS_BEARISH' if current_state in ['BEARISH', 'NEUTRAL'] else 'CHOCH_BEARISH'
                current_state = 'BEARISH'
                brk_dist = dist / current_sl
                current_sl = np.nan

        states.append(current_state)
        events.append(event)
        break_dists.append(brk_dist)
        last_sh.append(current_sh)
        last_sl.append(current_sl)

    df['structure_state'] = states
    df['structure_event'] = events
    df['break_distance'] = break_dists
    df['last_swing_high'] = last_sh
    df['last_swing_low'] = last_sl

    # FVG / OB Lifecycle with Cascade Protection
    p5_records = []
    active_fvgs = []
    active_obs = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        p5_records.append({
            "symbol": row['symbol'], "timeframe": row['timeframe'], "timestamp": row['timestamp'],
            "liquidity_type": 'NONE', "liquidity_level": np.nan, "liquidity_swept": 0,
            "fvg_type": 'NONE', "fvg_high": np.nan, "fvg_low": np.nan, "fvg_status": 'NONE',
            "ob_type": 'NONE', "ob_high": np.nan, "ob_low": np.nan, "ob_status": 'NONE',
            "phase5_score": 0, "phase5_confluence": 0
        })

    hist_sh = df['last_swing_high'].shift(1)
    hist_sl = df['last_swing_low'].shift(1)

    for i in range(2, len(df)):
        row = df.iloc[i]
        prev_sh = hist_sh.iloc[i]
        prev_sl = hist_sl.iloc[i]
        
        liq_swept, liq_type, liq_lvl = 0, 'NONE', np.nan
        if pd.notna(prev_sl) and row['low'] < prev_sl and row['close'] > prev_sl:
            liq_swept, liq_type, liq_lvl = 1, 'BULLISH_SWEEP', prev_sl
        elif pd.notna(prev_sh) and row['high'] > prev_sh and row['close'] < prev_sh:
            liq_swept, liq_type, liq_lvl = 1, 'BEARISH_SWEEP', prev_sh

        p5_records[i]["liquidity_type"] = liq_type
        p5_records[i]["liquidity_level"] = liq_lvl
        p5_records[i]["liquidity_swept"] = liq_swept

        for fvg in active_fvgs:
            if fvg['status'] == 'FRESH':
                old_status = fvg['status']
                if fvg['type'] == 'BULLISH_FVG':
                    if row['close'] < fvg['low']: fvg['status'] = 'INVALID'
                    elif row['low'] <= fvg['high']: fvg['status'] = 'MITIGATED'
                elif fvg['type'] == 'BEARISH_FVG':
                    if row['close'] > fvg['high']: fvg['status'] = 'INVALID'
                    elif row['high'] >= fvg['low']: fvg['status'] = 'MITIGATED'
                
                if fvg['status'] != old_status:
                    p5_records[fvg['record_idx']]['fvg_status'] = fvg['status']

        c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], row
        c2_ratio = abs(c2['close'] - c2['open']) / (c2['high'] - c2['low']) if (c2['high'] - c2['low']) > 0 else 0
        atr = c3['atr_14'] if pd.notna(c3['atr_14']) else 0
        
        if c2_ratio >= 0.55:
            if c3['low'] > c1['high'] and (c3['low'] - c1['high']) >= (0.5 * atr):
                p5_records[i].update({"fvg_type": 'BULLISH_FVG', "fvg_high": c3['low'], "fvg_low": c1['high'], "fvg_status": 'FRESH'})
                active_fvgs.append({'type': 'BULLISH_FVG', 'high': c3['low'], 'low': c1['high'], 'status': 'FRESH', 'record_idx': i})
            elif c3['high'] < c1['low'] and (c1['low'] - c3['high']) >= (0.5 * atr):
                p5_records[i].update({"fvg_type": 'BEARISH_FVG', "fvg_high": c1['low'], "fvg_low": c3['high'], "fvg_status": 'FRESH'})
                active_fvgs.append({'type': 'BEARISH_FVG', 'high': c1['low'], 'low': c3['high'], 'status': 'FRESH', 'record_idx': i})

        for ob in active_obs:
            if ob['status'] == 'FRESH':
                old_ob = ob['status']
                if ob['type'] == 'BULLISH_OB' and row['low'] <= ob['low']: ob['status'] = 'INVALID'
                elif ob['type'] == 'BEARISH_OB' and row['high'] >= ob['high']: ob['status'] = 'INVALID'
                if ob['status'] != old_ob:
                    p5_records[ob['record_idx']]['ob_status'] = ob['status']

        if row['structure_event'] in ['BOS_BULLISH', 'CHOCH_BULLISH']:
            for j in range(i-1, max(0, i-6), -1):
                if df.iloc[j]['close'] < df.iloc[j]['open']:
                    p5_records[i].update({"ob_type": 'BULLISH_OB', "ob_high": df.iloc[j]['open'], "ob_low": df.iloc[j]['low'], "ob_status": 'FRESH'})
                    active_obs.append({'type': 'BULLISH_OB', 'high': df.iloc[j]['open'], 'low': df.iloc[j]['low'], 'status': 'FRESH', 'record_idx': i})
                    break
        elif row['structure_event'] in ['BOS_BEARISH', 'CHOCH_BEARISH']:
            for j in range(i-1, max(0, i-6), -1):
                if df.iloc[j]['close'] > df.iloc[j]['open']:
                    p5_records[i].update({"ob_type": 'BEARISH_OB', "ob_high": df.iloc[j]['high'], "ob_low": df.iloc[j]['close'], "ob_status": 'FRESH'})
                    active_obs.append({'type': 'BEARISH_OB', 'high': df.iloc[j]['high'], 'low': df.iloc[j]['close'], 'status': 'FRESH', 'record_idx': i})
                    break

    for rec in p5_records:
        score = 0
        if rec["liquidity_swept"] == 1: score += 2
        if rec["fvg_status"] == 'FRESH': score += 1
        if rec["ob_status"] == 'FRESH': score += 1
        confluence = 1 if (rec["liquidity_swept"] == 1 and rec["fvg_status"] == 'FRESH' and rec["ob_status"] == 'FRESH') else 0
        if confluence: score += 1
        rec["phase5_score"] = min(score, 5)
        rec["phase5_confluence"] = confluence
        
    return df, p5_records

def save_to_db(df, p5_records):
    if df.empty: return
    now_str = datetime.now(timezone.utc).isoformat()
    conn = get_db_connection()
    c = conn.cursor()
    
    for _, row in df.iterrows():
        c.execute("""INSERT OR REPLACE INTO analysis 
                     (symbol, timeframe, timestamp, open, high, low, close, volume,
                      ema_50, ema_200, rsi_14, atr_14, vol_sma_20, ema_bias, 
                      structure_state, structure_event, break_distance, last_swing_high, last_swing_low, 
                      analysis_version, calc_timestamp)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                  (row['symbol'], row['timeframe'], row['timestamp'].isoformat(),
                   row.get('open'), row.get('high'), row.get('low'), row.get('close'), row.get('volume'),
                   row.get('ema_50'), row.get('ema_200'), row.get('rsi_14'), row.get('atr_14'), row.get('vol_sma_20'),
                   row.get('ema_bias'), row.get('structure_state'), row.get('structure_event'), row.get('break_distance'),
                   row.get('last_swing_high'), row.get('last_swing_low'), ANALYSIS_VERSION, now_str))
                   
    for rec in p5_records:
        c.execute("""INSERT OR REPLACE INTO phase5_features 
                     (symbol, timeframe, timestamp, liquidity_type, liquidity_level, liquidity_swept,
                      fvg_type, fvg_high, fvg_low, fvg_status,
                      ob_type, ob_high, ob_low, ob_status, phase5_score, phase5_confluence,
                      analysis_version, calc_timestamp)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                  (rec['symbol'], rec['timeframe'], rec['timestamp'].isoformat(),
                   rec['liquidity_type'], rec['liquidity_level'], rec['liquidity_swept'],
                   rec['fvg_type'], rec['fvg_high'], rec['fvg_low'], rec['fvg_status'],
                   rec['ob_type'], rec['ob_high'], rec['ob_low'], rec['ob_status'],
                   rec['phase5_score'], rec['phase5_confluence'], ANALYSIS_VERSION, now_str))
                   
    conn.commit()
    conn.close()

# ==========================================
# 4. REDESIGNED SCORING & RISK ENGINE (V1.7.4)
# ==========================================
def get_latest_record(table, symbol, tf):
    allowed_tables = {'analysis', 'phase5_features', 'signals'}
    if table not in allowed_tables: raise ValueError(f"Unauthorized table: {table}")
    conn = get_db_connection()
    query = f"SELECT * FROM {table} WHERE symbol = ? AND timeframe = ? ORDER BY timestamp DESC LIMIT 1"
    df = pd.read_sql_query(query, conn, params=(symbol, tf))
    conn.close()
    return df.iloc[0] if not df.empty else None

def get_active_phase5_confluence(symbol, tf):
    """ Scans recent historical rows to find any currently active (FRESH) FVG/OB """
    conn = get_db_connection()
    query = """
        SELECT * FROM phase5_features 
        WHERE symbol = ? AND timeframe = ? 
        ORDER BY timestamp DESC LIMIT 30
    """
    df = pd.read_sql_query(query, conn, params=(symbol, tf))
    conn.close()
    if df.empty: return 0
    
    # Check if there's any active FVG or OB in recent history
    fresh_fvg = any(df['fvg_status'] == 'FRESH')
    fresh_ob = any(df['ob_status'] == 'FRESH')
    recent_sweep = any(df['liquidity_swept'] == 1)
    
    score = 0
    if recent_sweep: score += 2
    if fresh_fvg: score += 1
    if fresh_ob: score += 1
    if recent_sweep and fresh_fvg and fresh_ob: score += 1
    return min(score, 5)

def calculate_risk_metrics_v174(m15_row, direction):
    """ ATR-buffered structural Risk/Reward Calculator """
    close = m15_row['close']
    atr = m15_row['atr_14'] if pd.notna(m15_row['atr_14']) else (close * 0.01)
    buffer = atr * 0.15 # ATR Buffer to avoid wick-hunts
    
    if direction == 'BULLISH':
        swing_low = m15_row['last_swing_low'] if pd.notna(m15_row['last_swing_low']) else (close - (atr * 1.5))
        sl = swing_low - buffer
        risk = close - sl
        if risk <= 0: risk = atr * 1.5
        tp1 = close + (risk * 1.5)
        tp2 = close + (risk * 2.5)
        rr = (tp1 - close) / risk
    else:
        swing_high = m15_row['last_swing_high'] if pd.notna(m15_row['last_swing_high']) else (close + (atr * 1.5))
        sl = swing_high + buffer
        risk = sl - close
        if risk <= 0: risk = atr * 1.5
        tp1 = close - (risk * 1.5)
        tp2 = close - (risk * 2.5)
        rr = (close - tp1) / risk
        
    return round(close, 2), round(sl, 2), round(tp1, 2), round(tp2, 2), round(rr, 2)

def save_signal_to_db(sig):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""INSERT OR REPLACE INTO signals 
                 (symbol, timeframe, timestamp, direction, score, grade, macro_bias, htf_bias,
                  ltf_bias, mtf_alignment, bos_choch, phase5_score, momentum, volatility,
                  volume_quality, entry_price, stop_loss, take_profit_1, take_profit_2, risk_reward,
                  trigger, signal_status, analysis_version, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (sig['symbol'], sig['timeframe'], sig['timestamp'], sig['direction'],
               sig['score'], sig['grade'], sig['macro_bias'], sig['htf_bias'],
               sig['ltf_bias'], sig['mtf_alignment'], sig['bos_choch'], sig['phase5_score'],
               sig['momentum'], sig['volatility'], sig['volume_quality'], sig['entry_price'],
               sig['stop_loss'], sig['take_profit_1'], sig['take_profit_2'], sig['risk_reward'],
               sig['trigger'], sig['status'], ANALYSIS_VERSION, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()

def check_5m_trigger_persistence(symbol):
    """ Lookback across the last 3 closed 5M candles for a BOS/CHOCH event """
    conn = get_db_connection()
    query = """
        SELECT structure_event FROM analysis 
        WHERE symbol = ? AND timeframe = '5m' 
        ORDER BY timestamp DESC LIMIT 3
    """
    df = pd.read_sql_query(query, conn, params=(symbol,))
    conn.close()
    if df.empty: return False
    
    events = df['structure_event'].tolist()
    return any(ev in ['BOS_BULLISH', 'CHOCH_BULLISH', 'BOS_BEARISH', 'CHOCH_BEARISH'] for ev in events)

def generate_v174_signal(symbol):
    d1 = get_latest_record('analysis', symbol, '1d')
    h4 = get_latest_record('analysis', symbol, '4h')
    h1 = get_latest_record('analysis', symbol, '1h')
    m15 = get_latest_record('analysis', symbol, '15m')
    
    if any(x is None for x in [d1, h4, h1, m15]):
        return None
        
    # Hard Requirement 1: Macro Alignment (30 Pts)
    if d1['structure_state'] == h4['structure_state'] and d1['structure_state'] != 'NEUTRAL':
        macro_direction = d1['structure_state']
    else:
        return {"direction": "⚪ NO TRADE", "reason": "1D & 4H Macro Structure Conflict", "status": "INVALID"}

    # Hard Requirement 2: 15M Direction Match
    if m15['structure_state'] != macro_direction:
        return {"direction": "⚪ NO TRADE", "reason": f"15M Counter-Trend ({m15['structure_state']})", "status": "INVALID"}

    score = 30  # Macro base points
    mtf_count = 2
    
    # HTF Alignment (15 Pts)
    if h1['structure_state'] == macro_direction: score += 15; mtf_count += 1
    
    # 15M Structure Event (15 Pts)
    event_quality = "Structure Aligned"
    if m15['structure_event'] in ['BOS_BULLISH', 'CHOCH_BULLISH', 'BOS_BEARISH', 'CHOCH_BEARISH']:
        score += 15; event_quality = "Active Breakout Event"

    # Liquidity Sweep & Confluence (Phase 5 - Up to 10 Pts)
    p5_score = get_active_phase5_confluence(symbol, '15m')
    score += (p5_score * 2) # Scaled to balance the 100-pt model cleanly

    # Momentum (RSI - 10 Pts)
    rsi = m15['rsi_14']
    mom_qual = "Healthy" if pd.notna(rsi) and ((macro_direction == 'BULLISH' and 45 <= rsi <= 70) or (macro_direction == 'BEARISH' and 30 <= rsi <= 55)) else "Neutral"
    if mom_qual == "Healthy": score += 10
            
    # Volume Quality (10 Pts)
    vol_qual = "Average"
    if pd.notna(m15['volume']) and pd.notna(m15['vol_sma_20']) and m15['vol_sma_20'] > 0:
        v_ratio = m15['volume'] / m15['vol_sma_20']
        if v_ratio > 1.50: score += 10; vol_qual = "Massive Surge"
        elif v_ratio >= 1.20: score += 7; vol_qual = "Strong"
        elif v_ratio >= 0.90: score += 4; vol_qual = "Average"

    # 5M Persistent Trigger Layer
    is_triggered = check_5m_trigger_persistence(symbol)

    grade, status = "🔴 NO TRADE", "WAITING TRIGGER"
    if score >= 90: grade = "🔥 A+ SNIPER"
    elif score >= 80: grade = "🟢 A SIGNAL"
    elif score >= 70: grade = "🟡 B SIGNAL"
    
    if is_triggered and score >= 70: status = "✅ VALID TRIGGER"
    else: status = "⏳ WAITING FOR 5M TRIGGER"

    ep, sl, tp1, tp2, rr = calculate_risk_metrics_v174(m15, macro_direction)

    sig_payload = {
        "symbol": symbol, "timeframe": "15m", "timestamp": m15['timestamp'], "direction": macro_direction,
        "score": min(score, 100), "grade": grade, "macro_bias": d1['structure_state'],
        "htf_bias": h4['structure_state'], "ltf_bias": m15['structure_state'],
        "mtf_alignment": f"{mtf_count}/4", "bos_choch": event_quality,
        "phase5_score": p5_score, "momentum": mom_qual, "volatility": "Healthy",
        "volume_quality": vol_qual, "entry_price": ep, "stop_loss": sl,
        "take_profit_1": tp1, "take_profit_2": tp2, "risk_reward": rr,
        "trigger": 1 if is_triggered else 0, "status": status, "reason": "Passed All Strict Institutional Filters"
    }
    
    save_signal_to_db(sig_payload)
    return sig_payload

# ==========================================
# 5. STREAMLIT UI - DASHBOARD V1.7.4
# ==========================================
st.title("🚀 COSMIC 108 | V1.7.4 Institutional Radar")
st.markdown("**Persistent Active FVG/OB Scans • 5M Lookback Triggers • ATR-Buffered Risk Management**")

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("📥 STEP 1: INGEST LIVE MARKET DATA", type="secondary", use_container_width=True):
        prog = st.progress(0)
        txt = st.empty()
        with st.spinner("Fetching live candles securely from KuCoin..."):
            cnt = fetch_and_store_live_candles(prog, txt)
        txt.text("✅ Live Ingestion Complete!")
        st.success(f"Successfully ingested {cnt} new candles.")
        
with col_btn2:
    if st.button("🧠 STEP 2: RUN V1.7.4 INSTITUTIONAL PIPELINE", type="primary", use_container_width=True):
        with st.spinner("Executing Causal Swings, Active Confluence & Risk Engine..."):
            for coin in TARGET_COINS:
                for tf in TIMEFRAMES:
                    raw_df = load_raw_candles(coin, tf)
                    analyzed_df, p5_records = run_v174_analysis(raw_df)
                    save_to_db(analyzed_df, p5_records)
            st.success("✅ V1.7.4 Pipeline Executed Successfully!")

st.divider()

selected_signal_coin = st.selectbox("Select Asset for V1.7.4 Signal Card", TARGET_COINS)
signal_data = generate_v174_signal(selected_signal_coin)

if signal_data:
    with st.container(border=True):
        st.markdown(f"### 🛡️ {selected_signal_coin} | Institutional Signal Card (V1.7.4)")
        
        if signal_data['direction'] == '⚪ NO TRADE':
            st.error(f"**⚪ NO TRADE**")
            st.markdown(f"*Reason: {signal_data['reason']}*")
        else:
            color = "green" if signal_data['direction'] == 'BULLISH' else "red"
            trigger_badge = "🔥 5M TRIGGERED" if signal_data['trigger'] == 1 else "⏳ WAITING FOR 5M TRIGGER"
            
            st.markdown(f"<h2 style='color:{color};'>{signal_data['direction']} — {signal_data['score']}/100</h2>", unsafe_allow_html=True)
            st.markdown(f"**{signal_data['grade']}** | Status: `{signal_data['status']}` | {trigger_badge}")
            
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MTF Alignment", signal_data['mtf_alignment'])
            c2.metric("Structure Quality", signal_data['bos_choch'])
            c3.metric("Active Confluence Score", f"{signal_data['phase5_score']} / 5 pts")
            c4.metric("Volume Quality", signal_data['volume_quality'])
            
            st.markdown("---")
            st.markdown("#### 🎯 Institutional Trade Plan (ATR-Buffered SL)")
            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("Entry Price", f"${signal_data['entry_price']}")
            r2.metric("Stop Loss", f"${signal_data['stop_loss']}")
            r3.metric("Take Profit 1", f"${signal_data['take_profit_1']}")
            r4.metric("Take Profit 2", f"${signal_data['take_profit_2']}")
            r5.metric("Est. R:R", f"1 : {signal_data['risk_reward']}")
else:
    st.info("Analysis pending. Please run Step 1 (Ingest) and Step 2 (Pipeline) above.")
