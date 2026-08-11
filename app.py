import streamlit as st
import pandas as pd
import ccxt.async_support as ccxt
import asyncio
from datetime import datetime, timezone
import os
import uuid
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import math
import threading

# ==========================================
# 1. Config, Constants & PostgreSQL Setup
# ==========================================
st.set_page_config(page_title="Cosmic 108 (Distributed PostgreSQL)", layout="wide")

SIMULATION_MODE = True
KU_KEY = os.environ.get("KUCOIN_API_KEY", "")
KU_SEC = os.environ.get("KUCOIN_API_SECRET", "")
KU_PASS = os.environ.get("KUCOIN_API_PASSWORD", "")

DB_URL = os.environ.get("DATABASE_URL")
if not DB_URL:
    raise ValueError("CRITICAL: DATABASE_URL (PostgreSQL) is required for distributed deployment.")

MIN_RR = 1.5
ENTRY_FEE, EXIT_FEE = 0.0006, 0.0006
SLIPPAGE_ALLOWANCE = 0.0002

def is_match(val1, val2, rel_tol=1e-5, abs_tol=1e-8):
    return math.isclose(float(val1), float(val2), rel_tol=rel_tol, abs_tol=abs_tol)

@st.cache_resource
def get_db_pool():
    db_pool = psycopg2.pool.ThreadedConnectionPool(1, 15, DB_URL)
    conn = db_pool.getconn()
    try:
        with conn.cursor() as c:
            c.execute('''CREATE TABLE IF NOT EXISTS active_trade (
                            id UUID PRIMARY KEY, coin TEXT UNIQUE, dir TEXT, 
                            planned_ep NUMERIC, actual_ep NUMERIC, 
                            struct_sl NUMERIC, struct_tp NUMERIC,
                            actual_sl NUMERIC, actual_tp NUMERIC,
                            planned_qty NUMERIC, actual_qty NUMERIC, 
                            status TEXT, main_id TEXT, sl_id TEXT, tp_id TEXT, 
                            created_at TIMESTAMPTZ, updated_at TIMESTAMPTZ)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS journal (
                            id UUID PRIMARY KEY, coin TEXT, dir TEXT, 
                            actual_ep NUMERIC, exit_ep NUMERIC, actual_qty NUMERIC,
                            gross_pnl NUMERIC, net_pnl NUMERIC, reason TEXT, closed_at TIMESTAMPTZ)''')
            conn.commit()
    finally:
        db_pool.putconn(conn)
    return db_pool

db_pool = get_db_pool()

class DistributedCoinLock:
    def __init__(self, coin):
        self.coin = coin
        self.conn = None
        self.locked = False

    def acquire(self):
        self.conn = db_pool.getconn()
        try:
            with self.conn.cursor() as c:
                c.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (self.coin,))
                self.locked = c.fetchone()[0]
                return self.locked
        except Exception as e:
            print(f"[Lock Error]: {e}")
            db_pool.putconn(self.conn)
            self.conn = None
            return False

    def release(self):
        if self.locked and self.conn:
            try:
                with self.conn.cursor() as c:
                    c.execute("SELECT pg_advisory_unlock(hashtext(%s))", (self.coin,))
                    self.conn.commit()
            except: pass
            finally:
                db_pool.putconn(self.conn)
                self.locked = False
                self.conn = None

def execute_db(query, params=(), fetchone=False, fetchall=False, commit=False, raise_err=False):
    conn = db_pool.getconn()
    res = None
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as c:
            c.execute(query, params)
            if fetchone: res = c.fetchone()
            elif fetchall: res = c.fetchall()
            if commit: conn.commit()
            return res if (fetchone or fetchall) else True
    except Exception as e:
        if commit: conn.rollback()
        if raise_err: raise e 
        print(f"[DB Error]: {e}")
        return False
    finally:
        db_pool.putconn(conn)

def update_state(trade_id, status, **kwargs):
    ALLOWED_COLS = {'actual_ep', 'actual_qty', 'actual_sl', 'actual_tp', 'status', 'main_id', 'sl_id', 'tp_id'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in ALLOWED_COLS}
    now_utc = datetime.now(timezone.utc)
    
    if not filtered_kwargs:
        query = "UPDATE active_trade SET status = %s, updated_at = %s WHERE id = %s"
        params = (status, now_utc, trade_id)
    else:
        set_clause = ", ".join([f"{k} = %s" for k in filtered_kwargs.keys()])
        query = f"UPDATE active_trade SET status = %s, updated_at = %s, {set_clause} WHERE id = %s"
        params = [status, now_utc] + list(filtered_kwargs.values()) + [trade_id]
    
    execute_db(query, params, commit=True, raise_err=True)

# ==========================================
# 2. Hardened State Reconstruction
# ==========================================
async def sync_state_from_exchange(coin="BTC/USDT:USDT"):
    if SIMULATION_MODE: return "Sim Mode Boot Bypass"
    exchange = ccxt.kucoinfutures({'apiKey': KU_KEY, 'secret': KU_SEC, 'password': KU_PASS})
    try:
        pos_data = await exchange.fetch_position(coin)
        pos_qty = float(pos_data.get('currentQty', 0))
        
        if abs(pos_qty) > 0:
            open_orders = await exchange.fetch_open_orders(coin)
            dir_str = "LONG" if pos_qty > 0 else "SHORT"
            actual_ep = float(pos_data.get('entryPrice', 0))
            
            trade_id, sl_id, tp_id, actual_sl, actual_tp = None, None, None, None, None
            
            for o in open_orders:
                client_oid = str(o.get('clientOrderId') or o.get('info', {}).get('clientOid') or '')
                if client_oid.startswith('C108-'):
                    parts = client_oid.split('-')
                    if len(parts) >= 3:
                        extracted_id = "-".join(parts[1:-1])
                        if not trade_id: trade_id = extracted_id
                        tag = parts[-1]
                        if tag in ['SL', 'SLR'] and o.get('reduceOnly'):
                            sl_id = o['id']; actual_sl = float(o.get('stopPrice', 0))
                        elif tag in ['TP', 'TPR'] and o.get('reduceOnly'):
                            tp_id = o['id']; actual_tp = float(o.get('stopPrice', 0))

            final_status = 'PROTECTED'
            if not trade_id:
                local_trade = execute_db("SELECT id FROM active_trade WHERE coin = %s", (coin,), fetchone=True)
                if local_trade: trade_id = str(local_trade['id'])
                else:
                    trade_id = str(uuid.uuid4())
                    final_status = 'UNKNOWN_EXTERNAL_POSITION'
            
            if final_status != 'UNKNOWN_EXTERNAL_POSITION' and (not sl_id or not tp_id):
                final_status = 'RECOVERED_UNPROTECTED'

            now_utc = datetime.now(timezone.utc)
            query = """INSERT INTO active_trade 
                       (id, coin, dir, actual_ep, actual_sl, actual_tp, actual_qty, status, sl_id, tp_id, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT(coin) DO UPDATE SET 
                       status=EXCLUDED.status, actual_qty=EXCLUDED.actual_qty, 
                       sl_id=EXCLUDED.sl_id, tp_id=EXCLUDED.tp_id, 
                       actual_sl=EXCLUDED.actual_sl, actual_tp=EXCLUDED.actual_tp,
                       updated_at=EXCLUDED.updated_at;"""
            execute_db(query, (trade_id, coin, dir_str, actual_ep, actual_sl, actual_tp, abs(pos_qty), final_status, sl_id, tp_id, now_utc, now_utc), commit=True)
            return f"Rebuilt ID: {trade_id} | Status: {final_status}"
    except Exception as e: print(f"[Sync Error]: {e}"); return str(e)
    finally: await exchange.close()

# ==========================================
# 3. Execution Engine & Reconciler
# ==========================================
async def async_execute_v108(trade_data):
    lock = DistributedCoinLock(trade_data['coin'])
    if not lock.acquire(): return False, "LOCKED: Coin is currently processing in another instance."
    
    exchange = ccxt.kucoinfutures({'apiKey': KU_KEY, 'secret': KU_SEC, 'password': KU_PASS, 'enableRateLimit': True})
    try:
        now_utc = datetime.now(timezone.utc)
        query = """INSERT INTO active_trade (id, coin, dir, planned_ep, struct_sl, struct_tp, planned_qty, status, created_at, updated_at) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s, 'RESERVED', %s, %s) ON CONFLICT(coin) DO NOTHING RETURNING id;"""
        if not execute_db(query, (trade_data['id'], trade_data['coin'], trade_data['dir'], trade_data['ep'], trade_data['sl'], trade_data['tp'], trade_data['qty'], now_utc, now_utc), fetchone=True, commit=True):
            return False, "Atomic Claim Failed."

        if SIMULATION_MODE:
            update_state(trade_data['id'], 'PROTECTED', actual_ep=trade_data['ep'], actual_qty=trade_data['qty'], actual_sl=trade_data['sl'], actual_tp=trade_data['tp'], main_id='SIM', sl_id='SIM', tp_id='SIM')
            return True, "Simulated Execution."

        side = 'buy' if trade_data['dir'] == "LONG" else 'sell'
        exit_side = 'sell' if trade_data['dir'] == "LONG" else 'buy'
        client_oid_base = f"C108-{trade_data['id']}"

        update_state(trade_data['id'], 'ENTRY_SUBMITTING')
        
        try:
            mkt_order = await exchange.create_market_order(trade_data['coin'], side, trade_data['qty'], params={'clientOid': f"{client_oid_base}-E"})
        except Exception as e:
            update_state(trade_data['id'], 'REPAIRING')
            raise RuntimeError(f"Entry Network Failure. Delegated to Reconciler: {e}")

        update_state(trade_data['id'], 'ENTRY_RECONCILING', main_id=mkt_order['id'])
        
        actual_qty, actual_ep = 0.0, 0.0
        for _ in range(5): 
            ord_info = await exchange.fetch_order(mkt_order['id'], trade_data['coin'])
            filled_amt = float(ord_info.get('filled', 0))
            if filled_amt > 0:
                actual_qty = filled_amt
                actual_ep = float(ord_info.get('average') or 0.0)
                if ord_info['status'] == 'closed': break 
                elif ord_info['status'] == 'open':
                    try: await exchange.cancel_order(mkt_order['id'], trade_data['coin'])
                    except: pass
                    await asyncio.sleep(2)
                    c_chk = await exchange.fetch_order(mkt_order['id'], trade_data['coin'])
                    actual_qty = float(c_chk.get('filled', actual_qty))
                    break
            await asyncio.sleep(1)
            
        if actual_qty == 0 or actual_ep == 0.0:
            update_state(trade_data['id'], 'UNKNOWN_NO_FILL')
            return False, "Zero Fill."

        update_state(trade_data['id'], 'ACTUAL_FILL_CONFIRMED', actual_ep=actual_ep, actual_qty=actual_qty)

        struct_sl, struct_tp = trade_data['sl'], trade_data['tp']
        if (trade_data['dir'] == "LONG" and struct_sl >= actual_ep) or (trade_data['dir'] == "SHORT" and struct_sl <= actual_ep):
            update_state(trade_data['id'], 'ABORTING_INVALID_SL')
            raise ValueError(f"Actual EP invalidates SL.")

        actual_rr = (abs(struct_tp - actual_ep) - (actual_ep * (ENTRY_FEE + EXIT_FEE + SLIPPAGE_ALLOWANCE))) / (abs(actual_ep - struct_sl) + (actual_ep * (ENTRY_FEE + EXIT_FEE + SLIPPAGE_ALLOWANCE)))
        
        if actual_rr < MIN_RR:
            update_state(trade_data['id'], 'ABORTING_POOR_RR')
            try:
                await exchange.create_market_order(trade_data['coin'], exit_side, actual_qty, params={'clientOid': f"{client_oid_base}-A"})
                update_state(trade_data['id'], 'REPAIRING')
            except Exception as e:
                update_state(trade_data['id'], 'CRITICAL_CLOSE_FAILED')
                raise RuntimeError(f"Abort Exception: {e}")
            raise ValueError(f"RR < {MIN_RR}. Abort initiated.")

        update_state(trade_data['id'], 'SL_SUBMITTING', actual_sl=struct_sl, actual_tp=struct_tp)
        
        try:
            sl_ord = await exchange.create_order(trade_data['coin'], 'market', exit_side, actual_qty, None, {'stop': 'down' if exit_side=='sell' else 'up', 'stopPrice': exchange.price_to_precision(trade_data['coin'], struct_sl), 'reduceOnly': True, 'clientOid': f"{client_oid_base}-SL"})
            update_state(trade_data['id'], 'SL_SUBMITTED_TP_PENDING', sl_id=sl_ord['id'])
            
            tp_ord = await exchange.create_order(trade_data['coin'], 'market', exit_side, actual_qty, None, {'stop': 'up' if exit_side=='sell' else 'down', 'stopPrice': exchange.price_to_precision(trade_data['coin'], struct_tp), 'reduceOnly': True, 'clientOid': f"{client_oid_base}-TP"})
            update_state(trade_data['id'], 'STOPS_SUBMITTED', tp_id=tp_ord['id'])
        except Exception as e:
            update_state(trade_data['id'], 'REPAIRING')
            raise RuntimeError(f"Stop placement failed atomically. Reconciler taking over: {e}")

        update_state(trade_data['id'], 'PROTECTED')
        return True, f"Execution Complete! EP: {actual_ep:.4f} | RR: {actual_rr:.2f}"
    except Exception as e:
        print(f"[Execution Error]: {e}")
        return False, f"Exception Handled: {e}"
    finally:
        await exchange.close()
        lock.release()

async def auto_reconciler_worker(coin):
    lock = DistributedCoinLock(coin)
    if not lock.acquire(): return "LOCKED."
    active_db = execute_db("SELECT * FROM active_trade WHERE coin = %s LIMIT 1", (coin,), fetchone=True)
    if not active_db: lock.release(); return "No active trade."
    
    if active_db['status'] == 'UNKNOWN_EXTERNAL_POSITION':
        lock.release(); return "Requires Manual Audit."
        
    if SIMULATION_MODE: lock.release(); return "Sim Mode."

    exchange = ccxt.kucoinfutures({'apiKey': KU_KEY, 'secret': KU_SEC, 'password': KU_PASS})
    try:
        pos_data = await exchange.fetch_position(coin)
        actual_pos_qty = abs(float(pos_data.get('currentQty', 0)))
        actual_qty_db = float(active_db['actual_qty'] or 0.0)

        if actual_pos_qty == 0:
            exit_reason, exit_ep = "UNKNOWN_EXIT", 0.0
            for oid, reason in [(active_db['sl_id'], "STOP_LOSS_HIT"), (active_db['tp_id'], "TAKE_PROFIT_HIT")]:
                if oid:
                    try:
                        ord_info = await exchange.fetch_order(oid, coin)
                        if ord_info['status'] == 'closed' or str(ord_info.get('info', {}).get('status')).lower() in ['triggered', 'done']:
                            exit_reason = reason
                            exit_ep = float(ord_info.get('average') or ord_info.get('price') or active_db['actual_ep'])
                            break
                    except: pass
            
            if exit_reason == "UNKNOWN_EXIT" and not active_db['actual_qty']:
                exit_reason = "ENTRY_FAILED_OR_CANCELLED"
                exit_ep = float(active_db['planned_ep'])
            
            if exit_reason == "UNKNOWN_EXIT" and actual_qty_db > 0:
                update_state(active_db['id'], 'UNKNOWN_EXIT_REQUIRES_AUDIT')
                return "Reconciliation Halted: Manual Audit Required."

            actual_ep = float(active_db['actual_ep'] or 0.0)
            gross_pnl = ((exit_ep - actual_ep) * actual_qty_db) * (1 if active_db['dir'] == "LONG" else -1) if actual_ep > 0 else 0
            net_pnl = gross_pnl - (actual_ep * actual_qty_db * ENTRY_FEE) - (exit_ep * actual_qty_db * EXIT_FEE)

            for o in await exchange.fetch_open_orders(coin):
                if str(o.get('clientOrderId') or '').startswith(f"C108-{active_db['id']}"):
                    try: await exchange.cancel_order(o['id'], coin)
                    except: pass
            
            now_utc = datetime.now(timezone.utc)
            conn = db_pool.getconn()
            try:
                with conn.cursor() as c:
                    c.execute("DELETE FROM active_trade WHERE id = %s", (active_db['id'],))
                    c.execute("INSERT INTO journal (id, coin, dir, actual_ep, exit_ep, actual_qty, gross_pnl, net_pnl, reason, closed_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", 
                              (active_db['id'], coin, active_db['dir'], actual_ep, exit_ep, actual_qty_db, gross_pnl, net_pnl, exit_reason, now_utc))
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise RuntimeError(f"Journal Atomic Commit Failed: {e}")
            finally: db_pool.putconn(conn)
            
            return f"Reconciled 0 Pos. Archived."

        if actual_pos_qty > 0:
            db_is_long = active_db['dir'] == 'LONG'
            if (float(pos_data.get('currentQty', 0)) > 0) != db_is_long:
                update_state(active_db['id'], 'CRITICAL_DIRECTION_MISMATCH')
                return "CRITICAL: Direction Mismatch."

            exit_side = 'sell' if db_is_long else 'buy'
            repairs = []
            sl_exists, tp_exists = False, False
            client_oid_base = f"C108-{active_db['id']}"
            
            for o in await exchange.fetch_open_orders(coin):
                if o['side'] == exit_side and o.get('reduceOnly'):
                    if is_match(o.get('stopPrice', 0), active_db['actual_sl']): sl_exists = True
                    elif is_match(o.get('stopPrice', 0), active_db['actual_tp']): tp_exists = True
                    elif str(o.get('clientOrderId') or '').startswith('C108-'):
                        try: await exchange.cancel_order(o['id'], coin)
                        except: pass

            if not sl_exists and active_db['actual_sl']:
                sl_ord = await exchange.create_order(coin, 'market', exit_side, actual_pos_qty, None, {'stop': 'down' if exit_side=='sell' else 'up', 'stopPrice': exchange.price_to_precision(coin, active_db['actual_sl']), 'reduceOnly': True, 'clientOid': f"{client_oid_base}-SLR"})
                update_state(active_db['id'], 'REPAIRING', sl_id=sl_ord['id'])
                repairs.append("SL")

            if not tp_exists and active_db['actual_tp']:
                tp_ord = await exchange.create_order(coin, 'market', exit_side, actual_pos_qty, None, {'stop': 'up' if exit_side=='sell' else 'down', 'stopPrice': exchange.price_to_precision(coin, active_db['actual_tp']), 'reduceOnly': True, 'clientOid': f"{client_oid_base}-TPR"})
                update_state(active_db['id'], 'REPAIRING', tp_id=tp_ord['id'])
                repairs.append("TP")
                
            if repairs or active_db['status'] in ['RECOVERED_UNPROTECTED', 'REPAIRING']:
                update_state(active_db['id'], 'REPAIRING')
                await asyncio.sleep(2) 
                
                v_pos = await exchange.fetch_position(coin)
                if not is_match(abs(float(v_pos.get('currentQty', 0))), actual_pos_qty):
                    update_state(active_db['id'], 'REPAIR_FAILED_QTY_CHANGED')
                    return "Repair failed: QTY changed."
                    
                v_ops = await exchange.fetch_open_orders(coin)
                v_sl, v_tp = False, False
                for o in v_ops:
                    if o['side'] == exit_side and o.get('reduceOnly'):
                        if is_match(o.get('stopPrice', 0), active_db['actual_sl']): v_sl = True
                        if is_match(o.get('stopPrice', 0), active_db['actual_tp']): v_tp = True
                
                if not (v_sl and v_tp):
                    update_state(active_db['id'], 'REPAIR_VERIFICATION_FAILED')
                    return "Verification Failed."

            update_state(active_db['id'], 'PROTECTED', actual_qty=actual_pos_qty)
            return f"Repaired: {repairs}. PROTECTED."
            
    except Exception as e: return f"Reconciler Fault: {e}"
    finally:
        await exchange.close()
        lock.release()

# ==========================================
# 4. Streamlit Daemon, Boot & Retention Logic
# ==========================================
async def background_reconciler_service():
    loop_count = 0
    while True:
        try:
            # Reconcile unverified trades
            unverified_trades = execute_db("SELECT coin FROM active_trade WHERE status NOT IN ('PROTECTED', 'UNKNOWN_EXTERNAL_POSITION') AND updated_at < NOW() - INTERVAL '30 seconds'", fetchall=True)
            if unverified_trades:
                for row in unverified_trades: await auto_reconciler_worker(row['coin'])
            
            # Retention Logic: 500MB Free Tier Constraint (Cleanup > 30 days every ~1 hour)
            loop_count += 1
            if loop_count >= 240:
                execute_db("DELETE FROM journal WHERE closed_at < NOW() - INTERVAL '30 days'", commit=True)
                loop_count = 0
                
        except Exception as e: print(f"Loop Err: {e}")
        await asyncio.sleep(15)

@st.cache_resource
def boot_system():
    # Sync initial exchange state
    asyncio.run(sync_state_from_exchange("BTC/USDT:USDT"))
    
    # Run a boot-time retention cleanup
    try: execute_db("DELETE FROM journal WHERE closed_at < NOW() - INTERVAL '30 days'", commit=True)
    except: pass
    
    # Start Reconciler Daemon
    def run_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(background_reconciler_service())
    threading.Thread(target=run_loop, daemon=True, name="C108-Reconciler").start()
    return True

boot_system()

# ==========================================
# 5. UI Layout
# ==========================================
st.title("Cosmic 108 | Distributed Enterprise Engine")
coin = "BTC/USDT:USDT" 
active_trade = execute_db("SELECT * FROM active_trade WHERE coin = %s LIMIT 1", (coin,), fetchone=True)
col_exec, col_recon = st.columns(2)

with col_exec:
    st.subheader("RISK ENGINE MOCK")
    if not active_trade:
        if st.button("🟢 EXECUTE LONG", use_container_width=True):
            payload = {"id": str(uuid.uuid4()), "coin": coin, "dir": "LONG", "ep": 65000.0, "qty": 0.001, "sl": 64800.0, "tp": 65300.0}
            with st.spinner("Executing Cosmic 108 FSM..."):
                success, msg = asyncio.run(async_execute_v108(payload))
                if success: st.success(msg)
                else: st.error(msg)
    else:
        if active_trade['status'] == 'UNKNOWN_EXTERNAL_POSITION':
            st.error("⚠️ UNKNOWN EXTERNAL POSITION DETECTED!")
            st.warning("To protect this trade, assign structural SL/TP and Claim it.")
            
            c1, c2 = st.columns(2)
            claim_sl = c1.number_input("Structural SL", value=float(active_trade['actual_ep'] * 0.99))
            claim_tp = c2.number_input("Structural TP", value=float(active_trade['actual_ep'] * 1.05))
            
            if st.button("🛠️ CLAIM & PROTECT POSITION", use_container_width=True):
                update_state(active_trade['id'], 'REPAIRING', actual_sl=claim_sl, actual_tp=claim_tp)
                st.success("Position Claimed! Cosmic 108 Reconciler is now assigning Stops.")
                st.rerun()
        else:
            st.info(f"STATUS: {active_trade['status']}")
            st.write(f"ID: {active_trade['id']}")

with col_recon:
    st.subheader("SYSTEM RECONCILER")
    if st.button("🔄 FORCE RECONCILE", use_container_width=True):
        with st.spinner("Auditing..."):
            msg = asyncio.run(auto_reconciler_worker(coin))
            st.success(msg)
            st.rerun()

    df_j = pd.DataFrame(execute_db("SELECT coin, dir, actual_ep, exit_ep, net_pnl FROM journal ORDER BY closed_at DESC LIMIT 3", fetchall=True))
    if not df_j.empty: st.dataframe(df_j)
