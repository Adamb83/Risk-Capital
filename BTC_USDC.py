import time
import os
import ccxt
import pandas as pd
from datetime import datetime
import config_low as config  # Contains API_KEY and API_SECRET

# --- User-Defined Settings & Strategy Parameters ---
SYMBOL                  = 'BTC/USDC'   ###USDC#####
TIMEFRAME               = '1h'
LIMIT                   = 500

SHORT_MA_LENGTH         = 18
MID_MA_LENGTH           = 21
LONG_MA_LENGTH          = 40
SHORT_MA_TYPE           = "ema"
MID_MA_TYPE             = "ema"
LONG_MA_TYPE            = "ema" 

POSITION_SIZE_PERCENT   = 7.5
TAKE_PROFIT_PERCENT     = 2000 
REENTRY_GAP_PERCENT     = 12
SLIPPAGE                = 0.0005
FEE_RATE                = 0.001
CLOSE_PROFIT_BUFFER_PERCENT = 5   # Must be in profit this much to close on crossdown.
CLOSE_ALL_ON_CROSSDOWN  = False
CLOSE_PROFIT_ON_CROSSDOWN   = True
MAX_OPEN_TRADES         = 15
ACCUMULATION_STEPS      = 2
PRICE_THRESHOLD         = 0.5

# Format the symbol for use in filenames (e.g., BTC_USDC)
formatted_symbol = SYMBOL.replace("/", "_")

# File paths for persistence using the formatted symbol
OPEN_POSITIONS_FILE     = f"open_positions_{formatted_symbol}.csv"
CLOSED_TRADES_FILE      = f"closed_trades_{formatted_symbol}.csv"

# Global state variables
open_positions = []
closed_trades = []
partial_plan   = {"active": False, "steps_left": 0, "remaining_value": 0.0}
historical_df  = pd.DataFrame()

# --- Initialize CCXT Exchange with MEXC (v3) ---
exchange = ccxt.mexc({
    'apiKey': config.API_KEY,
    'secret': config.API_SECRET,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
        'version': 'v3',
    },
})

# --- Persistence Utility Functions ---
def load_list_from_csv(filepath):
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        try:
            return pd.read_csv(filepath).to_dict('records')
        except Exception as e:
            print(f"[ERROR] Failed to load {filepath}: {e}")
    return []

def save_list_to_csv(data_list, filepath):
    try:
        pd.DataFrame(data_list).to_csv(filepath, index=False)
    except Exception as e:
        print(f"[ERROR] Failed to save {filepath}: {e}")

# --- Utility Functions ---
def calculate_ma(prices: pd.Series, window: int, ma_type: str) -> pd.Series:
    if ma_type.lower() == "sma":
        return prices.rolling(window=window).mean()
    elif ma_type.lower() == "ema":
        return prices.ewm(span=window, adjust=False).mean()
    else:
        raise ValueError("ma_type must be 'sma' or 'ema'.")

def fetch_ohlcv(symbol, timeframe, limit):
    print(f"[DEBUG] Fetching OHLCV for {symbol}, timeframe={timeframe}, limit={limit}")
    try:
        kline_data = exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(kline_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        print(f"[ERROR] Failed to fetch/process OHLCV data: {e}")
        return pd.DataFrame()

def initialize_data():
    global historical_df
    historical_df = fetch_ohlcv(SYMBOL, TIMEFRAME, LIMIT)
    if historical_df.empty or len(historical_df) < 2:
        print("[ERROR] Not enough historical data fetched during initialization.")
    else:
        historical_df["short_ma"] = calculate_ma(historical_df["close"], SHORT_MA_LENGTH, SHORT_MA_TYPE)
        historical_df["mid_ma"] = calculate_ma(historical_df["close"], MID_MA_LENGTH, MID_MA_TYPE)
        historical_df["long_ma"] = calculate_ma(historical_df["close"], LONG_MA_LENGTH, LONG_MA_TYPE)
        print(f"[INFO] Initialized historical data with {len(historical_df)} rows.")

def update_data():
    global historical_df
    new_data = fetch_ohlcv(SYMBOL, TIMEFRAME, 2)
    if historical_df.empty:
        initialize_data()
        return

    last_known_time = historical_df["timestamp"].iloc[-1]
    new_candles = new_data[new_data["timestamp"] > last_known_time]

    if not new_candles.empty:
        historical_df = pd.concat([historical_df, new_candles], ignore_index=True)
        if len(historical_df) > LIMIT:
            historical_df = historical_df.iloc[-LIMIT:].reset_index(drop=True)
        
        historical_df["short_ma"] = calculate_ma(historical_df["close"], SHORT_MA_LENGTH, SHORT_MA_TYPE)
        historical_df["mid_ma"] = calculate_ma(historical_df["close"], MID_MA_LENGTH, MID_MA_TYPE)
        historical_df["long_ma"] = calculate_ma(historical_df["close"], LONG_MA_LENGTH, LONG_MA_TYPE)
        print(f"[INFO] Updated historical data with {len(new_candles)} new candle(s).")
    else:
        print("[DEBUG] No new candles to append during update_data().")

def check_connectivity():
    try:
        exchange.load_markets()
        print("[INFO] Connectivity check passed.")
        return True
    except Exception as e:
        print("[ERROR] Connectivity check failed:", e)
        return False

def get_account_info():
    balances = exchange.fetch_balance()
    free_usdc = balances['free'].get('USDC', 0.0)
    return {
        "balances": {
            "USDC": {
                "free": free_usdc
            }
        }
    }

def create_market_order(symbol, side, quantity):
    try:
        print(f"[DEBUG] Creating market order: side={side}, quantity={quantity:.6f}")
        order = exchange.create_order(
            symbol=symbol,
            type='market',
            side=side,
            amount=quantity
        )
        return order
    except Exception as e:
        print(f"[ERROR] Failed to place {side} order: {e}")
        return None

def do_buy_trade(trade_value, current_price, timestamp):
    global open_positions
    account_info = get_account_info()
    available_balance = float(account_info['balances']['USDC']['free'])
    print(f"[DEBUG] Attempting BUY with trade_value={trade_value:.2f}, available_balance={available_balance:.2f}")

    if trade_value > available_balance:
        trade_value = available_balance
    if trade_value <= 0:
        print("[DEBUG] Trade value <= 0, skipping buy.")
        return

    actual_buy_price = current_price * (1 + SLIPPAGE)
    cost_before_fee = trade_value
    buy_fee = cost_before_fee * FEE_RATE
    total_cost = cost_before_fee + buy_fee

    if total_cost > available_balance:
        cost_before_fee = available_balance / (1 + FEE_RATE)
        buy_fee = cost_before_fee * FEE_RATE
        total_cost = cost_before_fee + buy_fee

    size = cost_before_fee / actual_buy_price
    print(f"[DEBUG] Calculated size={size:.6f}, actual_buy_price={actual_buy_price:.4f}, buy_fee={buy_fee:.4f}")

    order_response = create_market_order(SYMBOL, 'buy', size)
    if not order_response:
        print("[WARN] Buy order failed or was not executed.")
        return

    new_trade = {
        "timestamp_open": timestamp,
        "buy_price": current_price,
        "effective_buy_price": actual_buy_price,
        "size": size,
        "buy_fee": buy_fee
    }
    open_positions.append(new_trade)
    save_list_to_csv(open_positions, OPEN_POSITIONS_FILE)
    print(f"[INFO] Executed buy: {size:.6f} {SYMBOL} at {current_price} on {timestamp}")

def do_partial_sell(position, sell_ratio, current_price, timestamp):
    global closed_trades, open_positions
    actual_sell_price = current_price * (1 - SLIPPAGE)
    sell_size = position["size"] * sell_ratio
    proceeds_before_fee = sell_size * actual_sell_price
    sell_fee = proceeds_before_fee * FEE_RATE

    print(f"[DEBUG] Attempting SELL of ratio={sell_ratio}, sell_size={sell_size:.6f} at current_price={current_price:.4f}")

    order_response = create_market_order(SYMBOL, 'sell', sell_size)
    if not order_response:
        print("[WARN] Sell order failed or was not executed.")
        return

    net_proceeds = proceeds_before_fee - sell_fee
    buy_cost_for_this_sell = sell_size * position["effective_buy_price"]
    gross_pnl = proceeds_before_fee - buy_cost_for_this_sell
    net_pnl = net_proceeds - buy_cost_for_this_sell

    holding_sec = (timestamp - pd.to_datetime(position["timestamp_open"])).total_seconds()
    holding_days = holding_sec / (3600 * 24)

    closed_trade = {
        "timestamp_open": position["timestamp_open"],
        "timestamp_close": timestamp,
        "buy_price": position["buy_price"],
        "effective_buy_price": position["effective_buy_price"],
        "sell_price": current_price,
        "effective_sell_price": actual_sell_price,
        "size": sell_size,
        "buy_fee": position["buy_fee"],
        "sell_fee": sell_fee,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "holding_days": holding_days
    }
    closed_trades.append(closed_trade)

    position["size"] -= sell_size
    if position["size"] < 1e-8:
        open_positions.remove(position)
    save_list_to_csv(open_positions, OPEN_POSITIONS_FILE)
    save_list_to_csv(closed_trades, CLOSED_TRADES_FILE)
    print(f"[INFO] Executed sell: {sell_size:.6f} {SYMBOL} at {current_price} on {timestamp}")

def run_strategy():
    global partial_plan, open_positions, closed_trades, historical_df

    update_data()
    df = historical_df.copy()

    # Check if we have at least 2 rows after update
    if len(df) < 2:
        print("[WARN] Not enough data to proceed with strategy loop.")
        return

    current_data = df.iloc[-1]
    current_price = current_data["close"]
    timestamp = current_data["timestamp"]
    short_ma_curr = current_data["short_ma"]
    mid_ma_curr = current_data["mid_ma"]
    long_ma_curr = current_data["long_ma"]

    prev_close = df["close"].iloc[-2]
    price_diff = ((current_price - prev_close) / prev_close) * 100

    print(f"[DEBUG] run_strategy -> TS={timestamp}, close={current_price:.4f}, shortMA={short_ma_curr}, midMA={mid_ma_curr}, longMA={long_ma_curr}, price_diff={price_diff:.2f}%")

    # --- Step 1: Partial accumulation steps ---
    if partial_plan["active"] and partial_plan["steps_left"] > 0:
        step_value = partial_plan["remaining_value"] / partial_plan["steps_left"]
        print(f"[DEBUG] Partial plan active. steps_left={partial_plan['steps_left']}, step_value={step_value:.2f}")
        do_buy_trade(step_value, current_price, timestamp)
        partial_plan["remaining_value"] -= step_value
        partial_plan["steps_left"] -= 1
        if partial_plan["steps_left"] == 0:
            partial_plan["active"] = False
            print("[DEBUG] Partial plan completed.")

    # --- Step 2: Take-profit check ---
    target_mult = 1 + (TAKE_PROFIT_PERCENT / 100.0)
    for pos in open_positions.copy():
        if current_price >= pos["buy_price"] * target_mult:
            print(f"[DEBUG] Take-profit condition met for position with buy_price={pos['buy_price']:.4f}")
            do_partial_sell(pos, 1.0, current_price, timestamp)

    # --- Step 3: Crossdown check ---
    if pd.notna(short_ma_curr) and pd.notna(mid_ma_curr):
        short_ma_prev = df["short_ma"].iloc[-2]
        mid_ma_prev = df["mid_ma"].iloc[-2]
        cross_down = (short_ma_curr < mid_ma_curr) and (short_ma_prev >= mid_ma_prev)
        if cross_down:
            print("[INFO] MA Crossdown detected. Executing crossdown logic.")
            if partial_plan["active"]:
                partial_plan["active"] = False
                partial_plan["steps_left"] = 0
                partial_plan["remaining_value"] = 0.0
                print("[INFO] Partial accumulation plan canceled due to crossdown.")

            if CLOSE_ALL_ON_CROSSDOWN:
                for pos in open_positions.copy():
                    do_partial_sell(pos, 1.0, current_price, timestamp)
            else:
                if CLOSE_PROFIT_ON_CROSSDOWN:
                    for pos in open_positions.copy():
                        profit_percent = ((current_price - pos["buy_price"]) / pos["buy_price"]) * 100
                        if profit_percent > CLOSE_PROFIT_BUFFER_PERCENT:
                            do_partial_sell(pos, 1.0, current_price, timestamp)

    # --- Step 4: Entry condition & reentry gap check ---
    if all([
        pd.notna(short_ma_curr), pd.notna(mid_ma_curr), pd.notna(long_ma_curr),
        short_ma_curr > long_ma_curr,
        mid_ma_curr > long_ma_curr,
        (price_diff > PRICE_THRESHOLD),
        (len(open_positions) < MAX_OPEN_TRADES)
    ]):
        print("[DEBUG] Entry condition prelim check passed. Checking reentry gap...")

        can_buy = True

        # Apply reentry gap logic only if there are existing open positions
        if open_positions:
            # Retrieve the previous order's buy price (most recent order)
            last_buy_price = open_positions[-1]["buy_price"]
            needed_price = last_buy_price * (1 - REENTRY_GAP_PERCENT / 100.0)
            if current_price > needed_price:
                print(f"[DEBUG] Reentry gap not met. needed_price={needed_price:.4f}, current_price={current_price:.4f}")
                can_buy = False

        # If no open positions, skip reentry gap (can_buy remains True)

        if can_buy:
            if partial_plan["active"]:
                print("[DEBUG] Partial plan already active. Skipping new plan.")
            else:
                print("[DEBUG] All conditions passed. Initiating partial accumulation plan.")
                account_info = get_account_info()
                available_balance = float(account_info['balances']['USDC']['free'])
                total_trade_value = available_balance * (POSITION_SIZE_PERCENT / 100.0)

                partial_plan["active"] = True
                partial_plan["steps_left"] = ACCUMULATION_STEPS
                partial_plan["remaining_value"] = total_trade_value
                print(f"[INFO] Initiating partial accumulation plan: {ACCUMULATION_STEPS} steps with total value {total_trade_value:.2f}")
        else:
            print("[DEBUG] Did not initiate a new buy due to reentry gap condition.")
    else:
        print("[DEBUG] Entry condition not met this loop.")

def main():
    if not check_connectivity():
        print("[ERROR] Unable to connect to the exchange. Exiting.")
        return

    print("[INFO] Starting trading bot with CCXT using MEXC (v3 endpoints)...")

    # Attempt to load open/closed trades
    open_positions_list = load_list_from_csv(OPEN_POSITIONS_FILE)
    if open_positions_list:
        print(f"[DEBUG] Loaded open_positions from CSV: {len(open_positions_list)} records.")
    else:
        print("[DEBUG] No open positions loaded (CSV empty or doesn't exist).")

    closed_trades_list = load_list_from_csv(CLOSED_TRADES_FILE)
    if closed_trades_list:
        print(f"[DEBUG] Loaded closed_trades from CSV: {len(closed_trades_list)} records.")
    else:
        print("[DEBUG] No closed trades loaded (CSV empty or doesn't exist).")

    global open_positions, closed_trades
    open_positions = open_positions_list
    closed_trades = closed_trades_list
    print(f"[INFO] Loaded {len(open_positions)} open positions and {len(closed_trades)} closed trades from disk.")

    initialize_data()

    while True:
        try:
            run_strategy()
        except Exception as e:
            print(f"[ERROR] Error in strategy loop: {e}")
        time.sleep(3630)  # Sleep for 1 minute

if __name__ == "__main__":
    main()
