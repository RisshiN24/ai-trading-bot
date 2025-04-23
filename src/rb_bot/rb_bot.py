import os
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from alpaca_trade_api import REST
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.traders import Trader
from lumibot.strategies.strategy import Strategy
from timedelta import Timedelta

# === Load environment variables ===
load_dotenv()

# === Alpaca API setup ===
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2"

ALPACA_CREDS = {
    "API_KEY": ALPACA_API_KEY,
    "API_SECRET": ALPACA_API_SECRET,
    "PAPER": True
}

# === Technical Indicators ===
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def add_ema_ribbon_features(df):
    spans = [8, 13, 21, 34, 55, 89]

    for span in spans:
        df[f'EMA_{span}'] = df['close'].ewm(span=span).mean()

    df['EMA8_to_21'] = df['EMA_8'] / df['EMA_21']
    df['EMA13_to_34'] = df['EMA_13'] / df['EMA_34']
    df['EMA21_to_55'] = df['EMA_21'] / df['EMA_55']
    df['EMA34_to_89'] = df['EMA_34'] / df['EMA_89']

    ribbon_emas = [df[f'EMA_{s}'] for s in spans]
    df['ribbon_width'] = pd.concat(ribbon_emas, axis=1).max(axis=1) - pd.concat(ribbon_emas, axis=1).min(axis=1)
    df['EMA8_slope'] = df['EMA_8'].diff()
    df['EMA21_slope'] = df['EMA_21'].diff()

    return df

# === Mean Reversion Strategy using Bollinger Bands ===
class RBTrader(Strategy):
    def initialize(self, symbol="SPY", cash_at_risk=1):
        self.symbol = symbol
        self.cash_at_risk = cash_at_risk
        self.sleeptime = "24H"
        self.last_trade = None
        self.api = REST(base_url=ALPACA_BASE_URL, key_id=ALPACA_API_KEY, secret_key=ALPACA_API_SECRET)

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price, 0)
        return cash, last_price, quantity

    def get_dates(self):
        today = self.get_datetime()
        past = today - Timedelta(days=50)
        return today.strftime('%Y-%m-%d'), past.strftime('%Y-%m-%d')

    def get_info(self):
        # Pull recent price data and compute indicators
        today, past = self.get_dates()
        bars = self.api.get_bars(self.symbol, '1D', limit=1000, start=past, end=today)
        df = pd.DataFrame([{ 'time': bar.t, 'open': bar.o, 'high': bar.h, 'low': bar.l, 'close': bar.c, 'volume': bar.v } for bar in bars])

        df['price_change'] = df['close'].pct_change()
        df['MACD_line'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['RSI'] = calculate_rsi(df)
        df = add_ema_ribbon_features(df)
        df['volatility'] = df['price_change'].rolling(5).std()
        df['volume_change'] = df['volume'].pct_change()
        df['body_size'] = abs(df['close'] - df['open'])
        df['range'] = df['high'] - df['low']
        df['body_to_range'] = df['body_size'] / df['range']
        df['BB_upper'] = df['close'].rolling(20).mean() + 1.5 * df['close'].rolling(20).std()
        df['BB_lower'] = df['close'].rolling(20).mean() - 1.5 * df['close'].rolling(20).std()
        df['bb_percent'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['ret_1d'] = df['close'].pct_change(1)
        df['ret_3d'] = df['close'].pct_change(3)
        df['ret_5d'] = df['close'].pct_change(5)
        df['stoch_k'] = 100 * ((df['close'] - df['low'].rolling(14).min()) /
                               (df['high'].rolling(14).max() - df['low'].rolling(14).min()))
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        df = df.dropna()
        today_data = df.iloc[-1]

        # Return latest values for trade decision
        return [today_data['RSI'], today_data['BB_lower'], today_data['BB_upper'], today_data['open']]

    def on_trading_iteration(self):
        # Fetch prediction-relevant data
        cash, last_price, max_qty = self.position_sizing()
        rsi, bb_lower, bb_upper, td_open = self.get_info()

        print(f"[{self.get_datetime()}] Cash: {cash:.2f}, Last Price: {last_price:.2f}, "
              f"Max Qty: {max_qty}, RSI: {rsi}, BB Lower: {bb_lower}, BB Upper: {bb_upper}")

        # Check current position
        position = self.get_position(self.symbol)
        current_qty = position.quantity if position else 0

        # Clamp position size to available cash
        max_qty = min(max_qty, int(cash / last_price)) if last_price != 0 else 0

        # === Entry/Exit Conditions ===
        if td_open > bb_upper:
            # Price is above upper band → go long
            if current_qty > 0:
                print("Already in a long position. Doing nothing.")
            elif max_qty > 0:
                print(f"Submitting BUY order for {max_qty} shares (long).")
                order = self.create_order(
                    self.symbol,
                    max_qty,
                    "buy",
                    type="bracket"
                )
                self.submit_order(order)

        elif td_open < bb_lower:
            # Price is below lower band → exit long if held
            if current_qty > 0:
                print("Currently long. Flattening before going short.")
                self.sell_all()

# === Backtest Setup ===
start_date = datetime(2023, 1, 1)
end_date = datetime(2025, 4, 10)
broker = Alpaca(ALPACA_CREDS)

strategy = RBTrader(
    name='RBTrader',
    broker=broker,
    parameters={"symbol": "SPY", "cash_at_risk": 1}
)

strategy.backtest(YahooDataBacktesting, start_date, end_date, parameters={"symbol": "SPY", "cash_at_risk": 1})
