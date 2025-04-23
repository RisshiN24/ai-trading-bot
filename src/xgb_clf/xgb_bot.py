# === Imports ===
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
from joblib import load
from xgb_model import calculate_rsi, add_ema_ribbon_features

# === Load API keys from .env ===
load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2"

# === Credentials for Lumibot ===
ALPACA_CREDS = {
    "API_KEY": ALPACA_API_KEY,
    "API_SECRET": ALPACA_API_SECRET,
    "PAPER": True
}

# === XGBoost Trading Strategy ===
class XGBTrader(Strategy):
    def initialize(self, symbol="SPY", cash_at_risk=1, model_path="xgb_model.pkl", scaler_path="xgb_scaler.pkl"):
        self.symbol = symbol
        self.cash_at_risk = cash_at_risk
        self.model = load(model_path)
        self.scaler = load(scaler_path)
        self.sleeptime = "72H"
        self.last_trade = None
        self.api = REST(base_url=ALPACA_BASE_URL, key_id=ALPACA_API_KEY, secret_key=ALPACA_API_SECRET)

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price, 0)
        return cash, last_price, quantity

    def get_dates(self):
        today = self.get_datetime()
        past = today - Timedelta(days=60)
        return today.strftime('%Y-%m-%d'), past.strftime('%Y-%m-%d')

    def get_prediction(self):
        # Pull recent price history
        today, past = self.get_dates()
        bars = self.api.get_bars(self.symbol, '1D', limit=1000, start=past, end=today)
        df = pd.DataFrame([{ 
            'time': bar.t, 'open': bar.o, 'high': bar.h,
            'low': bar.l, 'close': bar.c, 'volume': bar.v 
        } for bar in bars])

        # === Feature Engineering ===
        df['price_change'] = df['close'].pct_change()
        df['MACD_line'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['RSI'] = calculate_rsi(df)
        df = add_ema_ribbon_features(df)
        df['volatility'] = df['price_change'].rolling(5).std()
        df['volume_change'] = df['volume'].pct_change()
        df['body_size'] = abs(df['close'] - df['open'])
        df['range'] = df['high'] - df['low']
        df['body_to_range'] = df['body_size'] / df['range']
        df['BB_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
        df['BB_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
        df['bb_percent'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['ret_1d'] = df['close'].pct_change(1)
        df['ret_3d'] = df['close'].pct_change(3)
        df['ret_5d'] = df['close'].pct_change(5)
        df['stoch_k'] = 100 * ((df['close'] - df['low'].rolling(14).min()) / (df['high'].rolling(14).max() - df['low'].rolling(14).min()))
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # Future return labeling for evaluation (not used in prediction)
        df['future_price'] = df['close'].shift(-3)
        df['future_return'] = (df['future_price'] - df['close']) / df['close']
        df['target'] = (df['future_return'] > 0.01).astype(int)

        df = df.dropna()
        print(df.shape)

        # === Feature selection and input prep ===
        features = [
            'MACD_line', 'RSI',
            'EMA_8', 'EMA_13', 'EMA_21', 'EMA_34', 'EMA_55', 'EMA_89',
            'EMA8_to_21', 'EMA13_to_34', 'EMA21_to_55', 'EMA34_to_89',
            'ribbon_width', 'EMA8_slope', 'EMA21_slope',
            'volatility', 'volume_change', 'body_size', 'range', 'body_to_range',
            'BB_upper', 'BB_lower', 'bb_percent', 'momentum_5',
            'ret_1d', 'ret_3d', 'ret_5d', 'stoch_k', 'stoch_d'
        ]

        raw_X = df[features].values
        latest_window = raw_X[-5:]  # Take last 5 rows
        latest_flat = latest_window.flatten().reshape(1, -1)  # Flatten to 1D
        latest_input = self.scaler.transform(latest_flat)  # Scale input
        pred = self.model.predict(latest_input)[0]
        return int(pred)

    def on_trading_iteration(self):
        # === Run prediction and place trades ===
        cash, last_price, max_qty = self.position_sizing()
        prediction = self.get_prediction()

        print(f"[{self.get_datetime()}] Cash: {cash:.2f}, Last Price: {last_price:.2f}, "
              f"Max Qty: {max_qty}, Prediction: {prediction}")

        if prediction is None:
            print("No prediction returned.")
            return

        position = self.get_position(self.symbol)
        current_qty = position.quantity if position else 0
        max_qty = min(max_qty, int(cash / last_price)) if last_price != 0 else 0

        if prediction == 1:
            if current_qty > 0:
                print("Already in a long position. Doing nothing.")
            elif current_qty == 0 and max_qty > 0:
                print(f"Submitting BUY order for {max_qty} shares.")
                order = self.create_order(
                    self.symbol,
                    max_qty,
                    "buy",
                    type="bracket",
                    take_profit_price=last_price * 1.03,
                    stop_loss_price=last_price * 0.97
                )
                self.submit_order(order)
        else:
            if current_qty > 0:
                print("Currently long. Flattening position.")
                self.sell_all()

# === Backtest Setup ===
start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 4, 10)
broker = Alpaca(ALPACA_CREDS)

strategy = XGBTrader(
    name='XGBTrader',
    broker=broker,
    parameters={
        "symbol": "SPY",
        "cash_at_risk": 1,
        "model_path": "xgb_model.pkl",
        "scaler_path": "xgb_scaler.pkl"
    }
)

# === Run backtest ===
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={
        "symbol": "SPY",
        "cash_at_risk": 1,
        "model_path": "xgb_model.pkl",
        "scaler_path": "xgb_scaler.pkl"
    }
)