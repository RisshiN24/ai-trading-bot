# Import dependencies
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

from tensorflow.keras.models import load_model  # type: ignore
from lstm_model import focal_loss
from indicators import add_all_indicators
from joblib import load

# Load API credentials
load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2"

ALPACA_CREDS = {
    "API_KEY": ALPACA_API_KEY,
    "API_SECRET": ALPACA_API_SECRET,
    "PAPER": True
}

# Deep learning trading strategy
class MLTrader(Strategy):
    # Initialize the strategy
    def initialize(self, symbol="AAPL", cash_at_risk=1, model_path="model.keras", scaler_path="scaler.pkl", best_threshold_path="best_threshold.pkl"):
        self.symbol = symbol
        self.cash_at_risk = cash_at_risk
        self.model = load_model(model_path, custom_objects={"focal_loss": focal_loss})
        self.scaler = load(scaler_path)
        self.best_threshold = load(best_threshold_path)
        self.sleeptime = "72H"
        self.last_trade = None
        self.api = REST(base_url=ALPACA_BASE_URL, key_id=ALPACA_API_KEY, secret_key=ALPACA_API_SECRET)

    def position_sizing(self):
        cash = self.get_cash()
        price = self.get_last_price(self.symbol)
        qty = round(cash * self.cash_at_risk / price, 0) # Calculate position size
        return cash, price, qty 

    def get_dates(self):
        yday = self.get_datetime() - Timedelta(days=1)
        past = yday - Timedelta(days=100) # Look back 100 days
        return yday.strftime('%Y-%m-%d'), past.strftime('%Y-%m-%d') 

    def get_prediction(self):
        yday, past = self.get_dates() # Get dates
        bars = self.api.get_bars(self.symbol, '1D', limit=1000, start=past, end=yday)
        df = pd.DataFrame([{ 'time': b.t, 'open': b.o, 'high': b.h, 'low': b.l, 'close': b.c, 'volume': b.v } for b in bars])

        # Feature engineering
        df = self._add_features(df)
        df = df.dropna()

        # Scale features
        features = ['close', 'MACD_line', 'RSI', 'ATR_14', 'BB_upper', 'BB_lower', 'OBV']
        X_scaled = self.scaler.transform(df[features].values)

        if len(X_scaled) < 20:
            return None

        # Make prediction
        input_seq = X_scaled[-20:].reshape(1, 20, len(features)) # Shape: (1, sequence_length, n_features)
        pred = self.model.predict(input_seq)[0][0]

        print(f"Predicted probability: {pred}")
        return 1 if pred > self.best_threshold else 0 # Return prediction

    # Add features (engineered indicators)
    def _add_features(self, df):
        df = add_all_indicators(df)
        return df

    def on_trading_iteration(self):
        cash, price, max_qty = self.position_sizing() # Get position size
        prediction = self.get_prediction() # Get prediction

        print(f"[{self.get_datetime()}] Cash: {cash:.2f}, Price: {price:.2f}, Qty: {max_qty}, Prediction: {prediction}")

        if prediction is None:
            print("No prediction returned.")
            return

        pos = self.get_position(self.symbol)
        curr_qty = pos.quantity if pos else 0
        max_qty = min(max_qty, int(cash / price)) if price != 0 else 0 # Double check that position size is clamped to available cash

        if prediction == 1 and max_qty > 0: # Bullish signal
            print(f"Submitting BUY order for {max_qty} shares.")
            order = self.create_order(
                self.symbol, max_qty, "buy", type="bracket",
                take_profit_price=price * 1.03,
                stop_loss_price=price * 0.97
            )
            self.submit_order(order)
        elif prediction == 0 and curr_qty > 0: # Bearish signal
            print("Flattening position.")
            self.sell_all()


# Backtesting
start = datetime(2020, 1, 1)
end = datetime.today() - Timedelta(days=1)

strategy = MLTrader(
    name='MLTrader',
    broker=Alpaca(ALPACA_CREDS),
    parameters={"symbol": "SPY", "cash_at_risk": 0.5, "model_path": "model.keras"}
)

strategy.backtest(
    YahooDataBacktesting, start, end,
    parameters={"symbol": "SPY", "cash_at_risk": 0.5, "model_path": "model.keras"}
)
