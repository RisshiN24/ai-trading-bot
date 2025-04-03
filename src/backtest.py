# backtest.py (Refactored and Cleaned)

import os
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from alpaca_trade_api import REST
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from timedelta import Timedelta
from tensorflow.keras.models import load_model
from model import calculate_rsi
from finbert_utils import estimate_sentiment

# Load environment variables
load_dotenv()

# Alpaca API setup
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2"

ALPACA_CREDS = {
    "API_KEY": ALPACA_API_KEY,
    "API_SECRET": ALPACA_API_SECRET,
    "PAPER": True
}

class MLTrader(Strategy):
    def initialize(self, symbol="SPY", cash_at_risk=0.5, model_path="model_fold_1.keras"):
        self.symbol = symbol
        self.cash_at_risk = cash_at_risk
        self.model = load_model(model_path)
        self.scaler = MinMaxScaler()
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
        three_days_prior = today - Timedelta(days=3)
        two_hundred_days_prior = today - Timedelta(days=200)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d'), two_hundred_days_prior.strftime('%Y-%m-%d')

    def get_prediction_and_sentiment(self):
        today, three_days_prior, two_hundred_days_prior = self.get_dates()

        # Fetch historical bars
        bars = self.api.get_bars(self.symbol, '1D', limit=1000, start=two_hundred_days_prior, end=today)
        df = pd.DataFrame([{ 'time': bar.t, 'open': bar.o, 'high': bar.h, 'low': bar.l, 'close': bar.c, 'volume': bar.v } for bar in bars])

        # Feature Engineering
        df['price_change'] = df['close'].pct_change()
        df['200EMA'] = df['close'].ewm(span=200, adjust=False).mean()
        df['MACD Line'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['RSI'] = calculate_rsi(df)
        df['price_to_ema'] = df['close'] / df['200EMA']
        df['price_minus_ema'] = df['close'] - df['200EMA']
        df['ema_slope'] = df['200EMA'].diff()

        # Get sentiment using FinBERT
        news = self.api.get_news(symbol=self.symbol, start=three_days_prior, end=today)
        headlines = [article.__dict__["_raw"].get("headline", "") for article in news]
        probability, sentiment = estimate_sentiment(headlines)

        df['sentiment_score'] = probability
        df = df.dropna()

        # Build feature matrix
        features = ['price_change', 'MACD Line', 'RSI', 'price_to_ema', 'price_minus_ema', 'ema_slope', 'sentiment_score']
        X = df[features].values
        X_scaled = self.scaler.fit_transform(X)

        # Sequence input for LSTM
        sequence_length = 20
        if len(X_scaled) < sequence_length:
            return None, None, None

        input_seq = X_scaled[-sequence_length:]
        input_seq = input_seq.reshape(1, sequence_length, len(features))
        pred_prob = self.model.predict(input_seq)[0][0]
        prediction = int(pred_prob > 0.5)

        return prediction, sentiment, probability

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()
        prediction, sentiment, probability = self.get_prediction_and_sentiment()

        if prediction is None:
            return

        if cash > last_price:
            if prediction == 1 or (sentiment == "positive" and probability > 0.75):
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=last_price * 1.20,
                    stop_loss_price=last_price * 0.95
                )
                self.submit_order(order)
                self.last_trade = "buy"

            elif prediction == 0 or (sentiment == "negative" and probability > 0.75):
                if self.last_trade == "buy":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type="bracket",
                    take_profit_price=last_price * 0.8,
                    stop_loss_price=last_price * 1.05
                )
                self.submit_order(order)
                self.last_trade = "sell"

# Backtest parameters
start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 3, 1)
broker = Alpaca(ALPACA_CREDS)

strategy = MLTrader(
    name='MLTrader',
    broker=broker,
    parameters={"symbol": "SPY", "cash_at_risk": 0.5, "model_path": "best_model.keras"}
)

strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol": "SPY", "cash_at_risk": 0.5}
)