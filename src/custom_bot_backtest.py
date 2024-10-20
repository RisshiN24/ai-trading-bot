import numpy as np
import pandas as pd
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime
from tensorflow.keras.models import load_model
from alpaca_trade_api import REST
from timedelta import Timedelta
import os
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from custom_bot import calculate_rsi
from finbert_utils import estimate_sentiment

# Load environment variables (API keys, secrets, etc.)
load_dotenv()

# Setup Alpaca API credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2"

# Alpaca credentials used for trading
ALPACA_CREDS = {
    "API_KEY": ALPACA_API_KEY,
    "API_SECRET": ALPACA_API_SECRET,
    "PAPER": True
}

# Define a custom trading strategy
class MLTrader(Strategy):
    def initialize(self, symbol="SPY", cash_at_risk=0.5, model_path="best_model.keras"):
        # Initialize with stock symbol, cash allocation, model, and Alpaca REST API connection
        self.symbol = symbol
        self.cash_at_risk = cash_at_risk
        self.model = load_model(model_path)  # Load pre-trained ML model
        self.scaler = MinMaxScaler()  # Feature scaling for predictions
        self.sleeptime = "24H"  # Sleep time between trading iterations
        self.last_trade = None  # To track last trade action
        self.api = REST(base_url=ALPACA_BASE_URL, key_id=ALPACA_API_KEY, secret_key=ALPACA_API_SECRET)

    def position_sizing(self):
        # Calculate position size based on available cash and current stock price
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price, 0)
        return cash, last_price, quantity

    def get_dates(self):
        # Calculate key dates for trading: today, 3 days prior, and 200 days prior
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        two_hundred_days_prior = today - Timedelta(days=200)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d'), two_hundred_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self):
        # Fetch news for the stock and estimate sentiment using a sentiment analysis model
        today, three_days_prior, _ = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, start=three_days_prior, end=today)
        news = [ev.__dict__["_raw"]["headline"] for ev in news]  # Extract headlines from news events
        probability, sentiment = estimate_sentiment(news)  # Use a sentiment model to evaluate
        return probability, sentiment

    def get_prediction(self):
        # Fetch stock data, engineer features, and make predictions using the ML model
        today, _, two_hundred_days_prior = self.get_dates()
        bars = self.api.get_bars(self.symbol, '1D', limit=1000, start=two_hundred_days_prior, end=today)
        df = pd.DataFrame([{
            'time': bar.t,
            'open': bar.o,
            'high': bar.h,
            'low': bar.l,
            'close': bar.c,
            'volume': bar.v
        } for bar in bars])

        # Feature engineering for model input
        df['price_change'] = df['close'].pct_change()
        df['200EMA'] = df['close'].ewm(span=200, adjust=False).mean()
        df['MACD Line'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        df['RSI'] = calculate_rsi(df)

        # Prepare features for prediction
        df = df[['price_change', '200EMA', 'MACD Line', 'RSI']].dropna()
        X = df.values
        X_scaled = self.scaler.fit_transform(X)  # Scale features to [0, 1]

        # Predict on the most recent data point
        if len(X_scaled) > 0:
            prediction = self.model.predict(X_scaled[-1].reshape(1, -1))
            prediction = (prediction > 0.5).astype(int)
            return prediction

    def on_trading_iteration(self):
        # Perform trading logic on each iteration
        cash, last_price, quantity = self.position_sizing()
        prediction = self.get_prediction()
        probability, sentiment = self.get_sentiment()

        # Trading rules: buy or sell based on prediction and sentiment
        if cash > last_price:  # Ensure enough cash to trade
            if prediction == 1 or (sentiment == "positive" and probability > 0.75):  # Buy signal
                if self.last_trade == "sell":
                    self.sell_all()  # Close any short positions
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
            elif prediction == 0 or (sentiment == "negative" and probability > 0.75):  # Sell signal
                if self.last_trade == "buy":
                    self.sell_all()  # Close any long positions
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

# Set up backtesting parameters
start_date = datetime(2024, 9, 1)
end_date = datetime(2024, 10, 1)
broker = Alpaca(ALPACA_CREDS)

# Create a strategy instance
strategy = MLTrader(
    name='MLTrader',
    broker=broker,
    parameters={"symbol": "SPY", "cash_at_risk": 0.5, "model_path": "best_model.keras"}
)

# Run backtest using YahooDataBacktesting
strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol": "SPY", "cash_at_risk": 0.5}
)
