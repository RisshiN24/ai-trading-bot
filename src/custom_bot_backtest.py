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

# Define your trading strategy
class MLTrader(Strategy):
    def initialize(self, symbol="SPY", cash_at_risk=0.5, model_path="best_model.keras"):
        self.symbol = symbol
        self.cash_at_risk = cash_at_risk
        self.model = load_model(model_path)  # Load the trained model
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
        thirty_days_prior = today - Timedelta(days=30)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d'), thirty_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self): 
        today, three_days_prior, thirty_days_prior = self.get_dates()

        news = self.api.get_news(symbol=self.symbol, 
                                 start=three_days_prior, 
                                 end=today) 
        news = [ev.__dict__["_raw"]["headline"] for ev in news]

        probability, sentiment = estimate_sentiment(news)
        return probability, sentiment
    
    def on_trading_iteration(self):
        # Get current cash, price, and size the position
        cash, last_price, quantity = self.position_sizing()
        today, three_days_prior, thirty_days_prior = self.get_dates()

        # Fetch the latest stock data (use get_bars to fetch historical bars)
        bars = self.api.get_bars(self.symbol, '1D', limit=1000, start=thirty_days_prior, end=today)  # Fetch latest 30 days

        df = pd.DataFrame([{
            'time': bar.t,
            'open': bar.o,
            'high': bar.h,
            'low': bar.l,
            'close': bar.c,
            'volume': bar.v
        } for bar in bars])

        # Feature engineering for model prediction (calculate moving average, RSI, etc.)
        df['price_change'] = df['close'].pct_change()
        df['moving_avg'] = df['close'].rolling(window=10).mean()
        df['RSI'] = calculate_rsi(df)

        # Prepare input data for prediction
        df = df[['price_change', 'moving_avg', 'RSI']].dropna()
        X = df.values
        X_scaled = self.scaler.fit_transform(X)

        # Make predictions
        if len(X_scaled) > 0:
            prediction = self.model.predict(X_scaled[-1].reshape(1, -1))  # Predict on the last available data
            prediction = (prediction > 0.5).astype(int)
            probability, sentiment = self.get_sentiment()

            # Trading logic based on prediction
            if cash > last_price:  # Ensure we have enough cash to trade
                if prediction == 1 or sentiment == "positive":  # Buy signal
                    if self.last_trade == "sell":
                        self.sell_all()  # Clear previous short position
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
                elif prediction == 0 or sentiment == "negative":  # Sell signal
                    if self.last_trade == "buy":
                        self.sell_all()  # Clear previous long position
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

# Backtesting with YahooDataBacktesting
start_date = datetime(2024, 9, 1)
end_date = datetime(2024, 10, 1)
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
