# === Imports ===
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime
from alpaca_trade_api import REST
from timedelta import Timedelta
from datetime import time as dt_time
from finbert_utils import estimate_sentiment
import os
from dotenv import load_dotenv

# === Load environment variables ===
load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2"

# === Alpaca credentials for Lumibot ===
ALPACA_CREDS = {
    "API_KEY": ALPACA_API_KEY,
    "API_SECRET": ALPACA_API_SECRET,
    "PAPER": True
}

# === Strategy that trades based on FinBERT news sentiment ===
class MLTrader(Strategy):
    def initialize(self, symbol: str = "SPY", cash_at_risk: float = 0.5):
        self.symbol = symbol
        self.sleeptime = "1H"  # Run every hour
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=ALPACA_BASE_URL, key_id=ALPACA_API_KEY, secret_key=ALPACA_API_SECRET)

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price, 0)
        return cash, last_price, quantity

    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def get_sentiment(self):
        # Get recent news headlines and run sentiment analysis using FinBERT
        today, three_days_prior = self.get_dates()
        news = self.api.get_news(symbol=self.symbol, start=three_days_prior, end=today)
        headlines = [ev.__dict__["_raw"]["headline"] for ev in news]
        probability, sentiment = estimate_sentiment(headlines)
        return probability, sentiment

    def on_trading_iteration(self):
        # Only trade at 10:00 AM
        current_time = self.get_datetime().time()
        if current_time.hour != 10:
            return

        cash, last_price, quantity = self.position_sizing()
        probability, sentiment = self.get_sentiment()

        print(f"[{self.get_datetime()}] Cash: {cash:.2f}, Last Price: {last_price:.2f}, Probability: {probability}, Sentiment: {sentiment}")

        if cash > last_price:
            if sentiment == "positive" and probability > 0.75:
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
            elif sentiment == "negative" and probability > 0.75:
                if self.last_trade == "buy":
                    self.sell_all()

# === Backtest Configuration ===
start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 4, 10)
broker = Alpaca(ALPACA_CREDS)

strategy = MLTrader(
    name='mlstrat',
    broker=broker,
    parameters={"symbol": "SPY", "cash_at_risk": 0.5}
)

strategy.backtest(
    YahooDataBacktesting,
    start_date,
    end_date,
    parameters={"symbol": "SPY", "cash_at_risk": 0.5}
)
