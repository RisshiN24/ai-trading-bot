# AI Trading Project

## Project Description

This bot analyzes historical price data and financial news headlines to generate trading signals. It uses:

**- Technical indicators** like RSI, MACD, EMA slope

**- News sentiment** via FinBERT

**- LSTM neural network** to predict next-day market movement

Trades are executed based on both model prediction and sentiment confidence.

**Files:**
- **model.py:** Contains the code for data retrieval, feature engineering, model training, and evaluation. It builds the ML model using technical indicators and saves the best performing model. Uses an LSTM architecutre.
- **backtest.py:** Implements a custom trading strategy that uses the pre-trained ML model from model.py to generate buy/sell signals. It is integrated with the Lumibot framework for backtesting the strategy.
- **finbert_utils.py:** Provides utility functions to access and use the FinBERT model for sentiment analysis on financial news headlines.

## Future Improvements

- **Live Trading Integration:** Enhance the system to support live trading, allowing the strategy to be deployed in a real-time trading environment.
- **Advanced Feature Engineering:** Further improve feature engineering by exploring additional technical indicators and alternative data sources (Twitter sentiment, for instance).
- **Robust Error Handling & Logging:** Implement more robust error handling and logging mechanisms to ensure system reliability, especially in live trading scenarios.

## Acknowledgements

Special thanks to Nicholas Renotte and his trading bot tutorial for the inspiration and for the code in finbert_utils.py. The implementations in model.py and backtest.py build upon the concepts presented in the tutorial while extending them with additional features and improvements.