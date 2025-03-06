# AI Trading Project

## Project Description

This project is an AI-driven trading system that leverages machine learning and deep learning models to analyze historical market data and news sentiment. The system currently uses a feedforward neural network (ANN) trained on SPY data with technical indicators such as RSI, 200EMA, MACD Line, and price change. Trading decisions are made based on both the model's prediction and news sentiment analyzed via FinBERT.

**Files:**
- **model.py:** Contains the code for data retrieval, feature engineering, model training, and evaluation. It builds the ML model using technical indicators and saves the best performing model.
- **backtest.py:** Implements a custom trading strategy that uses the pre-trained ML model from model.py along with news sentiment analysis (via finbert_utils.py) to generate buy/sell signals. It is integrated with the Lumibot framework for backtesting the strategy.
- **finbert_utils.py:** Provides utility functions to access and use the FinBERT model for sentiment analysis on financial news headlines.

## Future Improvements

- **Live Trading Integration:** Enhance the system to support live trading, allowing the strategy to be deployed in a real-time trading environment.
- **Model Experimentation:** Experiment with alternative models such as LSTM (Long Short-Term Memory) networks, which may be more appropriate for capturing temporal dependencies in market data.
- **Advanced Feature Engineering:** Further improve feature engineering by exploring additional technical indicators and alternative data sources.
- **Robust Error Handling & Logging:** Implement more robust error handling and logging mechanisms to ensure system reliability, especially in live trading scenarios.

## Acknowledgements

Special thanks to Nicholas Renotte and his trading bot tutorial for the inspiration and for the code in finbert_utils.py. The implementations in model.py and backtest.py build upon the concepts presented in the tutorial while extending them with additional features and improvements.