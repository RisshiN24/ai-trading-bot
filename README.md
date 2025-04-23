# AI Trading Project

## Project Description

I've created a trading bot that uses a deep learning model to predict market movement. The model is trained on technical indicators, like RSI, MACD, and EMAs. Trades are executed based on model predictions. The main bot that I've been using is in the `lstm_clf/` folder, which contains the code for the model (`lstm_model.py`) and the code for the bot and its trading logic (`lstm_bot.py`). 

There are other folders that contain bots I'm still experimenting with, including:
- A rules-based bot
- A bot using an XGBoost classification model
- A bot using a sentiment model called FinBERT to analyze news headlines

## Future Improvements

- **Live Trading Integration**  
  Enhance the system to support live trading, allowing the strategy to be deployed in a real-time trading environment.

- **Advanced Feature Engineering**  
  Further improve feature engineering by exploring additional technical indicators and alternative data sources (e.g., Twitter sentiment).

- **Robust Error Handling & Logging**  
  Implement more robust error handling and logging mechanisms to ensure system reliability, especially in live trading scenarios.

## Acknowledgements

Thanks to Nicholas Renotte and his Alpaca trading bot tutorial for the inspiration to start this personal project. The sentiment bot code is primarily adapted from the tutorial.
