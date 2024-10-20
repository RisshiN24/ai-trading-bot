# AI Trading Project

## Project Description

custom_bot.py: An ML/deep learning model (ANN) that is trained on SPY data. Features include RSI, 200EMA, MACD Line, and price change. If the model either predicts 1 (price will go up) or 0 (price will go down).

custom_bot_backtest.py: The code that allows us to backtest on the model we created in custom_bot.py. We also incorporate news sentiment into the trading logic, which is analyzed using FinBERT.

finbert_utils.py: The code in this file allows us to access the FinBERT model from Hugging Spaces and analyze the news headlines.

## Credits

Thanks to Nicholas Renotte and his trading bot tutorial for inspiration and for the code in finbert_utils.py.

The code in custom_bot.py and custom_bot_backtest.py is separate and builds on the code from the tutorial.