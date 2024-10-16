import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import alpaca_trade_api as tradeapi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from dotenv import load_dotenv
import os

from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime
from alpaca_trade_api import REST
from timedelta import Timedelta

# from sklearn.ensemble import RandomForestClassifier
#from datetime import datetime
# from sklearn.metrics import accuracy_score
# from xgboost import XGBClassifier

load_dotenv()

# Set up Alpaca API
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2"

ALPACA_CREDS = {
    "API_KEY":ALPACA_API_KEY,
    "API_SECRET":ALPACA_API_SECRET,
    "PAPER": True
}

api = tradeapi.REST(ALPACA_API_KEY , ALPACA_API_SECRET, ALPACA_BASE_URL)

# Define the start and end date for the historical data
start_date = '2023-01-01'  # Adjust as needed
end_date = '2024-01-01'    # Adjust as needed

# Fetch historical stock data using get_bars with a time range
bars = api.get_bars(
    'SPY',                # Stock symbol
    '1D',                  # Timeframe (daily bars)
    limit=1000,            # Max number of bars to retrieve
    start=start_date,      # Start date
    end=end_date           # End date
)

# Convert the bars to a DataFrame
df = pd.DataFrame([{
    'time': bar.t,
    'open': bar.o,
    'high': bar.h,
    'low': bar.l,
    'close': bar.c,
    'volume': bar.v
} for bar in bars])


def calculate_rsi(data, window=14):
    # Calculate the difference in prices between consecutive days
    delta = data['close'].diff()

    # Separate gains and losses
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))

    # Calculate the rolling average of gains and losses
    avg_gain = gain.rolling(window=window, min_periods=14).mean()
    avg_loss = loss.rolling(window=window, min_periods=14).mean()

    # Calculate the relative strength (RS)
    rs = avg_gain / avg_loss

    # Calculate RSI using the RS
    rsi = 100 - (100 / (1 + rs))

    return rsi


# Feature engineering
df['price_change'] = df['close'].pct_change()
df['moving_avg'] = df['close'].rolling(window=10).mean()
df['RSI'] = calculate_rsi(df)

# Label generation: Predict if the price will go up (1) or down (0)
df['future_price'] = df['close'].shift(-1)
df['target'] = (df['future_price'] > df['close']).astype(int)

# Drop NaN values
df = df[['price_change', 'moving_avg', 'RSI', 'target']].dropna()

# Split data into features and target
X = df[['price_change', 'moving_avg', 'RSI']].values
y = df['target'].values

# Scale the features to [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Build a basic ANN model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))  # First hidden layer
model.add(Dropout(0.2))  # Regularization
model.add(Dense(64, activation='relu'))  # Second hidden layer
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define the checkpoint callback
checkpoint = ModelCheckpoint(
    'best_model.keras',  # Path where to save the model
    monitor='val_accuracy',  # Metric to monitor
    save_best_only=True,  # Only save the model if val_accuracy improved
    mode='max',  # We want to maximize validation accuracy
    verbose=1  # Print messages when saving the model
)

# Train the model with the checkpoint
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[checkpoint])

# Load the best model
best_model = load_model('best_model.keras')

# Make predictions
y_pred = (best_model.predict(X_test) > 0.5).astype(int)

# Evaluate accuracy
accuracy = np.mean(y_pred.flatten() == y_test)
print(f'Best Model Accuracy: {accuracy:.2f}')

# Plotting the training history
plt.figure(figsize=(12, 6))

# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')

plt.show()