# Standard libraries
import os

# Data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Trading API
import alpaca_trade_api as tradeapi

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint

# Environment variables
from dotenv import load_dotenv

#-----------------------------------------------------------#

# Load environment variables for API keys
load_dotenv()

# Set up Alpaca API credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2"

# Alpaca credentials dictionary
ALPACA_CREDS = {
    "API_KEY": ALPACA_API_KEY,
    "API_SECRET": ALPACA_API_SECRET,
    "PAPER": True
}

# Initialize Alpaca API
api = tradeapi.REST(ALPACA_API_KEY , ALPACA_API_SECRET, ALPACA_BASE_URL)

# Define the start and end dates for historical data retrieval
start_date = '2023-01-01'
end_date = '2024-01-01'

# Fetch historical stock data (SPY) with daily bars
bars = api.get_bars(
    'SPY',                # Stock symbol
    '1D',                 # Timeframe (daily bars)
    limit=1000,           # Max number of bars to retrieve
    start=start_date,     # Start date
    end=end_date          # End date
)

# Convert the retrieved bars into a DataFrame
df = pd.DataFrame([{
    'time': bar.t,
    'open': bar.o,
    'high': bar.h,
    'low': bar.l,
    'close': bar.c,
    'volume': bar.v
} for bar in bars])

# RSI calculation function (Relative Strength Index)
def calculate_rsi(data, window=14):
    # Calculate the difference in prices between consecutive days
    delta = data['close'].diff()
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    
    # Calculate rolling averages of gains and losses
    avg_gain = gain.rolling(window=window, min_periods=14).mean()
    avg_loss = loss.rolling(window=window, min_periods=14).mean()
    
    # Calculate relative strength (RS) and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Feature engineering: adding price change, EMA, MACD Line, and RSI
df['price_change'] = df['close'].pct_change()  # Percentage change in price
df['200EMA'] = df['close'].ewm(span=200, adjust=False).mean()  # 200-period exponential moving average
df['MACD Line'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()  # MACD Line
df['RSI'] = calculate_rsi(df)  # RSI calculation

# Create target label: if next day's price is higher than the current day's price, target = 1, else target = 0
df['future_price'] = df['close'].shift(-1)
df['target'] = (df['future_price'] > df['close']).astype(int)

# Drop NaN values caused by feature engineering
df = df[['price_change', '200EMA', 'RSI', 'MACD Line', 'target']].dropna()

# Split data into features (X) and target labels (y)
X = df[['price_change', '200EMA', 'MACD Line', 'RSI']].values
y = df['target'].values

# Scale features to be between [0, 1] using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Build a sequential neural network model for classification
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))  # Input layer + first hidden layer with 128 units
model.add(Dropout(0.4))  # Add dropout to prevent overfitting
model.add(Dense(64, activation='relu'))  # Second hidden layer with 64 units
model.add(Dropout(0.4))  # Add dropout
model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model with Adam optimizer and binary crossentropy loss function
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define a checkpoint to save the best model based on validation accuracy
checkpoint = ModelCheckpoint(
    'best_model.keras',  # Filepath to save the model
    monitor='val_accuracy',  # Monitor validation accuracy
    save_best_only=True,  # Save only if validation accuracy improves
    mode='max',  # Maximize validation accuracy
    verbose=1  # Print progress
)

# Check if a pre-trained model exists
if os.path.exists('best_model.keras'):
    print("Model already exists. Loading pre-trained model.")
    best_model = load_model('best_model.keras')  # Load the pre-trained model
else:
    print("No pre-trained model found. Training a new model.")
    
    # Train the model with checkpoint callback
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[checkpoint])

    # Load the best model after training
    best_model = load_model('best_model.keras')

# Make predictions on the test set
y_pred = (best_model.predict(X_test) > 0.5).astype(int)

# Evaluate the model's accuracy
accuracy = np.mean(y_pred.flatten() == y_test)
print(f'Best Model Accuracy: {accuracy:.2f}')