# model.py (Updated to include real FinBERT sentiment via Alpaca headlines with K-fold Cross Validation)

import os
import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import KFold
from keras.callbacks import ModelCheckpoint
from dotenv import load_dotenv
from finbert_utils import estimate_sentiment

load_dotenv()

# Alpaca credentials
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2"

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL)

start_date = '2023-01-01'
end_date = '2024-01-01'

# Fetch bars
ticker = 'SPY'
bars = api.get_bars(ticker, '1D', limit=1000, start=start_date, end=end_date)
df = pd.DataFrame([{ 'time': bar.t, 'open': bar.o, 'high': bar.h, 'low': bar.l, 'close': bar.c, 'volume': bar.v } for bar in bars])

# RSI calculation
def calculate_rsi(data, window=14):
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=14).mean()
    avg_loss = loss.rolling(window=window, min_periods=14).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# Feature Engineering
df['price_change'] = df['close'].pct_change()
df['200EMA'] = df['close'].ewm(span=200, adjust=False).mean()
df['MACD Line'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
df['RSI'] = calculate_rsi(df)
df['price_to_ema'] = df['close'] / df['200EMA']
df['price_minus_ema'] = df['close'] - df['200EMA']
df['ema_slope'] = df['200EMA'].diff()

# Pull sentiment using real headlines from Alpaca API
sentiment_scores = []
for i in range(len(df)):
    today = pd.to_datetime(df.iloc[i]['time'])
    three_days_prior = today - pd.Timedelta(days=3)
    news = api.get_news(symbol=ticker, start=three_days_prior.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))
    headlines = [article.__dict__["_raw"].get("headline", "") for article in news]
    score, _ = estimate_sentiment(headlines)
    sentiment_scores.append(score)

df['sentiment_score'] = sentiment_scores

# Target label
df['future_price'] = df['close'].shift(-1)
df['target'] = (df['future_price'] > df['close']).astype(int)

# Drop NaNs
df = df.dropna()

# Features including sentiment
features = ['price_change', 'MACD Line', 'RSI', 'price_to_ema', 'price_minus_ema', 'ema_slope', 'sentiment_score']
X = df[features].values
y = df['target'].values

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Create sequences for LSTM
sequence_length = 20
X_seq, y_seq = [], []
for i in range(len(X_scaled) - sequence_length):
    X_seq.append(X_scaled[i:i+sequence_length])
    y_seq.append(y[i+sequence_length])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=False)
fold = 1
accuracies = []

for train_index, test_index in kf.split(X_seq):
    print(f"Training Fold {fold}...")
    X_train, X_test = X_seq[train_index], X_seq[test_index]
    y_train, y_test = y_seq[train_index], y_seq[test_index]

    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(f'model_fold_{fold}.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[checkpoint], verbose=0)
    best_model = load_model(f'model_fold_{fold}.keras')

    y_pred = (best_model.predict(X_test) > 0.5).astype(int)
    accuracy = np.mean(y_pred.flatten() == y_test)
    accuracies.append(accuracy)
    print(f'Fold {fold} Accuracy: {accuracy:.2f}')
    fold += 1

print(f'Average Cross-Validated Accuracy: {np.mean(accuracies):.2f}')