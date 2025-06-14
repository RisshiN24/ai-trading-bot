# Import dependencies
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, MaxPooling1D # type: ignore
from tensorflow.keras.metrics import AUC # type: ignore
from tensorflow.keras import backend as K # type: ignore
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore
from keras.saving import register_keras_serializable # type: ignore

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve

import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

from alpaca_trade_api import REST
from indicators import add_all_indicators

# Load environment variables
load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2"

# Initialize Alpaca API
api = REST(base_url=ALPACA_BASE_URL, key_id=ALPACA_API_KEY, secret_key=ALPACA_API_SECRET)

# Make sequences of specified length
def make_sequences(X, y, sequence_length=20):
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

# Get training data
def get_training_data(start_date="2014-01-01", end_date="2024-01-01", ticker="AAPL", sequence_length=20, target_return_threshold=0.01, future_days=3):
    
    # Pull historical OHLCV bars from Alpaca with retry logic
    max_retries = 3
    retry_delay = 60  # seconds
    
    for attempt in range(max_retries):
        try:
            print(f"Downloading {ticker} data from {start_date} to {end_date} (attempt {attempt + 1}/{max_retries})")
            
            # Get bars from Alpaca
            bars = api.get_bars(ticker, '1D', start=start_date, end=end_date)
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'time': b.t,
                'open': b.o,
                'high': b.h,
                'low': b.l,
                'close': b.c,
                'volume': b.v
            } for b in bars])
            
            # Check if download was successful
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")
            
            # Set time as index and sort
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)
            
            print(f"Successfully downloaded {len(df)} rows of data for {ticker}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            break
            
        except Exception as e:
            print(f"Error downloading data (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Waiting {retry_delay} seconds before retry...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("All download attempts failed. Please check your Alpaca credentials and try again.")
                raise e

    # Feature engineering
    df = add_all_indicators(df)

    # Add target
    df['future_price'] = df['close'].shift(-future_days)
    df['future_return'] = (df['future_price'] - df['close']) / df['close']
    df['target'] = (df['future_return'] > target_return_threshold).astype(int)

    # Clean data
    df = df.dropna()
    
    # Check if we have enough data after cleaning
    if len(df) < sequence_length + future_days:
        raise ValueError(f"Not enough data after cleaning. Got {len(df)} rows, need at least {sequence_length + future_days}")

    # Create X and y
    X = df[['close', 'MACD_line', 'RSI', 'ATR', 'BB_upper', 'BB_lower', 'volume']].values
    y = df['target'].astype(int).values

    # Make sequences
    return make_sequences(X, y, sequence_length)

# Define custom focal loss class for proper serialization (classes are best practice for custom loss functions)
@register_keras_serializable()
class FocalLoss:
    def __init__(self, gamma=2.0, alpha=0.5): # easily pass in parameters via __init__
        self.gamma = gamma
        self.alpha = alpha
        
    def __call__(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(K.equal(y_true, 1), self.alpha, 1 - self.alpha)
        loss = -alpha_t * K.pow(1 - p_t, self.gamma) * K.log(p_t)
        return K.mean(loss)
    
    def get_config(self): # necessary for serialization
        return {'gamma': self.gamma, 'alpha': self.alpha}
    
    @classmethod
    def from_config(cls, config): # necessary for serialization
        return cls(**config)

# Train model
def train_model(
    # Model Architecture Parameters
    conv_filters=16, 
    conv_kernel_size=3,
    maxpool_size=2,
    lstm_units=16, 
    num_lstm_layers=1,
    dense_units=32,
    dropout_rate=0.2,
    activation='relu',
    
    # Training Parameters
    sequence_length=20, 
    epochs=100, 
    batch_size=32,
    learning_rate=0.001,
    optimizer_name='adam',
    train_split=0.8,
    
    # Loss Function Parameters
    focal_gamma=2.0,
    focal_alpha=0.5,
    
    # Callback Parameters
    early_stopping_patience=10,
    reduce_lr_patience=3,
    reduce_lr_factor=0.5,
    min_lr=1e-6,
    
    # Data Parameters
    ticker="SPY",
    start_date="2020-01-01",
    end_date="2025-01-01",
    target_return_threshold=0.01,
    future_days=3
):
    # Initialize Weights & Biases (W&B)
    wandb.init(
        project="ai-trader",
        config={
            # Model Architecture
            "conv_filters": conv_filters,
            "conv_kernel_size": conv_kernel_size,
            "maxpool_size": maxpool_size,
            "lstm_units": lstm_units,
            "num_lstm_layers": num_lstm_layers,
            "dense_units": dense_units,
            "dropout_rate": dropout_rate,
            "activation": activation,
            
            # Training Parameters
            "sequence_length": sequence_length,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": optimizer_name,
            "train_split": train_split,
            
            # Loss Function
            "loss": "focal",
            "focal_gamma": focal_gamma,
            "focal_alpha": focal_alpha,
            
            # Callbacks
            "early_stopping_patience": early_stopping_patience,
            "reduce_lr_patience": reduce_lr_patience,
            "reduce_lr_factor": reduce_lr_factor,
            "min_lr": min_lr,
            
            # Data Parameters
            "ticker": ticker,
            "start_date": start_date,
            "end_date": end_date,
            "target_return_threshold": target_return_threshold,
            "future_days": future_days,
            
            "architecture": f"Conv+{num_lstm_layers}xLSTM"
        },
        name=f"Conv{conv_filters}-{num_lstm_layers}xLSTM{lstm_units}-seq{sequence_length}-bs{batch_size}-ep{epochs}-{ticker}"
    )

    # Get data
    X_seq, y_seq = get_training_data(
        start_date=start_date,
        end_date=end_date,
        ticker=ticker,
        sequence_length=sequence_length,
        target_return_threshold=target_return_threshold,
        future_days=future_days
    )

    # Print percentage of positive labels
    print(sum(y_seq)/len(y_seq))

    # Split data
    split = int(train_split * len(X_seq))
    X_train_raw, X_test_raw = X_seq[:split], X_seq[split:] # Maintain temporal order
    y_train, y_test = y_seq[:split], y_seq[split:]

    # Scale data
    scaler = MinMaxScaler()
    X_train_reshaped = X_train_raw.reshape(-1, X_train_raw.shape[-1]) # (samples * time, features)
    scaler.fit(X_train_reshaped)
    X_train_scaled = np.array([scaler.transform(x) for x in X_train_raw])
    X_test_scaled  = np.array([scaler.transform(x) for x in X_test_raw])
    joblib.dump(scaler, 'scaler.pkl') # Save scaler using joblib

    # Print X_train_scaled shape
    print(X_train_scaled.shape)

    ### MODEL ARCHITECTURE ###
    model = Sequential()
    
    # Convolutional layer to extract local patterns from the data
    model.add(Conv1D(conv_filters, conv_kernel_size, activation=activation, input_shape=(sequence_length, X_train_scaled.shape[2])))

    # Max pooling layer to downsample the data
    model.add(MaxPooling1D(maxpool_size))

    # Add dropout for regularization
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    # LSTM layers
    for i in range(num_lstm_layers):
        return_sequences = (i < num_lstm_layers - 1)  # Return sequences for all but last layer
        model.add(LSTM(lstm_units, return_sequences=return_sequences))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # Dense layer
    model.add(Dense(dense_units, activation=activation))

    # Add dropout before final layer
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))

    # Final dense layer with sigmoid activation for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Create optimizer
    if optimizer_name.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = 'adam'  # fallback

    # Compile model
    model.compile(
        optimizer=optimizer, 
        loss=FocalLoss(gamma=focal_gamma, alpha=focal_alpha), # Use focal loss with custom parameters
        metrics=[
            'accuracy', 
            AUC(curve='PR', name='pr_auc') # Log Precision-Recall AUC
        ]
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_pr_auc',
        mode='max',
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        min_lr=min_lr,
        verbose=1
    )

    # Callbacks for saving best model and for logging to W&B
    callbacks = [
        ModelCheckpoint('model.keras', monitor='val_pr_auc', save_best_only=True, mode='max', verbose=1),
        WandbMetricsLogger(),
        WandbModelCheckpoint(filepath='model_wandb.keras', save_best_only=True, monitor="val_pr_auc", mode="max"),
        EarlyStopping(monitor='val_pr_auc', patience=early_stopping_patience, mode='max', restore_best_weights=True),
        reduce_lr
    ]

    # Train model
    model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)

    # Load best model
    best_model = load_model('model.keras', custom_objects={'FocalLoss': FocalLoss})

    # Find best threshold
    probs = best_model.predict(X_test_scaled).flatten()
    prec, rec, thr = precision_recall_curve(y_test, probs)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-9)
    best_thr = thr[f1_scores.argmax()]
    joblib.dump(best_thr, "best_threshold.pkl") # Save best threshold

    y_pred = (probs > best_thr).astype(int)
    print(f"Best Threshold: {best_thr:.4f}")

    # Evaluate model
    acc = np.mean(y_pred == y_test)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    acc_05 = np.mean((probs > 0.5).astype(int) == y_test)
    print(f"Accuracy at 0.5: {acc_05:.4f}")
    print(f"Accuracy: {acc:.2f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # Log metrics
    wandb.log({"final_accuracy": acc, "final_precision": prec, "final_recall": rec, "final_f1_score": f1})

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    wandb.log({"confusion_matrix": wandb.Image(fig)})

    # Plot prediction confidence distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(probs, bins=40, kde=True, color="skyblue")
    plt.title("Prediction Confidence Distribution")
    plt.xlabel("Predicted Probability (Up)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # Conclude experiment
    wandb.finish()

# Run training if script is executed directly
if __name__ == "__main__":
    train_model()
