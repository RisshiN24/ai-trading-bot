# Import dependencies
import os
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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
import alpaca_trade_api as tradeapi

from indicators import add_all_indicators

# Load API credentials
load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2"

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL)

# Make sequences of specified length
def make_sequences(X, y, sequence_length=20):
    X_seq, y_seq = [], []
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

# Get training data
def get_training_data(start_date="2017-01-01", end_date="2023-12-31", ticker="AAPL", sequence_length=20):
    bars = api.get_bars(ticker, '1D', limit=10000, start=start_date, end=end_date)
    # Convert bars to DataFrame
    df = pd.DataFrame([{ 'time': b.t, 'open': b.o, 'high': b.h, 'low': b.l, 'close': b.c, 'volume': b.v } for b in bars])

    # Feature engineering
    df = add_all_indicators(df)

    # Add target
    df['future_price'] = df['close'].shift(-3)
    df['future_return'] = (df['future_price'] - df['close']) / df['close']
    df['target'] = (df['future_return'] > 0.01).astype(int)

    # Clean data
    df = df.dropna()

    # Create X and y
    X = df[['close', 'MACD_line', 'RSI', 'ATR_14', 'BB_upper', 'BB_lower', 'OBV']].values
    y = df['target'].astype(int).values

    # Make sequences
    return make_sequences(X, y, sequence_length)

# Define custom loss function (focal loss)
@register_keras_serializable()
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.5):
    y_true = tf.cast(y_true, tf.float32)
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_t = tf.where(K.equal(y_true, 1), alpha, 1 - alpha)
    loss = -alpha_t * K.pow(1 - p_t, gamma) * K.log(p_t)
    return K.mean(loss)

# Train model
def train_model(conv_filters=16, lstm_units=16, sequence_length=20, epochs=100, batch_size=32):
    # Initialize Weights & Biases (W&B)
    wandb.init(
        project="algo-trading",
        config={
            "conv_units": conv_filters, 
            "lstm_units": lstm_units,  
            "sequence_length": sequence_length, 
            "epochs": epochs, 
            "batch_size": batch_size, 
            "optimizer": "adam", 
            "loss": "focal", 
            "architecture": "Conv+LSTM"
        },
        name=f"Conv{conv_filters}-LSTM{lstm_units}-seq{sequence_length}-bs{batch_size}-ep{epochs}"
    )

    # Get data
    X_seq, y_seq = get_training_data(sequence_length=sequence_length)

    # Print percentage of positive labels
    print(sum(y_seq)/len(y_seq))

    # Split data (80% train, 20% test)
    split = int(0.8 * len(X_seq))
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

    # Build model
    model = Sequential([
        # Convultional layer to extract local patterns from the data
        Conv1D(conv_filters, 3, activation='relu', input_shape=(sequence_length, X_train_scaled.shape[2])),

        # Max pooling layer to downsample the data
        MaxPooling1D(2),
        
        # LSTM layer to learn long-term patterns and trends
        LSTM(lstm_units, return_sequences=False),

        Dense(32, activation='relu'),

        # Dense layer with sigmoid activation for binary classification
        Dense(1, activation='sigmoid')
    ])

    # Compile model
    model.compile(
        optimizer='adam', 
        loss=focal_loss, # Use focal loss
        metrics=[
            'accuracy', 
            AUC(curve='PR', name='pr_auc') # Log Precision-Recall AUC
        ]
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_pr_auc',       # or 'val_loss'
        mode='max',                 # 'min' for val_loss, 'max' for metrics like pr_auc
        factor=0.5,                 # reduce LR by this factor
        patience=3,                 # epochs to wait before reducing
        min_lr=1e-6,                # floor value
        verbose=1
    )

    # Callbacks for saving best model and for logging to W&B
    callbacks = [
        ModelCheckpoint('model.keras', monitor='val_pr_auc', save_best_only=True, mode='max', verbose=1),
        WandbMetricsLogger(),
        WandbModelCheckpoint(filepath='model_wandb.keras', save_best_only=True, monitor="val_pr_auc", mode="max"),
        EarlyStopping(monitor='val_pr_auc', patience=10, mode='max', restore_best_weights=True),
        reduce_lr
    ]

    # Train model
    model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)

    # Load best model
    best_model = load_model('model.keras')

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
