# === Imports ===
import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# === Load API credentials ===
load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets/v2"
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL)

# === Technical Indicators ===
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def add_ema_ribbon_features(df):
    spans = [8, 13, 21, 34, 55, 89]
    for span in spans:
        df[f'EMA_{span}'] = df['close'].ewm(span=span).mean()

    df['EMA8_to_21'] = df['EMA_8'] / df['EMA_21']
    df['EMA13_to_34'] = df['EMA_13'] / df['EMA_34']
    df['EMA21_to_55'] = df['EMA_21'] / df['EMA_55']
    df['EMA34_to_89'] = df['EMA_34'] / df['EMA_89']

    ribbon_emas = [df[f'EMA_{s}'] for s in spans]
    df['ribbon_width'] = pd.concat(ribbon_emas, axis=1).max(axis=1) - pd.concat(ribbon_emas, axis=1).min(axis=1)
    df['EMA8_slope'] = df['EMA_8'].diff()
    df['EMA21_slope'] = df['EMA_21'].diff()
    return df

# === Create time-windowed feature matrix ===
def create_windowed_dataset(X, y, window_size=5):
    X_windows, y_labels = [], []
    for i in range(window_size, len(X)):
        X_window = X[i - window_size:i].flatten()
        X_windows.append(X_window)
        y_labels.append(y[i])
    return np.array(X_windows), np.array(y_labels)

# === Data Fetching and Feature Engineering ===
def get_data(ticker="SPY", start="2014-01-01", end="2023-12-31"):
    bars = api.get_bars(ticker, '1D', limit=10000, start=start, end=end)
    df = pd.DataFrame([{ 
        'time': bar.t, 'open': bar.o, 'high': bar.h,
        'low': bar.l, 'close': bar.c, 'volume': bar.v 
    } for bar in bars])

    # === Add indicators ===
    df['price_change'] = df['close'].pct_change()
    df['MACD_line'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['RSI'] = calculate_rsi(df)
    df = add_ema_ribbon_features(df)
    df['volatility'] = df['price_change'].rolling(5).std()
    df['volume_change'] = df['volume'].pct_change()
    df['body_size'] = abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    df['body_to_range'] = df['body_size'] / df['range']
    df['BB_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
    df['BB_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
    df['bb_percent'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['ret_1d'] = df['close'].pct_change(1)
    df['ret_3d'] = df['close'].pct_change(3)
    df['ret_5d'] = df['close'].pct_change(5)
    df['stoch_k'] = 100 * ((df['close'] - df['low'].rolling(14).min()) /
                           (df['high'].rolling(14).max() - df['low'].rolling(14).min()))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # === Label: 3-day future return threshold ===
    df['future_price'] = df['close'].shift(-3)
    df['future_return'] = (df['future_price'] - df['close']) / df['close']
    df['target'] = (df['future_return'] > 0.01).astype(int)

    df = df.dropna()
    print(df.shape)

    # === Select features ===
    features = [
        'MACD_line', 'RSI',
        'EMA_8', 'EMA_13', 'EMA_21', 'EMA_34', 'EMA_55', 'EMA_89',
        'EMA8_to_21', 'EMA13_to_34', 'EMA21_to_55', 'EMA34_to_89',
        'ribbon_width', 'EMA8_slope', 'EMA21_slope',
        'volatility', 'volume_change', 'body_size', 'range', 'body_to_range',
        'BB_upper', 'BB_lower', 'bb_percent', 'momentum_5',
        'ret_1d', 'ret_3d', 'ret_5d', 'stoch_k', 'stoch_d'
    ]

    X = df[features].values
    y = df['target'].values
    return X, y

# === Train XGBoost Classifier ===
def train_xgb_model():
    X, y = get_data()
    X, y = create_windowed_dataset(X, y, window_size=5)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === Scale inputs ===
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "xgb_scaler.pkl")

    # === Class imbalance adjustment ===
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # === Model & hyperparam tuning ===
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )

    param_dist = {
        "n_estimators": [100, 200, 300, 400],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [3, 4, 5, 6, 7],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    }

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=25,
        scoring='f1',
        verbose=1,
        n_jobs=-1,
        cv=3
    )

    random_search.fit(X_train_scaled, y_train)
    best_model = random_search.best_estimator_
    joblib.dump(best_model, "xgb_model.pkl")

    # === Evaluation ===
    y_pred = best_model.predict(X_test_scaled)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print("Best Params:", random_search.best_params_)
    print("Feature Importances:", best_model.feature_importances_)

    # === Confusion Matrix ===
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# === Run Training ===
if __name__ == "__main__":
    train_xgb_model()
