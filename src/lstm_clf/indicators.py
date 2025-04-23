import numpy as np
import pandas as pd

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i - 1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

def add_ema_ribbon_features(df):
    spans = [8, 13, 21, 34, 55, 89]
    for span in spans:
        df[f'EMA_{span}'] = df['close'].ewm(span=span).mean()
    df['EMA8_to_21'] = df['EMA_8'] / df['EMA_21']
    df['EMA13_to_34'] = df['EMA_13'] / df['EMA_34']
    df['EMA21_to_55'] = df['EMA_21'] / df['EMA_55']
    df['EMA34_to_89'] = df['EMA_34'] / df['EMA_89']
    ribbon = pd.concat([df[f'EMA_{s}'] for s in spans], axis=1)
    df['ribbon_width'] = ribbon.max(axis=1) - ribbon.min(axis=1)
    df['EMA8_slope'] = df['EMA_8'].diff()
    df['EMA21_slope'] = df['EMA_21'].diff()
    return df

def add_all_indicators(df):
    df['price_change'] = df['close'].pct_change()
    df['close_rel'] = df['close'] / df['close'].rolling(20).mean()
    df['close_zscore'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
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
    df['stoch_k'] = 100 * ((df['close'] - df['low'].rolling(14).min()) / (df['high'].rolling(14).max() - df['low'].rolling(14).min()))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['ATR_14'] = calculate_atr(df)
    df['OBV'] = calculate_obv(df)
    df['OBV_ratio'] = df['OBV'] / df['OBV'].rolling(20).mean()
    return df
