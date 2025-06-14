import numpy as np
import pandas as pd

def add_rsi(df, period=14):
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def add_atr(df, period=14):
    df = df.copy()
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=period).mean()
    return df

def add_obv(df):
    df = df.copy()
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i - 1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = pd.Series(obv, index=df.index)
    return df

def add_ema_ribbon_features(df):
    df = df.copy()
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

def add_price_metrics(df):
    df = df.copy()
    df['price_change'] = df['close'].pct_change()
    df['close_rel'] = df['close'] / df['close'].rolling(20).mean()
    df['close_zscore'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
    return df

def add_macd(df):
    df = df.copy()
    df['MACD_line'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    return df

def add_volatility(df, period=5):
    df = df.copy()
    if 'price_change' not in df.columns:
        df['price_change'] = df['close'].pct_change()
    df['volatility'] = df['price_change'].rolling(period).std()
    return df

def add_volume_metrics(df):
    df = df.copy()
    df['volume_change'] = df['volume'].pct_change()
    return df

def add_candlestick_metrics(df):
    df = df.copy()
    df['body_size'] = abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    df['body_to_range'] = df['body_size'] / df['range']
    return df

def add_bollinger_bands(df, period=20, std_dev=2):
    df = df.copy()
    df['BB_upper'] = df['close'].rolling(period).mean() + std_dev * df['close'].rolling(period).std()
    df['BB_lower'] = df['close'].rolling(period).mean() - std_dev * df['close'].rolling(period).std()
    df['bb_percent'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    return df

def add_momentum(df, period=5):
    df = df.copy()
    df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
    return df

def add_return_metrics(df):
    df = df.copy()
    df['ret_1d'] = df['close'].pct_change(1)
    df['ret_3d'] = df['close'].pct_change(3)
    df['ret_5d'] = df['close'].pct_change(5)
    return df

def add_stochastic(df, k_period=14, d_period=3):
    df = df.copy()
    df['stoch_k'] = 100 * ((df['close'] - df['low'].rolling(k_period).min()) / (df['high'].rolling(k_period).max() - df['low'].rolling(k_period).min()))
    df['stoch_d'] = df['stoch_k'].rolling(d_period).mean()
    return df

def add_obv_ratio(df, period=20):
    df = df.copy()
    if 'OBV' not in df.columns:
        df = add_obv(df)
    df['OBV_ratio'] = df['OBV'] / df['OBV'].rolling(period).mean()
    return df

def add_all_indicators(df):
    df = df.copy()
    df = add_price_metrics(df)
    df = add_macd(df)
    df = add_rsi(df)
    df = add_ema_ribbon_features(df)
    df = add_volatility(df)
    df = add_volume_metrics(df)
    df = add_candlestick_metrics(df)
    df = add_bollinger_bands(df)
    df = add_momentum(df)
    df = add_return_metrics(df)
    df = add_stochastic(df)
    df = add_atr(df)
    df = add_obv(df)
    df = add_obv_ratio(df)
    return df
