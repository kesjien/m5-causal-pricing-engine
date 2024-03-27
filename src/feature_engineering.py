import pandas as pd
import numpy as np

def generate_confounders(df: pd.DataFrame) -> pd.DataFrame:
    """Generates temporal lag features, rolling statistics, and log transforms for prices."""
    df = df.sort_values(by=["id", "date"]).reset_index(drop=True)
    
    # Log transformation for treatment variable (price)
    df['log_price'] = np.log(df['sell_price'].replace(0, np.nan))
    df['log_price'] = df['log_price'].fillna(method='ffill').fillna(0)
    
    # Lags and rolling means
    df['sales_lag_7'] = df.groupby('id')['sales'].shift(7)
    df['sales_roll_mean_7'] = df.groupby('id')['sales'].transform(lambda x: x.rolling(7, min_periods=1).mean())
    
    # Calendar interaction indicators
    df['is_snap'] = df[['snap_CA', 'snap_TX', 'snap_WI']].max(axis=1)
    
    return df.dropna().reset_index(drop=True)