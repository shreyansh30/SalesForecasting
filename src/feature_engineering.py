"""Feature engineering for BrandX India time series forecasting."""
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Indian calendar-aware time features."""
    df = df.copy()
    df["dayofweek"]  = df.index.dayofweek          # 0=Mon, 6=Sun
    df["day"]        = df.index.day
    df["month"]      = df.index.month
    df["quarter"]    = df.index.quarter
    df["year"]       = df.index.year
    df["dayofyear"]  = df.index.dayofyear
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # Indian festive months (Diwali ~Oct/Nov, Navratri ~Sep/Oct, Holi ~Mar)
    df["is_festive_month"] = df["month"].isin([3, 9, 10, 11]).astype(int)

    # Indian financial year quarter (Apr–Jun=Q1, Jul–Sep=Q2, Oct–Dec=Q3, Jan–Mar=Q4)
    df["india_fy_quarter"] = df["month"].map(
        {4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 2, 10: 3, 11: 3, 12: 3, 1: 4, 2: 4, 3: 4}
    )
    return df


def add_lag_features(df: pd.DataFrame, target_col: str = "sales",
                     lags: list = [1, 7, 14, 30]) -> pd.DataFrame:
    """Add lag features (previous day/week/month sales)."""
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str = "sales",
                         windows: list = [7, 14, 30]) -> pd.DataFrame:
    """Add rolling mean and std features."""
    df = df.copy()
    for w in windows:
        df[f"rolling_mean_{w}"] = df[target_col].shift(1).rolling(window=w).mean()
        df[f"rolling_std_{w}"]  = df[target_col].shift(1).rolling(window=w).std()
    return df