"""Forecasting models for BrandX India Sales — small monthly dataset."""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler


# ─── ARIMA ───────────────────────────────────────────────

def train_arima(train_series: pd.Series, order: tuple = (2, 1, 1)):
    """Train ARIMA. Default (2,1,1) works best for monthly retail data."""
    model = ARIMA(train_series, order=order)
    return model.fit()


def forecast_arima(fitted_model, steps: int) -> pd.Series:
    return fitted_model.forecast(steps=steps)


# ─── HOLT-WINTERS ────────────────────────────────────────

def train_holtwinters(train_series: pd.Series):
    """
    Holt-Winters Exponential Smoothing.
    Best for seasonal monthly data. Always works — no external deps.
    Uses additive trend + seasonality with period=12 (months).
    Falls back to simpler model if not enough data.
    """
    n = len(train_series)
    try:
        if n >= 24:
            # Full seasonal model needs at least 2 full years
            model = ExponentialSmoothing(
                train_series,
                trend="add",
                seasonal="add",
                seasonal_periods=12
            ).fit(optimized=True)
        else:
            # Simple trend-only model for short series
            print(f"   INFO: Only {n} data points — using trend-only Holt-Winters.")
            model = ExponentialSmoothing(
                train_series,
                trend="add",
                seasonal=None
            ).fit(optimized=True)
        return model
    except Exception as e:
        print(f"   WARNING: Holt-Winters failed ({e}). Using simple exponential smoothing.")
        return ExponentialSmoothing(train_series).fit()


def forecast_holtwinters(fitted_model, steps: int) -> pd.Series:
    return fitted_model.forecast(steps)


# ─── PROPHET ─────────────────────────────────────────────

def train_prophet(df: pd.DataFrame, target_col: str = "sales"):
    """Prophet with automatic Holt-Winters fallback on Windows."""
    try:
        from prophet import Prophet
        prophet_df = df.reset_index().rename(
            columns={"date": "ds", target_col: "y"}
        )
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_mode="additive"
        )
        model.add_seasonality(name="monthly", period=30.5, fourier_order=3)
        model.fit(prophet_df)
        print("   Prophet trained successfully.")
        return ("prophet", model)

    except Exception as e:
        print(f"   Prophet failed: {str(e)[:80]}")
        print("   Falling back to Holt-Winters...")
        hw = train_holtwinters(df[target_col])
        print("   Holt-Winters trained successfully.")
        return ("holtwinters", hw)


def forecast_prophet(model_tuple, periods: int = 6,
                     df: pd.DataFrame = None) -> pd.DataFrame:
    """Forecast using Prophet or Holt-Winters fallback."""
    model_type, model = model_tuple

    if model_type == "prophet":
        future   = model.make_future_dataframe(periods=periods, freq="MS")
        forecast = model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    else:
        # Holt-Winters fallback — build future date index
        last_date  = df.index[-1] if df is not None else pd.Timestamp("today")
        future_idx = pd.date_range(
            start=last_date + pd.offsets.MonthBegin(1),
            periods=periods, freq="MS"
        )
        # Also include historical fitted values for test matching
        all_idx  = pd.date_range(
            start=df.index[0] if df is not None else last_date,
            periods=len(df) + periods if df is not None else periods,
            freq="MS"
        )
        all_preds = list(model.fittedvalues) + list(model.forecast(periods))
        return pd.DataFrame({
            "ds"         : all_idx[:len(all_preds)],
            "yhat"       : all_preds,
            "yhat_lower" : [v * 0.92 for v in all_preds],
            "yhat_upper" : [v * 1.08 for v in all_preds]
        })


# ─── LSTM ────────────────────────────────────────────────

def create_sequences(data: np.ndarray, seq_length: int = 6):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i: i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def build_lstm(seq_length: int = 6) -> Sequential:
    """
    Smaller LSTM architecture — suitable for small (36-point) monthly dataset.
    Fewer parameters = less overfitting.
    """
    model = Sequential([
        LSTM(32, input_shape=(seq_length, 1), return_sequences=False),
        Dropout(0.1),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def train_lstm(series: pd.Series, seq_length: int = 6,
               epochs: int = 100, batch_size: int = 4):
    """
    Train LSTM on monthly sales.
    Uses EarlyStopping to prevent overfitting on small dataset.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = create_sequences(scaled, seq_length)
    X    = X.reshape((X.shape[0], X.shape[1], 1))

    split   = max(1, int(0.8 * len(X)))   # at least 1 sample in test
    X_train = X[:split];  X_test = X[split:]
    y_train = y[:split];  y_test = y[split:]

    model = build_lstm(seq_length)

    # EarlyStopping stops training when val_loss stops improving
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=0
    )

    print(f"   Training on {len(X_train)} sequences, validating on {len(X_test)}...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    return model, scaler, X_test, y_test


def forecast_lstm(model, scaler, last_sequence: np.ndarray,
                  steps: int = 6) -> np.ndarray:
    """Auto-regressive future forecasting with LSTM."""
    preds = []
    seq   = last_sequence.copy()
    for _ in range(steps):
        p = model.predict(seq.reshape(1, -1, 1), verbose=0)[0][0]
        preds.append(p)
        seq = np.append(seq[1:], p)
    return scaler.inverse_transform(
        np.array(preds).reshape(-1, 1)
    ).flatten()