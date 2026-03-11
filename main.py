"""
Sales Forecasting — BrandX India Store Dataset
Models: ARIMA, Holt-Winters, Prophet (with fallback), LSTM
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"       # kills TF oneDNN messages
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"       # kills all TF info messages

import warnings
warnings.filterwarnings("ignore")               # kills all Python warnings

# rest of your imports below...
import matplotlib
matplotlib.use("Agg")
# ... etc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from src.data_loader import (
    load_brandx, aggregate_brandx,
    get_city_list, train_test_split_ts
)
from src.feature_engineering import (
    add_time_features, add_lag_features, add_rolling_features
)
from src.model import (
    train_arima,        forecast_arima,
    train_prophet,      forecast_prophet,
    train_holtwinters,  forecast_holtwinters,
    train_lstm,         forecast_lstm
)
from src.evaluate import evaluate_model

# ─── CONFIG ──────────────────────────────────────────────
DATA_DIR       = "data/brandx"
CITY           = None        # e.g. "Mumbai" or None = all cities
STORE_ID       = None
FREQ           = "MS"         # Monthly
FORECAST_STEPS = 6           # 6 months ahead
ARIMA_ORDER    = (2, 1, 1)   # Fixed: better for monthly data
LSTM_SEQ_LEN   = 6           # 6-month lookback
LSTM_EPOCHS    = 100         # More epochs for small dataset
# ─────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  BrandX India -- Sales Forecasting")   # removed emoji (font fix)
    print("=" * 60)

    # ── 1. Load Data ─────────────────────────────────────
    raw_df = load_brandx(DATA_DIR)
    get_city_list(raw_df)
    df     = aggregate_brandx(raw_df, city=CITY, store_id=STORE_ID, freq=FREQ)
    print(f"\nSample time series:\n{df.head(10)}")

    # ── 2. Feature Engineering ────────────────────────────
    df_feat = add_time_features(df)
    df_feat = add_lag_features(df_feat,     lags=[1, 2, 3, 6])
    df_feat = add_rolling_features(df_feat, windows=[3, 6])
    df_feat.dropna(inplace=True)

    # ── 3. Train/Test Split ───────────────────────────────
    train, test = train_test_split_ts(df, test_ratio=0.2)
    all_metrics = []

    # ── 4. ARIMA ──────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"[1/4] Training ARIMA{ARIMA_ORDER}...")
    arima_model = train_arima(train["sales"], order=ARIMA_ORDER)
    arima_preds = forecast_arima(arima_model, steps=len(test))
    all_metrics.append(
        evaluate_model(test["sales"].values, arima_preds.values, "ARIMA")
    )

    # ── 5. Holt-Winters ───────────────────────────────────
    print(f"\n{'─'*60}")
    print("[2/4] Training Holt-Winters...")
    hw_model = train_holtwinters(train["sales"])
    hw_preds = forecast_holtwinters(hw_model, steps=len(test))
    all_metrics.append(
        evaluate_model(test["sales"].values, hw_preds.values, "Holt-Winters")
    )

    # ── 6. Prophet ────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("[3/4] Training Prophet...")
    prophet_model  = train_prophet(df)

    # Fix: request EXACTLY len(test) future periods beyond training
    prophet_fc     = forecast_prophet(
        prophet_model,
        periods=FORECAST_STEPS,
        df=df
    )
    # Match prophet predictions to test index dates
    prophet_test   = prophet_fc[
        prophet_fc["ds"].dt.to_period("M").isin(
            test.index.to_period("M")
        )
    ]
    if len(prophet_test) == len(test):
        all_metrics.append(
            evaluate_model(
                test["sales"].values,
                prophet_test["yhat"].values,
                "Prophet/HW"
            )
        )
    else:
        print(f"   INFO: Prophet forecast has {len(prophet_test)} points vs {len(test)} test points.")
        print("   Using Holt-Winters for this slot instead...")
        full_hw2   = train_holtwinters(train["sales"])
        hw2_preds  = forecast_holtwinters(full_hw2, steps=len(test))
        all_metrics.append(
            evaluate_model(test["sales"].values, hw2_preds.values, "Holt-Winters 2")
        )

    # ── 7. LSTM ───────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("[4/4] Training LSTM...")
    lstm_model, scaler, X_test, y_test = train_lstm(
        df["sales"],
        seq_length=LSTM_SEQ_LEN,
        epochs=LSTM_EPOCHS
    )
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    lstm_preds = scaler.inverse_transform(
        lstm_model.predict(X_test)
    ).flatten()
    all_metrics.append(
        evaluate_model(y_test_inv, lstm_preds, "LSTM")
    )

    # ── 8. Future Forecast ────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Generating {FORECAST_STEPS}-month future forecast...")

    # ARIMA future
    full_arima   = train_arima(df["sales"], order=ARIMA_ORDER)
    future_arima = forecast_arima(full_arima, steps=FORECAST_STEPS)

    # Holt-Winters future
    full_hw      = train_holtwinters(df["sales"])
    future_hw    = forecast_holtwinters(full_hw, steps=FORECAST_STEPS)

    # LSTM future
    last_seq    = scaler.transform(
        df["sales"].values[-LSTM_SEQ_LEN:].reshape(-1, 1)
    ).flatten()
    future_lstm = forecast_lstm(lstm_model, scaler, last_seq, steps=FORECAST_STEPS)

    # ── 9. Summary Table ──────────────────────────────────
    best = min(all_metrics, key=lambda x: x["MAPE (%)"])
    print(f"\n{'='*60}")
    print(f"  Model Comparison Summary")
    print(f"{'='*60}")
    print(f"{'Model':<16} {'MAE':>15} {'RMSE':>15} {'MAPE%':>8}")
    print(f"{'─'*60}")
    for m in all_metrics:
        flag = " <<< Best" if m["Model"] == best["Model"] else ""
        print(
            f"{m['Model']:<16} "
            f"Rs.{m['MAE']/1e7:>8.2f}Cr  "
            f"Rs.{m['RMSE']/1e7:>8.2f}Cr  "
            f"{m['MAPE (%)']:>7.2f}%{flag}"
        )
    print(f"\nBest Model: {best['Model']} (MAPE: {best['MAPE (%)']}%)")

    # ── 10. Plot ──────────────────────────────────────────
    city_label = CITY if CITY else "All Cities"
    fig, axes  = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle(
        f"BrandX India Sales Forecast | {city_label}",  # no emoji
        fontsize=14, fontweight="bold"
    )

    def crore_fmt(x, _):
        return f"Rs.{x/1e7:.1f}Cr"

    # Chart 1 — Actual vs Models on test set
    axes[0].plot(train.index, train["sales"],
                 label="Train", color="steelblue", alpha=0.6, linewidth=2)
    axes[0].plot(test.index, test["sales"],
                 label="Actual", color="green", linewidth=2.5, marker="o")
    axes[0].plot(test.index, arima_preds.values,
                 label=f"ARIMA (MAPE: {all_metrics[0]['MAPE (%)']:.1f}%)",
                 linestyle="--", color="red", linewidth=2)
    axes[0].plot(test.index, hw_preds.values,
                 label=f"Holt-Winters (MAPE: {all_metrics[1]['MAPE (%)']:.1f}%)",
                 linestyle="-.", color="orange", linewidth=2)
    axes[0].set_title("Actual vs Forecast — Test Period (Last 6 Months)")
    axes[0].set_ylabel("Revenue")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(crore_fmt))

    # Chart 2 — Future forecast
    axes[1].plot(df.index, df["sales"],
                 label="Historical Sales", color="steelblue", linewidth=2)
    axes[1].plot(future_arima.index, future_arima.values,
                 label="ARIMA Forecast", linestyle="--", color="red", linewidth=2)
    axes[1].plot(future_hw.index, future_hw.values,
                 label="Holt-Winters Forecast", linestyle="-.", color="orange", linewidth=2)
    axes[1].plot(future_arima.index, future_lstm,
                 label="LSTM Forecast", linestyle=":", color="purple", linewidth=2)
    axes[1].fill_between(
        future_arima.index,
        np.array(future_arima.values) * 0.92,
        np.array(future_arima.values) * 1.08,
        alpha=0.15, color="red", label="ARIMA +/-8% Band"
    )
    axes[1].axvline(x=df.index[-1], color="gray",
                    linestyle="--", alpha=0.7, label="Forecast Start")
    axes[1].set_title(f"Future {FORECAST_STEPS}-Month Sales Forecast")
    axes[1].set_ylabel("Revenue")
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(crore_fmt))

    plt.tight_layout()
    out = "brandx_india_forecast.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved as '{out}'")
    print("\nDone!")


if __name__ == "__main__":
    main()