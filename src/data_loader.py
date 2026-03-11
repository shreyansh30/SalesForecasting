"""
Data loader for BrandX India Store Dataset.
Fixed for pandas 2.2+ (M -> ME, Q -> QE, Y -> YE)
"""

import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler


# ─── Pandas 2.2+ frequency mapping ───────────────────────
FREQ_MAP = {
    "M": "ME",    # Month End (was "M")
    "Q": "QE",    # Quarter End (was "Q")
    "Y": "YE",    # Year End (was "Y")
    "D": "D",     # Daily (unchanged)
    "W": "W",     # Weekly (unchanged)
    "MS": "MS",   # Month Start (unchanged)
    "QS": "QS",   # Quarter Start (unchanged)
    "YS": "YS",   # Year Start (unchanged)
}


def load_brandx(data_dir: str = "data/brandx") -> pd.DataFrame:
    """Load BrandX India Store Dataset from CSV."""
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in '{data_dir}'.\n"
            "Download from: https://www.kaggle.com/datasets/laxdippatel/brandx-india-store-dataset"
        )

    filepath = os.path.join(data_dir, csv_files[0])
    print(f"📂 Loading: {filepath}")
    df = pd.read_csv(filepath, low_memory=False)

    print(f"\n📋 Columns available: {list(df.columns)}")

    # ── Build date from Month + Year ─────────────────────
    if "Month" in df.columns and "Year" in df.columns:
        print("   📅 Building date from 'Month' + 'Year' columns...")

        month_map = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12,
            "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
            "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9,
            "Oct": 10, "Nov": 11, "Dec": 12
        }

        if df["Month"].dtype == object:
            df["month_num"] = df["Month"].map(month_map).fillna(
                pd.to_numeric(df["Month"], errors="coerce")
            ).astype(int)
        else:
            df["month_num"] = df["Month"].astype(int)

        df["date"] = pd.to_datetime(
            df["Year"].astype(str) + "-" + df["month_num"].astype(str) + "-01",
            format="%Y-%m-%d"
        )

    elif "Quarter" in df.columns and "Year" in df.columns:
        print("   📅 Building date from 'Quarter' + 'Year' columns...")
        quarter_to_month = {1: 1, 2: 4, 3: 7, 4: 10,
                            "Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}
        df["month_num"] = df["Quarter"].map(quarter_to_month)
        df["date"] = pd.to_datetime(
            df["Year"].astype(str) + "-" + df["month_num"].astype(str) + "-01"
        )

    else:
        date_col = _detect_date_column(df)
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        df = df.rename(columns={date_col: "date"})

    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"\n✅ BrandX dataset loaded successfully!")
    print(f"   Rows        : {df.shape[0]:,}")
    print(f"   Columns     : {df.shape[1]}")
    print(f"   Date range  : {df['date'].min().date()} → {df['date'].max().date()}")

    if "City" in df.columns:
        print(f"   Cities      : {df['City'].unique().tolist()}")
    if "Store_ID" in df.columns:
        print(f"   Stores      : {df['Store_ID'].nunique()}")

    return df


def aggregate_brandx(
    df: pd.DataFrame,
    city: str = None,
    store_id: str = None,
    freq: str = "MS"       # ← Changed default to "MS" (Month Start, always works)
) -> pd.DataFrame:
    """
    Aggregate BrandX data to a time series of total Revenue.

    Args:
        freq: Use "MS" (monthly), "QS" (quarterly), "YS" (yearly), "W" (weekly)
              These are all pandas 2.2+ compatible.
    """
    df = df.copy()

    # ── Remap old-style freq to pandas 2.2+ compatible ───
    freq = FREQ_MAP.get(freq, freq)

    if city:
        if "City" not in df.columns:
            raise ValueError("'City' column not found in dataset.")
        df = df[df["City"] == city]
        print(f"   🏙️  Filtered to city  : {city}  ({len(df):,} rows)")

    if store_id:
        df = df[df["Store_ID"] == store_id]
        print(f"   🏪 Filtered to store : {store_id}  ({len(df):,} rows)")

    # Always use Revenue
    sales_col = "Revenue" if "Revenue" in df.columns else _detect_sales_column(df)
    print(f"   💰 Sales column used : '{sales_col}'")

    freq_label = {
        "ME": "Monthly", "MS": "Monthly",
        "QE": "Quarterly", "QS": "Quarterly",
        "YE": "Yearly", "YS": "Yearly",
        "D": "Daily", "W": "Weekly"
    }.get(freq, freq)

    grouped = df.groupby(pd.Grouper(key="date", freq=freq))[sales_col].sum()
    grouped = grouped[grouped > 0]

    result = grouped.reset_index()
    result.columns = ["date", "sales"]
    result.set_index("date", inplace=True)

    print(f"   📅 Aggregation       : {freq_label}")
    print(f"   📊 Time series length: {len(result)} periods")
    print(f"\nTime series preview:\n{result.head(8)}")

    return result


def get_city_list(df: pd.DataFrame) -> list:
    """Return and print list of unique cities."""
    if "City" not in df.columns:
        raise ValueError("'City' column not found in dataset.")
    cities = sorted(df["City"].dropna().unique().tolist())
    print(f"\n🏙️  Available cities: {cities}")
    return cities


def train_test_split_ts(df: pd.DataFrame, test_ratio: float = 0.2):
    """Split time series into train/test without shuffling."""
    if len(df) < 5:
        raise ValueError(
            f"Not enough data points ({len(df)}) to split.\n"
            "Try freq='MS' (monthly) instead."
        )
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]
    print(f"\n✂️  Train/Test Split:")
    print(f"   Train : {train.index.min().date()} → {train.index.max().date()} ({len(train)} records)")
    print(f"   Test  : {test.index.min().date()} → {test.index.max().date()} ({len(test)} records)")
    return train, test


def scale_series(series: pd.Series):
    """Scale a pandas Series using MinMaxScaler."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    return scaled, scaler


# ─── PRIVATE HELPERS ─────────────────────────────────────

def _detect_date_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            return col
    raise ValueError(f"No date column found. Columns: {list(df.columns)}")


def _detect_sales_column(df: pd.DataFrame) -> str:
    priority = ["Revenue", "Sales", "Units_Sold", "Unit_Price"]
    for col in priority:
        if col in df.columns:
            return col
    keywords = ["sale", "revenue", "amount", "total", "income"]
    for col in df.columns:
        if any(k in col.lower() for k in keywords):
            return col
    raise ValueError(f"No sales column found. Columns: {list(df.columns)}")