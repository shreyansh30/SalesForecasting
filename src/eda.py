"""Exploratory Data Analysis for BrandX India Dataset."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_sales_trend(df: pd.DataFrame):
    """Plot overall monthly sales trend."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df["sales"], marker="o", color="steelblue", linewidth=2)
    ax.set_title("📈 BrandX India — Monthly Sales Trend (2022–2024)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (₹)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("eda_sales_trend.png", dpi=150)
    plt.show()
    print("✅ Saved: eda_sales_trend.png")


def plot_city_wise_sales(raw_df: pd.DataFrame):
    """Bar chart of total revenue by city."""
    city_sales = raw_df.groupby("City")["Revenue"].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    city_sales.plot(kind="bar", ax=ax, color="coral", edgecolor="black")
    ax.set_title("🏙️ Total Revenue by City")
    ax.set_xlabel("City")
    ax.set_ylabel("Total Revenue (₹)")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig("eda_city_sales.png", dpi=150)
    plt.show()
    print("✅ Saved: eda_city_sales.png")


def plot_category_sales(raw_df: pd.DataFrame):
    """Pie chart of sales by product category."""
    cat_sales = raw_df.groupby("Category")["Revenue"].sum()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(cat_sales, labels=cat_sales.index, autopct="%1.1f%%", startangle=140)
    ax.set_title("🛍️ Revenue Share by Category")
    plt.tight_layout()
    plt.savefig("eda_category_sales.png", dpi=150)
    plt.show()
    print("✅ Saved: eda_category_sales.png")


def plot_monthly_heatmap(raw_df: pd.DataFrame):
    """Heatmap of monthly sales by year."""
    raw_df = raw_df.copy()
    pivot = raw_df.groupby(["Year", "month_num"])["Revenue"].sum().unstack()
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
    ax.set_title("🔥 Monthly Revenue Heatmap by Year")
    plt.tight_layout()
    plt.savefig("eda_heatmap.png", dpi=150)
    plt.show()
    print("✅ Saved: eda_heatmap.png")


def plot_seasonal_decomposition(df: pd.DataFrame):
    """Decompose time series into trend, seasonality, residual."""
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomp = seasonal_decompose(df["sales"], model="additive", period=12)

    fig = decomp.plot()
    fig.set_size_inches(14, 8)
    fig.suptitle("📊 Seasonal Decomposition — BrandX Sales", fontsize=13)
    plt.tight_layout()
    plt.savefig("eda_decomposition.png", dpi=150)
    plt.show()
    print("✅ Saved: eda_decomposition.png")


def run_full_eda(raw_df: pd.DataFrame, ts_df: pd.DataFrame):
    """Run all EDA plots."""
    print("\n🔍 Running Full EDA...\n")
    plot_sales_trend(ts_df)
    plot_city_wise_sales(raw_df)
    plot_category_sales(raw_df)
    plot_seasonal_decomposition(ts_df)
    print("\n✅ EDA Complete! Check the saved PNG files.")