"""Interactive Streamlit Dashboard for BrandX India Sales Forecasting."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from src.data_loader import load_brandx, aggregate_brandx, get_city_list, train_test_split_ts
from src.model import train_arima, forecast_arima, train_prophet, forecast_prophet
from src.evaluate import evaluate_model

# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="🇮🇳 BrandX India — Sales Forecast",
    page_icon="📈",
    layout="wide"
)

st.title("🇮🇳 BrandX India — Sales Forecasting Dashboard")
st.markdown("Real-world Indian retail store sales forecasting using ARIMA & Prophet")

# ── Sidebar Controls ─��───────────────────────────────────
st.sidebar.header("⚙️ Configuration")

@st.cache_data
def load_data():
    return load_brandx("data/brandx")

raw_df = load_data()
cities = ["All Cities"] + get_city_list(raw_df)

selected_city  = st.sidebar.selectbox("🏙️ Select City", cities)
selected_model = st.sidebar.selectbox("🤖 Select Model", ["ARIMA", "Prophet", "Both"])
forecast_steps = st.sidebar.slider("🔮 Forecast Months", 3, 12, 6)

city_filter = None if selected_city == "All Cities" else selected_city
df = aggregate_brandx(raw_df, city=city_filter, freq="M")

# ── KPI Cards ────────────────────────────────────────────
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

col1.metric("📅 Total Months",    f"{len(df)}")
col2.metric("💰 Total Revenue",   f"₹{df['sales'].sum():,.0f}")
col3.metric("📈 Avg Monthly",     f"₹{df['sales'].mean():,.0f}")
col4.metric("🏆 Peak Month",      f"₹{df['sales'].max():,.0f}")

# ── Sales Trend Chart ─────────────────────────────────────
st.subheader("📈 Sales Trend")
fig_trend = px.line(df.reset_index(), x="date", y="sales",
                    title=f"Monthly Revenue — {selected_city}",
                    labels={"sales": "Revenue (₹)", "date": "Month"})
fig_trend.update_traces(line_color="steelblue", line_width=2)
st.plotly_chart(fig_trend, use_container_width=True)

# ── City Comparison ───────────────────────────────────────
st.subheader("🏙️ Revenue by City")
city_rev = raw_df.groupby("City")["Revenue"].sum().sort_values(ascending=False).reset_index()
fig_city = px.bar(city_rev, x="City", y="Revenue", color="Revenue",
                  color_continuous_scale="Blues",
                  title="Total Revenue by Indian City")
st.plotly_chart(fig_city, use_container_width=True)

# ── Forecast ─────────────────────────────────────────────
st.subheader(f"🔮 {forecast_steps}-Month Sales Forecast")
train, test = train_test_split_ts(df, test_ratio=0.2)

fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(
    x=df.index, y=df["sales"],
    name="Actual Sales", line=dict(color="steelblue", width=2)
))

if selected_model in ["ARIMA", "Both"]:
    with st.spinner("Training ARIMA..."):
        arima_m    = train_arima(train["sales"], order=(2, 1, 1))
        arima_test = forecast_arima(arima_m, steps=len(test))
        full_arima = train_arima(df["sales"], order=(2, 1, 1))
        future_fc  = forecast_arima(full_arima, steps=forecast_steps)

    fig_fc.add_trace(go.Scatter(
        x=future_fc.index, y=future_fc.values,
        name="ARIMA Forecast", line=dict(color="red", dash="dash", width=2)
    ))
    metrics = evaluate_model(test["sales"].values, arima_test.values, "ARIMA")
    st.info(f"**ARIMA** → MAE: ₹{metrics['MAE']:,.2f} | RMSE: ₹{metrics['RMSE']:,.2f} | MAPE: {metrics['MAPE (%)']:.2f}%")

if selected_model in ["Prophet", "Both"]:
    with st.spinner("Training Prophet..."):
        prophet_m  = train_prophet(df)
        prophet_fc = forecast_prophet(prophet_m, periods=forecast_steps)

    fig_fc.add_trace(go.Scatter(
        x=prophet_fc["ds"], y=prophet_fc["yhat"],
        name="Prophet Forecast", line=dict(color="purple", dash="dot", width=2)
    ))
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([prophet_fc["ds"], prophet_fc["ds"][::-1]]),
        y=pd.concat([prophet_fc["yhat_upper"], prophet_fc["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(128,0,128,0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Prophet Confidence Band"
    ))

fig_fc.update_layout(
    title=f"Sales Forecast — Next {forecast_steps} Months",
    xaxis_title="Month", yaxis_title="Revenue (₹)",
    legend=dict(orientation="h", y=-0.2), hovermode="x unified"
)
st.plotly_chart(fig_fc, use_container_width=True)

# ── Category Breakdown ────────────────────────────────────
if "Category" in raw_df.columns:
    st.subheader("🛍️ Revenue by Category")
    cat_df = raw_df.groupby("Category")["Revenue"].sum().reset_index()
    fig_cat = px.pie(cat_df, names="Category", values="Revenue",
                     title="Revenue Share by Product Category",
                     color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig_cat, use_container_width=True)

st.markdown("---")
st.caption("Built with ❤️ using Python | ARIMA · Prophet · Streamlit | BrandX India Dataset")