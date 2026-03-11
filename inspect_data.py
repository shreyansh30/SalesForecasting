"""Run this first to understand your BrandX dataset columns."""
import pandas as pd
import os

data_dir = "data/brandx"
csv_file = [f for f in os.listdir(data_dir) if f.endswith(".csv")][0]
df = pd.read_csv(f"{data_dir}/{csv_file}", low_memory=False)

print("=" * 50)
print(f"File     : {csv_file}")
print(f"Shape    : {df.shape[0]:,} rows × {df.shape[1]} columns")
print("=" * 50)
print("\n📋 All Columns:")
for col in df.columns:
    print(f"  - {col} : {df[col].dtype}  |  Sample: {df[col].iloc[0]}")

print("\n📅 Potential Date Columns:")
for col in df.columns:
    if "date" in col.lower() or "time" in col.lower():
        print(f"  ✅ {col}")

print("\n💰 Potential Sales Columns:")
for col in df.columns:
    if any(k in col.lower() for k in ["sale", "revenue", "amount", "total", "price"]):
        print(f"  ✅ {col}")

if "City" in df.columns:
    print(f"\n🏙️  Cities: {df['City'].unique().tolist()}")