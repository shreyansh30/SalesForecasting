"""Run EDA for BrandX India Sales dataset."""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"

import warnings
warnings.filterwarnings("ignore")

from src.data_loader import load_brandx, aggregate_brandx
from src.eda import run_full_eda

def main():
    print("=" * 50)
    print("  BrandX India -- Exploratory Data Analysis")
    print("=" * 50)

    raw_df = load_brandx("data/brandx")
    df     = aggregate_brandx(raw_df, freq="MS")
    run_full_eda(raw_df, df)

if __name__ == "__main__":
    main()