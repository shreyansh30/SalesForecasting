"""Unit tests for BrandX India Sales Forecasting pipeline."""
import unittest
import numpy as np
import pandas as pd
from src.feature_engineering import add_lag_features, add_rolling_features, add_time_features
from src.evaluate import mape, evaluate_model


class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        dates = pd.date_range("2022-01-01", periods=90, freq="D")
        self.df = pd.DataFrame({"sales": np.random.rand(90) * 10000}, index=dates)

    def test_lag_features_columns(self):
        result = add_lag_features(self.df, lags=[1, 7, 30])
        self.assertIn("lag_1",  result.columns)
        self.assertIn("lag_7",  result.columns)
        self.assertIn("lag_30", result.columns)

    def test_rolling_features_columns(self):
        result = add_rolling_features(self.df, windows=[7, 14])
        self.assertIn("rolling_mean_7",  result.columns)
        self.assertIn("rolling_std_14",  result.columns)

    def test_time_features_india(self):
        result = add_time_features(self.df)
        self.assertIn("is_festive_month",  result.columns)
        self.assertIn("india_fy_quarter",  result.columns)
        self.assertIn("is_weekend",        result.columns)


class TestEvaluation(unittest.TestCase):

    def test_mape_zero_error(self):
        y = np.array([1000.0, 2000.0, 3000.0])
        self.assertAlmostEqual(mape(y, y), 0.0)

    def test_mape_known_value(self):
        y_true = np.array([100.0, 200.0, 400.0])
        y_pred = np.array([110.0, 180.0, 420.0])
        result = mape(y_true, y_pred)
        self.assertGreater(result, 0)
        self.assertLess(result, 20)

    def test_evaluate_returns_dict(self):
        y = np.array([500.0, 600.0, 700.0])
        result = evaluate_model(y, y + 10, "TestModel")
        self.assertIn("MAE",       result)
        self.assertIn("RMSE",      result)
        self.assertIn("MAPE (%)",  result)


if __name__ == "__main__":
    unittest.main()