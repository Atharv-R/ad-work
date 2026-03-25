# src/adwork/models/forecaster.py

"""
Demand Forecaster
=================
Time-series forecasting for advertising campaign metrics.

Primary engine: Facebook Prophet
- Weekly seasonality detection
- US holiday effects  
- Google Trends as external regressor
- Backtesting with proper train/test split

Fallback engine: Holt-Winters Exponential Smoothing (statsmodels)
- Used when Prophet is not installed
- Supports weekly seasonality
- No external regressors or holidays

Design decision: The system auto-detects which engine is available
and uses the best one. This graceful degradation is intentional —
the forecaster always works, even in minimal environments.

Usage:
    from adwork.models.forecaster import DemandForecaster
    
    forecaster = DemandForecaster()
    results = forecaster.run(
        historical_df=daily_metrics,
        metric="clicks",
        campaign_id="ggl_brand_001",
    )
    # results has: forecast, backtest, evaluation, components
"""

# Suppress Prophet's verbose logging
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# Check what's available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
    logger.info("Prophet available — using full forecasting engine")
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not installed — falling back to ExponentialSmoothing")


class ForecastConfig(BaseModel):
    """Configuration for the forecasting model."""
    horizon_days: int = Field(default=14, description="Days ahead to forecast")
    test_days: int = Field(default=15, description="Days held out for backtesting")
    interval_width: float = Field(default=0.80, description="Confidence interval width")
    weekly_seasonality: bool = True
    yearly_seasonality: bool = False  # Need 2+ years of data
    include_holidays: bool = True
    holiday_country: str = "US"
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    use_trends_regressor: bool = True


class DemandForecaster:
    """
    Forecaster for ad campaign demand metrics.
    
    Automatically uses Prophet if available, otherwise falls back
    to Holt-Winters exponential smoothing from statsmodels.
    """

    def __init__(self, config: ForecastConfig | None = None):
        self.config = config or ForecastConfig()
        self.engine = "prophet" if PROPHET_AVAILABLE else "statsmodels"

    def run(
        self,
        historical_df: pd.DataFrame,
        metric: str = "clicks",
        campaign_id: str = "",
        trends_df: pd.DataFrame | None = None,
    ) -> dict:
        """
        Full forecasting pipeline: backtest → train on all data → forecast.
        
        Args:
            historical_df: Daily metrics (needs 'date' column + metric column)
            metric: Which column to forecast ('clicks','conversions','spend','revenue')
            campaign_id: Identifier for logging and storage
            trends_df: Optional Google Trends data (columns: date, interest)
            
        Returns:
            Dict with keys:
            - engine: str ('prophet' or 'statsmodels')
            - forecast: DataFrame [date, actual, predicted, lower, upper, is_forecast]
            - backtest: DataFrame [date, actual, predicted, lower, upper]
            - evaluation: dict {mape, mae, rmse, coverage}
            - components: dict with seasonality info
        """
        df = historical_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found. Available: {list(df.columns)}")
        if len(df) < 30:
            raise ValueError(f"Need at least 30 days of data, got {len(df)}")

        logger.info(f"Forecasting {metric} for {campaign_id} | {len(df)} days | engine={self.engine}")

        if self.engine == "prophet":
            return self._run_prophet(df, metric, campaign_id, trends_df)
        else:
            return self._run_statsmodels(df, metric, campaign_id)

    # ──────────────────────────────────────────────
    # PROPHET ENGINE
    # ──────────────────────────────────────────────

    def _run_prophet(self, df, metric, campaign_id, trends_df):
        """Full pipeline using Prophet."""

        test_days = min(self.config.test_days, len(df) // 4)
        train_df = df.iloc[:-test_days]
        test_df = df.iloc[-test_days:]

        # ── Backtest ──
        bt_pdf = self._prophet_prepare(train_df, metric, trends_df)
        has_trends = "search_trend" in bt_pdf.columns
        bt_model = self._prophet_build(has_trends)
        bt_model.fit(bt_pdf)

        bt_future = bt_model.make_future_dataframe(periods=test_days)
        if has_trends:
            bt_future = self._prophet_add_trends(bt_future, bt_pdf, trends_df)
        bt_fc = bt_model.predict(bt_future)

        actual_vals = test_df[metric].values.astype(float)
        pred_vals = bt_fc.iloc[-test_days:]["yhat"].values
        lower_vals = bt_fc.iloc[-test_days:]["yhat_lower"].values
        upper_vals = bt_fc.iloc[-test_days:]["yhat_upper"].values

        evaluation = self._evaluate(actual_vals, pred_vals, lower_vals, upper_vals)

        backtest = pd.DataFrame({
            "date": test_df["date"].values,
            "actual": actual_vals,
            "predicted": pred_vals,
            "lower": lower_vals,
            "upper": upper_vals,
        })

        # ── Full train + forward forecast ──
        full_pdf = self._prophet_prepare(df, metric, trends_df)
        has_trends_full = "search_trend" in full_pdf.columns
        model = self._prophet_build(has_trends_full)
        model.fit(full_pdf)

        future = model.make_future_dataframe(periods=self.config.horizon_days)
        if has_trends_full:
            future = self._prophet_add_trends(future, full_pdf, trends_df)
        fc = model.predict(future)

        max_date = df["date"].max()
        actuals_map = dict(zip(df["date"], df[metric].astype(float)))

        forecast = pd.DataFrame({
            "date": fc["ds"],
            "predicted": fc["yhat"],
            "lower": fc["yhat_lower"],
            "upper": fc["yhat_upper"],
        })
        forecast["actual"] = forecast["date"].map(lambda d: actuals_map.get(d, None))
        forecast["is_forecast"] = forecast["date"] > max_date

        # ── Seasonality components ──
        components = self._prophet_components(fc)

        logger.info(
            f"Prophet forecast done for {campaign_id}/{metric}: "
            f"MAPE={evaluation['mape']:.1f}%, Coverage={evaluation['coverage']:.0%}"
        )

        return {
            "engine": "prophet",
            "campaign_id": campaign_id,
            "metric": metric,
            "forecast": forecast,
            "backtest": backtest,
            "evaluation": evaluation,
            "components": components,
        }

    def _prophet_prepare(self, df, metric, trends_df):
        """Convert to Prophet format with optional trends regressor."""
        pdf = pd.DataFrame({
            "ds": df["date"],
            "y": df[metric].astype(float).clip(lower=0),
        })

        if (
            trends_df is not None
            and not trends_df.empty
            and self.config.use_trends_regressor
        ):
            trends = trends_df.copy()
            trends["ds"] = pd.to_datetime(trends["date"])

            if "keyword" in trends.columns:
                trends = trends.groupby("ds")["interest"].mean().reset_index()

            pdf = pdf.merge(
                trends[["ds", "interest"]].rename(columns={"interest": "search_trend"}),
                on="ds",
                how="left",
            )
            pdf["search_trend"] = pdf["search_trend"].ffill().bfill()

            if pdf["search_trend"].isna().all():
                pdf = pdf.drop(columns=["search_trend"])

        return pdf

    def _prophet_build(self, has_trends):
        """Create a configured Prophet model."""
        model = Prophet(
            yearly_seasonality=self.config.yearly_seasonality,
            weekly_seasonality=self.config.weekly_seasonality,
            daily_seasonality=False,
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            seasonality_prior_scale=self.config.seasonality_prior_scale,
            interval_width=self.config.interval_width,
        )
        if self.config.include_holidays:
            model.add_country_holidays(country_name=self.config.holiday_country)
        if has_trends:
            model.add_regressor("search_trend")
        return model

    def _prophet_add_trends(self, future, train_pdf, trends_df):
        """Fill search_trend for future dates."""
        future = future.merge(
            train_pdf[["ds", "search_trend"]],
            on="ds",
            how="left",
        )
        last_val = train_pdf["search_trend"].iloc[-1]
        future["search_trend"] = future["search_trend"].fillna(last_val)
        return future

    def _prophet_components(self, fc):
        """Extract seasonality components from Prophet forecast."""
        components = {}

        if "weekly" in fc.columns:
            fc_copy = fc[["ds", "weekly"]].copy()
            fc_copy["dow"] = fc_copy["ds"].dt.day_name()
            weekly_avg = fc_copy.groupby("dow")["weekly"].mean()

            day_order = [
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday",
            ]
            components["weekly"] = {
                day: round(float(weekly_avg.get(day, 0)), 2)
                for day in day_order
            }

        if "trend" in fc.columns:
            components["trend"] = {
                "start": round(float(fc["trend"].iloc[0]), 2),
                "end": round(float(fc["trend"].iloc[-1]), 2),
                "direction": "up" if fc["trend"].iloc[-1] > fc["trend"].iloc[0] else "down",
            }

        return components

    # ──────────────────────────────────────────────
    # STATSMODELS FALLBACK ENGINE
    # ──────────────────────────────────────────────

    def _run_statsmodels(self, df, metric, campaign_id):
        """Fallback pipeline using Holt-Winters exponential smoothing."""
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        test_days = min(self.config.test_days, len(df) // 4)
        train_series = df[metric].astype(float).iloc[:-test_days].values
        test_series = df[metric].astype(float).iloc[-test_days:].values

        # Clamp to avoid zero/negative issues with multiplicative models
        train_clamped = np.clip(train_series, 1, None)

        # ── Backtest ──
        bt_model = ExponentialSmoothing(
            train_clamped,
            seasonal_periods=7,
            trend="add",
            seasonal="add",
        ).fit(optimized=True)

        bt_pred = bt_model.forecast(test_days)

        # Approximate confidence intervals using residual std
        residuals = train_clamped - bt_model.fittedvalues
        std = np.std(residuals)
        z = 1.28  # ~80% CI
        bt_lower = bt_pred - z * std
        bt_upper = bt_pred + z * std

        evaluation = self._evaluate(test_series, bt_pred, bt_lower, bt_upper)

        backtest = pd.DataFrame({
            "date": df["date"].iloc[-test_days:].values,
            "actual": test_series,
            "predicted": bt_pred,
            "lower": bt_lower,
            "upper": bt_upper,
        })

        # ── Full train + forward forecast ──
        full_series = np.clip(df[metric].astype(float).values, 1, None)
        full_model = ExponentialSmoothing(
            full_series,
            seasonal_periods=7,
            trend="add",
            seasonal="add",
        ).fit(optimized=True)

        fwd = full_model.forecast(self.config.horizon_days)
        fitted = full_model.fittedvalues

        resid_full = full_series - fitted
        std_full = np.std(resid_full)

        # Build forecast DataFrame
        max_date = df["date"].max()
        future_dates = [max_date + timedelta(days=i + 1) for i in range(self.config.horizon_days)]

        hist_part = pd.DataFrame({
            "date": df["date"],
            "predicted": fitted,
            "lower": fitted - z * std_full,
            "upper": fitted + z * std_full,
            "actual": df[metric].astype(float),
            "is_forecast": False,
        })

        future_part = pd.DataFrame({
            "date": future_dates,
            "predicted": fwd,
            "lower": fwd - z * std_full,
            "upper": fwd + z * std_full,
            "actual": None,
            "is_forecast": True,
        })

        forecast = pd.concat([hist_part, future_part], ignore_index=True)

        # Weekly component from seasonal decomposition
        seasonal_vals = full_model.params.get("seasonal", None)
        components = {}
        if seasonal_vals is not None and len(seasonal_vals) >= 7:
            day_order = [
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday",
            ]
            start_dow = df["date"].iloc[0].weekday()
            components["weekly"] = {
                day_order[(start_dow + i) % 7]: round(float(seasonal_vals[i]), 2)
                for i in range(7)
            }

        logger.info(
            f"Statsmodels forecast done for {campaign_id}/{metric}: "
            f"MAPE={evaluation['mape']:.1f}%"
        )

        return {
            "engine": "statsmodels",
            "campaign_id": campaign_id,
            "metric": metric,
            "forecast": forecast,
            "backtest": backtest,
            "evaluation": evaluation,
            "components": components,
        }

    # ──────────────────────────────────────────────
    # EVALUATION
    # ──────────────────────────────────────────────

    @staticmethod
    def _evaluate(actual, predicted, lower, upper):
        """Compute forecast accuracy metrics."""
        actual = np.array(actual, dtype=float)
        predicted = np.array(predicted, dtype=float)
        lower = np.array(lower, dtype=float)
        upper = np.array(upper, dtype=float)

        nonzero = actual != 0
        mape = (
            np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100
            if nonzero.any() else 0.0
        )
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        coverage = np.mean((actual >= lower) & (actual <= upper))

        return {
            "mape": round(float(mape), 2),
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "coverage": round(float(coverage), 4),
        }