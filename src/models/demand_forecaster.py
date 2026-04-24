"""Demand forecasting model using Gradient Boosted Trees."""
from typing import Dict

from pyspark.sql import DataFrame
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from .base import BaseModel
from ..features.engineer import NYISOFeatureEngineer


class DemandForecaster(BaseModel):
    """Forecast electricity demand (Load) using GBT Regression."""

    def __init__(self, feature_cols: list = None):
        """Initialize demand forecaster.

        Args:
            feature_cols: Feature columns. If None, uses default from feature engineer.
        """
        if feature_cols is None:
            engineer = NYISOFeatureEngineer()
            feature_cols = engineer.get_feature_columns(target="demand")

        super().__init__(
            name="demand_forecaster",
            feature_cols=feature_cols,
            target_col="Load_MW"
        )

    def _create_model_stage(self) -> GBTRegressor:
        """Create GBT Regressor for demand forecasting."""
        config = self.model_config.get("demand_forecast", {})

        return GBTRegressor(
            labelCol=self.target_col,
            featuresCol="features",
            predictionCol="predicted_demand",
            maxDepth=config.get("max_depth", 8),
            maxIter=config.get("max_iter", 100),
            stepSize=config.get("step_size", 0.1),
            seed=42
        )

    def evaluate(self, predictions_df: DataFrame) -> Dict[str, float]:
        """Evaluate demand forecast performance.

        Args:
            predictions_df: DataFrame with predictions

        Returns:
            Dictionary of metrics (RMSE, MAE, R2, MAPE)
        """
        evaluators = {
            "rmse": RegressionEvaluator(
                labelCol=self.target_col,
                predictionCol="predicted_demand",
                metricName="rmse"
            ),
            "mae": RegressionEvaluator(
                labelCol=self.target_col,
                predictionCol="predicted_demand",
                metricName="mae"
            ),
            "r2": RegressionEvaluator(
                labelCol=self.target_col,
                predictionCol="predicted_demand",
                metricName="r2"
            )
        }

        metrics = {name: eval.evaluate(predictions_df) for name, eval in evaluators.items()}

        # Calculate MAPE
        from pyspark.sql.functions import col, abs as spark_abs, avg, when

        mape_df = predictions_df.withColumn(
            "ape",
            when(col(self.target_col) != 0,
                 spark_abs((col(self.target_col) - col("predicted_demand")) / col(self.target_col)) * 100)
            .otherwise(0)
        )
        mape = mape_df.select(avg("ape")).collect()[0][0]
        metrics["mape"] = mape if mape else 0.0

        return metrics

    def forecast_horizon(self, df: DataFrame, hours_ahead: int = 24) -> DataFrame:
        """Generate multi-step ahead forecasts.

        Note: This is a simplified approach. For production, you'd want
        a proper recursive or direct multi-step forecasting strategy.

        Args:
            df: DataFrame with features
            hours_ahead: Number of hours to forecast

        Returns:
            DataFrame with forecast horizon
        """
        # For now, just return single-step predictions
        # A full implementation would recursively generate features
        return self.predict(df)
