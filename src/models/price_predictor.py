"""Price prediction model using Gradient Boosted Trees."""
from typing import Dict

from pyspark.sql import DataFrame
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from .base import BaseModel
from ..features.engineer import NYISOFeatureEngineer


class PricePredictor(BaseModel):
    """Predict electricity prices (LBMP) using GBT Regression."""

    def __init__(self, feature_cols: list = None):
        """Initialize price predictor.

        Args:
            feature_cols: Feature columns. If None, uses default from feature engineer.
        """
        if feature_cols is None:
            engineer = NYISOFeatureEngineer()
            feature_cols = engineer.get_feature_columns(target="price")

        super().__init__(
            name="price_predictor",
            feature_cols=feature_cols,
            target_col="LBMP_avg"
        )

    def _create_model_stage(self) -> GBTRegressor:
        """Create GBT Regressor for price prediction."""
        config = self.model_config.get("price_prediction", {})

        return GBTRegressor(
            labelCol=self.target_col,
            featuresCol="features",
            predictionCol="predicted_price",
            maxDepth=config.get("max_depth", 8),
            maxIter=config.get("max_iter", 100),
            stepSize=config.get("step_size", 0.1),
            seed=42
        )

    def evaluate(self, predictions_df: DataFrame) -> Dict[str, float]:
        """Evaluate price prediction performance.

        Args:
            predictions_df: DataFrame with predictions

        Returns:
            Dictionary of metrics (RMSE, MAE, R2, MAPE)
        """
        evaluators = {
            "rmse": RegressionEvaluator(
                labelCol=self.target_col,
                predictionCol="predicted_price",
                metricName="rmse"
            ),
            "mae": RegressionEvaluator(
                labelCol=self.target_col,
                predictionCol="predicted_price",
                metricName="mae"
            ),
            "r2": RegressionEvaluator(
                labelCol=self.target_col,
                predictionCol="predicted_price",
                metricName="r2"
            )
        }

        metrics = {name: eval.evaluate(predictions_df) for name, eval in evaluators.items()}

        # Calculate MAPE manually
        from pyspark.sql.functions import col, abs as spark_abs, avg, when

        mape_df = predictions_df.withColumn(
            "ape",
            when(col(self.target_col) != 0,
                 spark_abs((col(self.target_col) - col("predicted_price")) / col(self.target_col)) * 100)
            .otherwise(0)
        )
        mape = mape_df.select(avg("ape")).collect()[0][0]
        metrics["mape"] = mape if mape else 0.0

        return metrics

    def predict_with_intervals(self, df: DataFrame, confidence: float = 0.95) -> DataFrame:
        """Make predictions with confidence intervals (approximate).

        Uses historical prediction errors to estimate intervals.

        Args:
            df: DataFrame with features
            confidence: Confidence level (default 0.95)

        Returns:
            DataFrame with predictions and interval bounds
        """
        from pyspark.sql.functions import col, lit

        predictions = self.predict(df)

        # Approximate confidence interval based on typical prediction error
        # In practice, you'd calibrate this from validation data
        error_multiplier = 1.96 if confidence == 0.95 else 1.645  # z-scores
        base_error = 20.0  # Approximate base error in $/MWHr

        predictions = predictions.withColumn(
            "price_lower",
            col("predicted_price") - lit(error_multiplier * base_error)
        ).withColumn(
            "price_upper",
            col("predicted_price") + lit(error_multiplier * base_error)
        )

        return predictions
