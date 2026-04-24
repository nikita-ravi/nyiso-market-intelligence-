"""Model evaluation and comparison framework."""
from typing import Dict, List, Any
from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, hour, avg

from ..utils.config import get_project_root


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    model_name: str
    metrics: Dict[str, float]
    timestamp: str
    data_info: Dict[str, Any]

    def to_dict(self) -> Dict:
        return {
            "model_name": self.model_name,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "data_info": self.data_info
        }


class ModelEvaluator:
    """Evaluate and compare ML models."""

    def __init__(self):
        """Initialize evaluator."""
        self.results_dir = get_project_root() / "results"
        self.results_dir.mkdir(exist_ok=True)
        self.evaluation_history: List[ModelMetrics] = []

    def evaluate_model(self, model, test_df: DataFrame, model_name: str = None) -> ModelMetrics:
        """Evaluate a single model.

        Args:
            model: Trained model with evaluate() method
            test_df: Test DataFrame
            model_name: Optional name override

        Returns:
            ModelMetrics object
        """
        predictions = model.predict(test_df)
        metrics = model.evaluate(predictions)

        result = ModelMetrics(
            model_name=model_name or model.name,
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
            data_info={
                "test_size": test_df.count(),
                "features": len(model.feature_cols)
            }
        )

        self.evaluation_history.append(result)
        return result

    def evaluate_by_hour(self, model, predictions_df: DataFrame,
                         target_col: str, pred_col: str) -> Dict[int, Dict[str, float]]:
        """Evaluate model performance by hour of day.

        Args:
            model: The model (for reference)
            predictions_df: DataFrame with predictions
            target_col: Actual value column
            pred_col: Predicted value column

        Returns:
            Dictionary mapping hour to metrics
        """
        from pyspark.sql.functions import abs as spark_abs, when

        hourly_metrics = {}

        for h in range(24):
            hour_df = predictions_df.filter(col("hour") == h)

            if hour_df.count() == 0:
                continue

            # Calculate metrics for this hour
            stats = hour_df.withColumn(
                "abs_error", spark_abs(col(target_col) - col(pred_col))
            ).withColumn(
                "pct_error",
                when(col(target_col) != 0,
                     spark_abs((col(target_col) - col(pred_col)) / col(target_col)) * 100)
                .otherwise(0)
            ).agg(
                avg("abs_error").alias("mae"),
                avg("pct_error").alias("mape")
            ).collect()[0]

            hourly_metrics[h] = {
                "mae": stats["mae"],
                "mape": stats["mape"],
                "count": hour_df.count()
            }

        return hourly_metrics

    def evaluate_by_zone(self, model, predictions_df: DataFrame,
                         target_col: str, pred_col: str) -> Dict[str, Dict[str, float]]:
        """Evaluate model performance by zone.

        Args:
            model: The model (for reference)
            predictions_df: DataFrame with predictions
            target_col: Actual value column
            pred_col: Predicted value column

        Returns:
            Dictionary mapping zone to metrics
        """
        from pyspark.sql.functions import abs as spark_abs, when

        zones = predictions_df.select("Name").distinct().collect()
        zone_metrics = {}

        for zone_row in zones:
            zone = zone_row["Name"]
            zone_df = predictions_df.filter(col("Name") == zone)

            if zone_df.count() == 0:
                continue

            stats = zone_df.withColumn(
                "abs_error", spark_abs(col(target_col) - col(pred_col))
            ).withColumn(
                "pct_error",
                when(col(target_col) != 0,
                     spark_abs((col(target_col) - col(pred_col)) / col(target_col)) * 100)
                .otherwise(0)
            ).agg(
                avg("abs_error").alias("mae"),
                avg("pct_error").alias("mape")
            ).collect()[0]

            zone_metrics[zone] = {
                "mae": stats["mae"],
                "mape": stats["mape"],
                "count": zone_df.count()
            }

        return zone_metrics

    def compare_models(self, results: List[ModelMetrics], metric: str = "rmse") -> List[Dict]:
        """Compare multiple model results.

        Args:
            results: List of ModelMetrics
            metric: Metric to compare on

        Returns:
            Sorted list of comparisons
        """
        comparisons = []
        for r in results:
            if metric in r.metrics:
                comparisons.append({
                    "model": r.model_name,
                    metric: r.metrics[metric],
                    "timestamp": r.timestamp
                })

        return sorted(comparisons, key=lambda x: x[metric])

    def save_results(self, filename: str = None):
        """Save evaluation results to JSON.

        Args:
            filename: Optional filename. Defaults to results_{timestamp}.json
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"

        filepath = self.results_dir / filename

        results_data = [r.to_dict() for r in self.evaluation_history]

        with open(filepath, "w") as f:
            json.dump(results_data, f, indent=2)

        print(f"Results saved to {filepath}")

    def load_results(self, filename: str) -> List[ModelMetrics]:
        """Load evaluation results from JSON.

        Args:
            filename: Filename to load

        Returns:
            List of ModelMetrics
        """
        filepath = self.results_dir / filename

        with open(filepath, "r") as f:
            data = json.load(f)

        return [
            ModelMetrics(
                model_name=r["model_name"],
                metrics=r["metrics"],
                timestamp=r["timestamp"],
                data_info=r["data_info"]
            )
            for r in data
        ]

    def print_summary(self, result: ModelMetrics):
        """Print a formatted summary of model results.

        Args:
            result: ModelMetrics to display
        """
        print(f"\n{'='*50}")
        print(f"Model: {result.model_name}")
        print(f"Evaluated: {result.timestamp}")
        print(f"Test samples: {result.data_info.get('test_size', 'N/A')}")
        print("-" * 50)
        print("Metrics:")
        for metric, value in result.metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        print("=" * 50)
