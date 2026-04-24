"""Price spike detection model using Random Forest Classification."""
from typing import Dict

from pyspark.sql import DataFrame
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

from .base import BaseModel
from ..features.engineer import NYISOFeatureEngineer


class SpikeDetector(BaseModel):
    """Detect price spikes using Random Forest Classification."""

    def __init__(self, feature_cols: list = None):
        """Initialize spike detector.

        Args:
            feature_cols: Feature columns. If None, uses default from feature engineer.
        """
        if feature_cols is None:
            engineer = NYISOFeatureEngineer()
            feature_cols = engineer.get_feature_columns(target="spike")

        super().__init__(
            name="spike_detector",
            feature_cols=feature_cols,
            target_col="is_price_spike"
        )

    def _create_model_stage(self) -> RandomForestClassifier:
        """Create Random Forest Classifier for spike detection."""
        config = self.model_config.get("spike_detection", {})

        return RandomForestClassifier(
            labelCol=self.target_col,
            featuresCol="features",
            predictionCol="predicted_spike",
            probabilityCol="spike_probability",
            numTrees=config.get("num_trees", 100),
            maxDepth=config.get("max_depth", 10),
            seed=42
        )

    def evaluate(self, predictions_df: DataFrame) -> Dict[str, float]:
        """Evaluate spike detection performance.

        Args:
            predictions_df: DataFrame with predictions

        Returns:
            Dictionary of metrics (AUC, accuracy, precision, recall, F1)
        """
        metrics = {}

        # AUC-ROC
        auc_evaluator = BinaryClassificationEvaluator(
            labelCol=self.target_col,
            rawPredictionCol="spike_probability",
            metricName="areaUnderROC"
        )
        metrics["auc_roc"] = auc_evaluator.evaluate(predictions_df)

        # Accuracy
        acc_evaluator = MulticlassClassificationEvaluator(
            labelCol=self.target_col,
            predictionCol="predicted_spike",
            metricName="accuracy"
        )
        metrics["accuracy"] = acc_evaluator.evaluate(predictions_df)

        # Precision, Recall, F1
        precision_evaluator = MulticlassClassificationEvaluator(
            labelCol=self.target_col,
            predictionCol="predicted_spike",
            metricName="weightedPrecision"
        )
        metrics["precision"] = precision_evaluator.evaluate(predictions_df)

        recall_evaluator = MulticlassClassificationEvaluator(
            labelCol=self.target_col,
            predictionCol="predicted_spike",
            metricName="weightedRecall"
        )
        metrics["recall"] = recall_evaluator.evaluate(predictions_df)

        f1_evaluator = MulticlassClassificationEvaluator(
            labelCol=self.target_col,
            predictionCol="predicted_spike",
            metricName="f1"
        )
        metrics["f1"] = f1_evaluator.evaluate(predictions_df)

        # Class distribution (for context)
        from pyspark.sql.functions import col, sum as spark_sum, count

        total = predictions_df.count()
        spikes = predictions_df.filter(col(self.target_col) == 1).count()
        metrics["spike_rate"] = spikes / total if total > 0 else 0

        return metrics

    def get_spike_analysis(self, predictions_df: DataFrame) -> Dict[str, any]:
        """Analyze detected spikes.

        Args:
            predictions_df: DataFrame with predictions

        Returns:
            Dictionary with spike analysis
        """
        from pyspark.sql.functions import col, count, avg, max as spark_max

        # Filter to predicted spikes
        spikes = predictions_df.filter(col("predicted_spike") == 1)

        analysis = {
            "total_predictions": predictions_df.count(),
            "predicted_spikes": spikes.count(),
        }

        if spikes.count() > 0:
            # Analyze spike characteristics
            spike_stats = spikes.agg(
                avg("LBMP_avg").alias("avg_spike_price"),
                spark_max("LBMP_avg").alias("max_spike_price"),
                avg("Load_MW").alias("avg_spike_load"),
                avg("congestion_impact").alias("avg_congestion_impact")
            ).collect()[0]

            analysis.update({
                "avg_spike_price": spike_stats["avg_spike_price"],
                "max_spike_price": spike_stats["max_spike_price"],
                "avg_spike_load": spike_stats["avg_spike_load"],
                "avg_congestion_impact": spike_stats["avg_congestion_impact"],
            })

            # Hourly distribution
            hourly = spikes.groupBy("hour").count().orderBy("hour").collect()
            analysis["hourly_distribution"] = {row["hour"]: row["count"] for row in hourly}

        return analysis
