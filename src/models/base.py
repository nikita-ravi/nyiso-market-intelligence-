"""Base model class for NYISO ML models."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

from pyspark.sql import DataFrame
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator

from ..utils.config import CONFIG, get_project_root


class BaseModel(ABC):
    """Abstract base class for NYISO prediction models."""

    def __init__(self, name: str, feature_cols: list, target_col: str):
        """Initialize base model.

        Args:
            name: Model name for saving/loading
            feature_cols: List of feature column names
            target_col: Target column name
        """
        self.name = name
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.model: Optional[PipelineModel] = None
        self.model_config = CONFIG.get("models", {})
        self.model_dir = get_project_root() / "models"

    def _create_feature_pipeline(self) -> list:
        """Create feature preparation stages (assembler + scaler)."""
        assembler = VectorAssembler(
            inputCols=self.feature_cols,
            outputCol="features_raw",
            handleInvalid="skip"
        )

        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withStd=True,
            withMean=True
        )

        return [assembler, scaler]

    @abstractmethod
    def _create_model_stage(self):
        """Create the ML model stage. Must be implemented by subclasses."""
        pass

    def build_pipeline(self) -> Pipeline:
        """Build the complete ML pipeline."""
        stages = self._create_feature_pipeline()
        stages.append(self._create_model_stage())
        return Pipeline(stages=stages)

    def train(self, train_df: DataFrame) -> "BaseModel":
        """Train the model.

        Args:
            train_df: Training DataFrame

        Returns:
            self for chaining
        """
        pipeline = self.build_pipeline()
        self.model = pipeline.fit(train_df)
        return self

    def predict(self, df: DataFrame) -> DataFrame:
        """Make predictions.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.transform(df)

    @abstractmethod
    def evaluate(self, predictions_df: DataFrame) -> Dict[str, float]:
        """Evaluate model performance. Must be implemented by subclasses."""
        pass

    def save(self, path: Optional[Path] = None):
        """Save the trained model.

        Args:
            path: Optional custom path. Defaults to models/{name}
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        save_path = path or (self.model_dir / self.name)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.write().overwrite().save(str(save_path))
        print(f"Model saved to {save_path}")

    def load(self, path: Optional[Path] = None) -> "BaseModel":
        """Load a trained model.

        Args:
            path: Optional custom path. Defaults to models/{name}

        Returns:
            self for chaining
        """
        load_path = path or (self.model_dir / self.name)
        self.model = PipelineModel.load(str(load_path))
        print(f"Model loaded from {load_path}")
        return self

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available.

        Returns:
            Dictionary of feature names to importance scores, or None
        """
        if self.model is None:
            return None

        # Try to get feature importance from tree-based models
        try:
            # Get the last stage (the actual model)
            model_stage = self.model.stages[-1]
            if hasattr(model_stage, 'featureImportances'):
                importances = model_stage.featureImportances.toArray()
                return dict(zip(self.feature_cols, importances))
        except Exception:
            pass

        return None
