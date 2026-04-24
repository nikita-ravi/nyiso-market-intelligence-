"""Feature engineering for NYISO ML models."""
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, lag, avg, stddev, min as spark_min, max as spark_max,
    when, lit, abs as spark_abs
)
from pyspark.sql.window import Window

from ..utils.config import CONFIG


class NYISOFeatureEngineer:
    """Engineer features for NYISO price and demand prediction."""

    def __init__(self):
        """Initialize feature engineer with config."""
        self.config = CONFIG.get("features", {})
        self.rolling_windows = self.config.get("rolling_windows", [24, 168])

    def add_lag_features(self, df: DataFrame, target_col: str, lags: list = None) -> DataFrame:
        """Add lagged features for a target column.

        Args:
            df: Input DataFrame
            target_col: Column to create lags for
            lags: List of lag periods (default: [1, 2, 3, 6, 12, 24])

        Returns:
            DataFrame with lag features
        """
        if lags is None:
            lags = [1, 2, 3, 6, 12, 24]

        # Window spec partitioned by zone, ordered by timestamp
        window_spec = Window.partitionBy("Name").orderBy("timestamp")

        for lag_val in lags:
            df = df.withColumn(
                f"{target_col}_lag_{lag_val}",
                lag(col(target_col), lag_val).over(window_spec)
            )

        return df

    def add_rolling_statistics(self, df: DataFrame, target_col: str, windows: list = None) -> DataFrame:
        """Add rolling window statistics.

        Args:
            df: Input DataFrame
            target_col: Column to compute statistics for
            windows: List of window sizes in hours

        Returns:
            DataFrame with rolling statistics
        """
        if windows is None:
            windows = self.rolling_windows

        for window_size in windows:
            # Define rolling window
            window_spec = Window.partitionBy("Name").orderBy("timestamp").rowsBetween(-window_size, -1)

            # Rolling mean
            df = df.withColumn(
                f"{target_col}_ma_{window_size}",
                avg(col(target_col)).over(window_spec)
            )

            # Rolling std dev
            df = df.withColumn(
                f"{target_col}_std_{window_size}",
                stddev(col(target_col)).over(window_spec)
            )

            # Rolling min
            df = df.withColumn(
                f"{target_col}_min_{window_size}",
                spark_min(col(target_col)).over(window_spec)
            )

            # Rolling max
            df = df.withColumn(
                f"{target_col}_max_{window_size}",
                spark_max(col(target_col)).over(window_spec)
            )

        return df

    def add_price_features(self, df: DataFrame) -> DataFrame:
        """Add price-specific engineered features.

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with price features
        """
        # Price momentum (current vs 24h ago)
        window_spec = Window.partitionBy("Name").orderBy("timestamp")

        df = df.withColumn(
            "price_momentum_24h",
            col("LBMP_avg") - lag(col("LBMP_avg"), 24).over(window_spec)
        )

        # Price volatility ratio
        df = df.withColumn(
            "price_volatility_ratio",
            when(col("LBMP_avg_ma_24") != 0,
                 col("LBMP_avg_std_24") / spark_abs(col("LBMP_avg_ma_24")))
            .otherwise(lit(0))
        )

        # Congestion indicator (high congestion = high price impact)
        df = df.withColumn(
            "congestion_impact",
            when(col("LBMP_avg") != 0,
                 spark_abs(col("Marginal_Cost_Congestion_avg")) / spark_abs(col("LBMP_avg")))
            .otherwise(lit(0))
        )

        # Price deviation from 24h mean
        df = df.withColumn(
            "price_deviation_24h",
            when(col("LBMP_avg_std_24") != 0,
                 (col("LBMP_avg") - col("LBMP_avg_ma_24")) / col("LBMP_avg_std_24"))
            .otherwise(lit(0))
        )

        return df

    def add_load_features(self, df: DataFrame) -> DataFrame:
        """Add load-specific engineered features.

        Args:
            df: DataFrame with load data

        Returns:
            DataFrame with load features
        """
        window_spec = Window.partitionBy("Name").orderBy("timestamp")

        # Load change from previous hour
        df = df.withColumn(
            "load_change_1h",
            col("Load_MW") - lag(col("Load_MW"), 1).over(window_spec)
        )

        # Load ratio to 24h average
        df = df.withColumn(
            "load_ratio_24h",
            when(col("Load_MW_ma_24") != 0,
                 col("Load_MW") / col("Load_MW_ma_24"))
            .otherwise(lit(1))
        )

        # Peak hour indicator (load > 24h mean + 1 std)
        df = df.withColumn(
            "is_peak_load",
            when(col("Load_MW") > (col("Load_MW_ma_24") + col("Load_MW_std_24")),
                 lit(1))
            .otherwise(lit(0))
        )

        return df

    def add_spike_labels(self, df: DataFrame, threshold_std: float = 3.0) -> DataFrame:
        """Add price spike labels for classification.

        A spike is defined as price > (24h rolling mean + threshold_std * rolling std)

        Args:
            df: DataFrame with price data
            threshold_std: Number of standard deviations above mean to classify as spike

        Returns:
            DataFrame with spike labels
        """
        df = df.withColumn(
            "spike_threshold",
            col("LBMP_avg_ma_24") + (threshold_std * col("LBMP_avg_std_24"))
        )

        df = df.withColumn(
            "is_price_spike",
            when(col("LBMP_avg") > col("spike_threshold"), lit(1)).otherwise(lit(0))
        )

        # Categorize spike cause
        df = df.withColumn(
            "spike_cause",
            when(
                (col("is_price_spike") == 1) & (spark_abs(col("Marginal_Cost_Congestion_avg")) > spark_abs(col("LBMP_avg")) * 0.5),
                lit("congestion")
            ).when(
                (col("is_price_spike") == 1) & (col("is_peak_load") == 1),
                lit("high_demand")
            ).when(
                col("is_price_spike") == 1,
                lit("other")
            ).otherwise(lit("none"))
        )

        return df

    def engineer_all_features(self, df: DataFrame) -> DataFrame:
        """Run the complete feature engineering pipeline.

        Args:
            df: Processed DataFrame with price and load data

        Returns:
            DataFrame with all engineered features
        """
        # Add rolling statistics for price
        df = self.add_rolling_statistics(df, "LBMP_avg", windows=[24, 168])

        # Add rolling statistics for load
        df = self.add_rolling_statistics(df, "Load_MW", windows=[24, 168])

        # Add lag features
        df = self.add_lag_features(df, "LBMP_avg", lags=[1, 2, 3, 6, 12, 24])
        df = self.add_lag_features(df, "Load_MW", lags=[1, 2, 3, 6, 12, 24])

        # Add price-specific features
        df = self.add_price_features(df)

        # Add load-specific features
        df = self.add_load_features(df)

        # Add spike labels
        threshold = CONFIG.get("models", {}).get("spike_detection", {}).get("threshold_std", 3)
        df = self.add_spike_labels(df, threshold_std=threshold)

        # Drop rows with nulls from windowing operations
        df = df.na.drop()

        return df

    def get_feature_columns(self, target: str = "price") -> list:
        """Get the list of feature columns for a specific prediction target.

        Args:
            target: One of "price", "demand", or "spike"

        Returns:
            List of feature column names
        """
        # Base features
        base_features = [
            "hour", "day_of_week", "day_of_month", "month", "is_weekend",
            "hour_sin", "hour_cos", "month_sin", "month_cos"
        ]

        # Rolling features
        rolling_features = [
            "LBMP_avg_ma_24", "LBMP_avg_std_24", "LBMP_avg_min_24", "LBMP_avg_max_24",
            "LBMP_avg_ma_168", "LBMP_avg_std_168",
            "Load_MW_ma_24", "Load_MW_std_24", "Load_MW_min_24", "Load_MW_max_24",
            "Load_MW_ma_168", "Load_MW_std_168"
        ]

        # Lag features
        lag_features = [
            "LBMP_avg_lag_1", "LBMP_avg_lag_2", "LBMP_avg_lag_3",
            "LBMP_avg_lag_6", "LBMP_avg_lag_12", "LBMP_avg_lag_24",
            "Load_MW_lag_1", "Load_MW_lag_2", "Load_MW_lag_3",
            "Load_MW_lag_6", "Load_MW_lag_12", "Load_MW_lag_24"
        ]

        # Engineered features
        engineered_features = [
            "price_momentum_24h", "price_volatility_ratio", "congestion_impact",
            "price_deviation_24h", "load_change_1h", "load_ratio_24h", "is_peak_load"
        ]

        # Grid features
        grid_features = [
            "Marginal_Cost_Losses_avg", "Marginal_Cost_Congestion_avg"
        ]

        if target == "price":
            # For price prediction, include load but not current price
            return base_features + rolling_features + lag_features + engineered_features + grid_features + ["Load_MW"]

        elif target == "demand":
            # For demand prediction, include price
            return base_features + rolling_features + lag_features + engineered_features + grid_features + ["LBMP_avg"]

        elif target == "spike":
            # For spike detection
            return base_features + rolling_features + lag_features + engineered_features + grid_features + ["Load_MW"]

        else:
            return base_features + rolling_features + lag_features + engineered_features + grid_features
