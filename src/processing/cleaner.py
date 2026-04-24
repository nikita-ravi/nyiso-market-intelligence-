"""Data cleaning and processing for NYISO data."""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col, hour, dayofweek, dayofmonth, month, year,
    when, lit, avg, count, first
)
from pyspark.sql.window import Window
from pathlib import Path

from ..utils.config import CONFIG, get_project_root


class NYISODataProcessor:
    """Clean and process NYISO price and load data."""

    def __init__(self, spark: SparkSession):
        """Initialize the processor.

        Args:
            spark: Active Spark session
        """
        self.spark = spark
        self.config = CONFIG.get("data", {})
        self.processed_dir = get_project_root() / self.config.get("processed_dir", "data/processed")

    def clean_price_data(self, df: DataFrame) -> DataFrame:
        """Clean price data.

        - Remove rows with null timestamps
        - Handle missing values
        - Filter out invalid prices (keep reasonable range)

        Args:
            df: Raw price DataFrame

        Returns:
            Cleaned price DataFrame
        """
        df_clean = df.filter(col("timestamp").isNotNull())

        # Fill missing price components with 0
        df_clean = df_clean.fillna({
            "LBMP": 0.0,
            "Marginal_Cost_Losses": 0.0,
            "Marginal_Cost_Congestion": 0.0
        })

        # Add data quality flag for extreme values
        df_clean = df_clean.withColumn(
            "is_extreme_price",
            when((col("LBMP") < -500) | (col("LBMP") > 2000), lit(True)).otherwise(lit(False))
        )

        return df_clean

    def clean_load_data(self, df: DataFrame) -> DataFrame:
        """Clean load data.

        - Remove rows with null timestamps
        - Handle missing values
        - Filter out invalid loads

        Args:
            df: Raw load DataFrame

        Returns:
            Cleaned load DataFrame
        """
        df_clean = df.filter(col("timestamp").isNotNull())

        # Fill missing loads with 0 (will be filtered in analysis)
        df_clean = df_clean.fillna({"Load": 0.0})

        # Add flag for negative loads (possible data quality issue)
        df_clean = df_clean.withColumn(
            "is_negative_load",
            when(col("Load") < 0, lit(True)).otherwise(lit(False))
        )

        return df_clean

    def aggregate_price_to_hourly(self, df: DataFrame) -> DataFrame:
        """Aggregate 5-minute price data to hourly.

        Args:
            df: Price DataFrame with 5-minute intervals

        Returns:
            Hourly aggregated price DataFrame
        """
        from pyspark.sql.functions import date_trunc

        df_hourly = df.withColumn(
            "hour_timestamp",
            date_trunc("hour", col("timestamp"))
        )

        aggregated = df_hourly.groupBy("hour_timestamp", "Name", "PTID").agg(
            avg("LBMP").alias("LBMP_avg"),
            avg("Marginal_Cost_Losses").alias("Marginal_Cost_Losses_avg"),
            avg("Marginal_Cost_Congestion").alias("Marginal_Cost_Congestion_avg"),
            count("*").alias("price_count")
        )

        return aggregated.withColumnRenamed("hour_timestamp", "timestamp")

    def merge_price_load(self, price_df: DataFrame, load_df: DataFrame) -> DataFrame:
        """Merge price and load data on timestamp and zone.

        Args:
            price_df: Hourly price DataFrame
            load_df: Load DataFrame

        Returns:
            Merged DataFrame
        """
        # Ensure both have the same timestamp granularity
        from pyspark.sql.functions import date_trunc

        load_hourly = load_df.withColumn(
            "timestamp",
            date_trunc("hour", col("timestamp"))
        ).select(
            "timestamp",
            "Name",
            "PTID",
            col("Load").alias("Load_MW"),
            "Time Zone"
        ).dropDuplicates(["timestamp", "Name"])

        merged = price_df.join(
            load_hourly,
            on=["timestamp", "Name"],
            how="inner"
        )

        # Drop duplicate PTID column from load
        if "PTID" in [c for c in merged.columns if merged.columns.count(c) > 1]:
            merged = merged.drop(load_hourly["PTID"])

        return merged

    def add_temporal_features(self, df: DataFrame) -> DataFrame:
        """Add basic temporal features to DataFrame.

        Args:
            df: DataFrame with timestamp column

        Returns:
            DataFrame with temporal features
        """
        import math
        from pyspark.sql.functions import sin, cos, udf
        from pyspark.sql.types import DoubleType

        df = df.withColumn("hour", hour("timestamp"))
        df = df.withColumn("day_of_week", dayofweek("timestamp"))
        df = df.withColumn("day_of_month", dayofmonth("timestamp"))
        df = df.withColumn("month", month("timestamp"))
        df = df.withColumn("year", year("timestamp"))

        # Is weekend flag
        df = df.withColumn(
            "is_weekend",
            when(col("day_of_week").isin([1, 7]), lit(1)).otherwise(lit(0))
        )

        # Cyclical encoding for hour
        df = df.withColumn("hour_sin", sin(2 * math.pi * col("hour") / 24))
        df = df.withColumn("hour_cos", cos(2 * math.pi * col("hour") / 24))

        # Cyclical encoding for month
        df = df.withColumn("month_sin", sin(2 * math.pi * col("month") / 12))
        df = df.withColumn("month_cos", cos(2 * math.pi * col("month") / 12))

        return df

    def save_to_parquet(self, df: DataFrame, name: str) -> Path:
        """Save DataFrame to parquet format.

        Args:
            df: DataFrame to save
            name: Name for the parquet file/directory

        Returns:
            Path to saved parquet
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.processed_dir / name

        df.write.mode("overwrite").parquet(str(output_path))
        print(f"Saved to {output_path}")

        return output_path

    def process_all(self, price_df: DataFrame, load_df: DataFrame) -> DataFrame:
        """Run full processing pipeline.

        Args:
            price_df: Raw price DataFrame
            load_df: Raw load DataFrame

        Returns:
            Fully processed and merged DataFrame
        """
        # Clean data
        price_clean = self.clean_price_data(price_df)
        load_clean = self.clean_load_data(load_df)

        # Aggregate price to hourly
        price_hourly = self.aggregate_price_to_hourly(price_clean)

        # Merge datasets
        merged = self.merge_price_load(price_hourly, load_clean)

        # Add temporal features
        processed = self.add_temporal_features(merged)

        return processed
