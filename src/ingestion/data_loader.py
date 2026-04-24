"""Data loader for NYISO price and load data."""
from pathlib import Path
from typing import Optional
from glob import glob

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, TimestampType
)
from pyspark.sql.functions import to_timestamp, col

from ..utils.config import CONFIG, get_project_root


class NYISODataLoader:
    """Load NYISO price and load data from CSV files."""

    PRICE_SCHEMA = StructType([
        StructField("Time Stamp", StringType(), True),
        StructField("Name", StringType(), True),
        StructField("PTID", StringType(), True),
        StructField("LBMP", DoubleType(), True),
        StructField("Marginal_Cost_Losses", DoubleType(), True),
        StructField("Marginal_Cost_Congestion", DoubleType(), True),
    ])

    LOAD_SCHEMA = StructType([
        StructField("Time Stamp", StringType(), True),
        StructField("Time Zone", StringType(), True),
        StructField("Name", StringType(), True),
        StructField("PTID", StringType(), True),
        StructField("Load", DoubleType(), True),
    ])

    def __init__(self, spark: SparkSession, data_dir: Optional[Path] = None):
        """Initialize the data loader.

        Args:
            spark: Active Spark session
            data_dir: Directory containing data folders (defaults to project root)
        """
        self.spark = spark
        self.data_dir = data_dir or get_project_root()
        self.config = CONFIG.get("data", {})

    def _find_csv_files(self, pattern: str) -> list:
        """Find all CSV files matching the directory pattern."""
        dir_pattern = str(self.data_dir / pattern)
        matching_dirs = glob(dir_pattern)
        csv_files = []
        for d in matching_dirs:
            csv_files.extend(glob(f"{d}/*.csv"))
        return sorted(csv_files)

    def load_price_data(self) -> DataFrame:
        """Load all price (LBMP) data from realtime_zone CSV files.

        Returns:
            Spark DataFrame with price data
        """
        pattern = self.config.get("price_pattern", "20*realtime_zone_csv")
        csv_files = self._find_csv_files(pattern)

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found matching pattern: {pattern}")

        print(f"Loading {len(csv_files)} price data files...")

        df = self.spark.read.csv(
            csv_files,
            header=True,
            inferSchema=True,
            quote='"'
        )

        # Rename columns to be Spark-friendly
        df = df.withColumnRenamed("LBMP ($/MWHr)", "LBMP") \
               .withColumnRenamed("Marginal Cost Losses ($/MWHr)", "Marginal_Cost_Losses") \
               .withColumnRenamed("Marginal Cost Congestion ($/MWHr)", "Marginal_Cost_Congestion")

        # Parse timestamp
        df = df.withColumn(
            "timestamp",
            to_timestamp(col("Time Stamp"), "MM/dd/yyyy HH:mm:ss")
        )

        return df

    def load_load_data(self) -> DataFrame:
        """Load all load (demand) data from pal CSV files.

        Returns:
            Spark DataFrame with load data
        """
        pattern = self.config.get("load_pattern", "20*pal_csv")
        csv_files = self._find_csv_files(pattern)

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found matching pattern: {pattern}")

        print(f"Loading {len(csv_files)} load data files...")

        df = self.spark.read.csv(
            csv_files,
            header=True,
            inferSchema=True,
            quote='"'
        )

        # Parse timestamp
        df = df.withColumn(
            "timestamp",
            to_timestamp(col("Time Stamp"), "MM/dd/yyyy HH:mm:ss")
        )

        return df

    def load_all(self) -> tuple:
        """Load both price and load data.

        Returns:
            Tuple of (price_df, load_df)
        """
        price_df = self.load_price_data()
        load_df = self.load_load_data()
        return price_df, load_df

    def get_data_summary(self, df: DataFrame, name: str = "DataFrame") -> dict:
        """Get summary statistics for a DataFrame."""
        return {
            "name": name,
            "row_count": df.count(),
            "column_count": len(df.columns),
            "columns": df.columns,
        }
