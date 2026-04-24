"""Spark session management for NYISO project."""
import os
from pyspark.sql import SparkSession
from .config import CONFIG

# Set JAVA_HOME for Spark
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"


def get_spark_session() -> SparkSession:
    """Create or get existing Spark session with project configuration."""
    spark_config = CONFIG.get("spark", {})

    builder = SparkSession.builder \
        .appName(spark_config.get("app_name", "NYISO-Analytics")) \
        .master(spark_config.get("master", "local[*]"))

    # Apply additional Spark configurations
    for key, value in spark_config.get("config", {}).items():
        builder = builder.config(key, value)

    return builder.getOrCreate()


def stop_spark_session():
    """Stop the active Spark session."""
    spark = SparkSession.getActiveSession()
    if spark:
        spark.stop()
