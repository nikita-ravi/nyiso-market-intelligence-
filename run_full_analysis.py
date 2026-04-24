#!/usr/bin/env python3
"""
NYISO Full Analysis Pipeline
Runs Spark processing, trains all models, saves results for dashboard
"""
import os
import sys

# Set JAVA_HOME
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"

import math
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.pipeline import Pipeline
import pandas as pd
import numpy as np
import json
from pathlib import Path
from glob import glob

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print("="*70)
print("NYISO FULL ANALYSIS PIPELINE")
print("="*70)

# Initialize Spark
print("\n[1/8] Initializing Spark...")
spark = SparkSession.builder \
    .appName("NYISO_Full_Analysis") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .master("local[*]") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")
print(f"Spark {spark.version} initialized")

# Load Data
print("\n[2/8] Loading data...")
price_files = glob(str(PROJECT_ROOT / "*realtime_zone_csv" / "*.csv"))
load_files = glob(str(PROJECT_ROOT / "*pal_csv" / "*.csv"))

price_df = spark.read.csv(price_files, header=True, inferSchema=True, quote='"')
price_df = price_df \
    .withColumnRenamed("LBMP ($/MWHr)", "LBMP") \
    .withColumnRenamed("Marginal Cost Losses ($/MWHr)", "Marginal_Cost_Losses") \
    .withColumnRenamed("Marginal Cost Congestion ($/MWHr)", "Marginal_Cost_Congestion") \
    .withColumn("timestamp", to_timestamp(col("Time Stamp"), "MM/dd/yyyy HH:mm:ss"))

load_df = spark.read.csv(load_files, header=True, inferSchema=True, quote='"')
load_df = load_df.withColumn("timestamp", to_timestamp(col("Time Stamp"), "MM/dd/yyyy HH:mm:ss"))

print(f"  Price records: {price_df.count():,}")
print(f"  Load records: {load_df.count():,}")

# Process Data
print("\n[3/8] Processing and aggregating data...")
price_hourly = price_df \
    .withColumn("hour_timestamp", date_trunc("hour", col("timestamp"))) \
    .groupBy("hour_timestamp", "Name", "PTID") \
    .agg(
        avg("LBMP").alias("LBMP_avg"),
        stddev("LBMP").alias("LBMP_std"),
        avg("Marginal_Cost_Losses").alias("Marginal_Cost_Losses_avg"),
        avg("Marginal_Cost_Congestion").alias("Marginal_Cost_Congestion_avg")
    )

load_hourly = load_df \
    .withColumn("hour_timestamp", date_trunc("hour", col("timestamp"))) \
    .groupBy("hour_timestamp", "Name") \
    .agg(avg("Load").alias("Load_MW"))

merged_df = price_hourly.join(load_hourly, on=["hour_timestamp", "Name"], how="inner")
print(f"  Merged records: {merged_df.count():,}")

# Add Features
print("\n[4/8] Engineering features...")
df = merged_df \
    .withColumn("hour", hour("hour_timestamp")) \
    .withColumn("day_of_week", dayofweek("hour_timestamp")) \
    .withColumn("day_of_month", dayofmonth("hour_timestamp")) \
    .withColumn("month", month("hour_timestamp")) \
    .withColumn("year", year("hour_timestamp")) \
    .withColumn("is_weekend", when(dayofweek("hour_timestamp").isin([1, 7]), 1).otherwise(0)) \
    .withColumn("is_peak_hour", when((hour("hour_timestamp") >= 7) & (hour("hour_timestamp") <= 22), 1).otherwise(0)) \
    .withColumn("hour_sin", sin(2 * math.pi * col("hour") / 24)) \
    .withColumn("hour_cos", cos(2 * math.pi * col("hour") / 24)) \
    .withColumn("month_sin", sin(2 * math.pi * col("month") / 12)) \
    .withColumn("month_cos", cos(2 * math.pi * col("month") / 12))

# Window functions
window_24h = Window.partitionBy("Name").orderBy("hour_timestamp").rowsBetween(-24, -1)
window_lag = Window.partitionBy("Name").orderBy("hour_timestamp")

df = df \
    .withColumn("price_ma_24h", avg("LBMP_avg").over(window_24h)) \
    .withColumn("price_std_24h", stddev("LBMP_avg").over(window_24h)) \
    .withColumn("load_ma_24h", avg("Load_MW").over(window_24h)) \
    .withColumn("load_std_24h", stddev("Load_MW").over(window_24h)) \
    .withColumn("price_lag_1h", lag("LBMP_avg", 1).over(window_lag)) \
    .withColumn("price_lag_24h", lag("LBMP_avg", 24).over(window_lag)) \
    .withColumn("load_lag_1h", lag("Load_MW", 1).over(window_lag)) \
    .withColumn("load_lag_24h", lag("Load_MW", 24).over(window_lag)) \
    .withColumn("price_volatility", when(col("price_ma_24h") != 0, col("price_std_24h") / abs(col("price_ma_24h"))).otherwise(0)) \
    .withColumn("load_ratio_24h", when(col("load_ma_24h") != 0, col("Load_MW") / col("load_ma_24h")).otherwise(1)) \
    .withColumn("congestion_impact", when(col("LBMP_avg") != 0, abs(col("Marginal_Cost_Congestion_avg")) / abs(col("LBMP_avg"))).otherwise(0)) \
    .withColumn("is_price_spike", when(col("LBMP_avg") > (col("price_ma_24h") + 3 * col("price_std_24h")), 1).otherwise(0))

df = df.na.drop()
df.cache()
print(f"  Featured records: {df.count():,}")

# Split Data
print("\n[5/8] Splitting data...")
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
train_df.cache()
test_df.cache()
print(f"  Train: {train_df.count():,}, Test: {test_df.count():,}")

# Define feature columns
price_features = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour',
                  'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                  'Load_MW', 'load_ma_24h', 'price_lag_1h', 'price_lag_24h',
                  'price_ma_24h', 'Marginal_Cost_Losses_avg', 'Marginal_Cost_Congestion_avg']

demand_features = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour',
                   'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                   'LBMP_avg', 'price_ma_24h', 'load_lag_1h', 'load_lag_24h',
                   'load_ma_24h', 'Marginal_Cost_Congestion_avg']

spike_features = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour',
                  'Load_MW', 'load_ma_24h', 'load_ratio_24h', 'price_lag_1h',
                  'price_ma_24h', 'price_volatility', 'congestion_impact']

results = {}

# Train Price Predictor
print("\n[6/8] Training Price Prediction Model (GBT)...")
price_assembler = VectorAssembler(inputCols=price_features, outputCol="features_raw", handleInvalid="skip")
price_scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
price_gbt = GBTRegressor(labelCol="LBMP_avg", featuresCol="features", predictionCol="predicted_price", maxDepth=8, maxIter=50, seed=42)
price_pipeline = Pipeline(stages=[price_assembler, price_scaler, price_gbt])
price_model = price_pipeline.fit(train_df)

price_predictions = price_model.transform(test_df)
price_rmse = RegressionEvaluator(labelCol="LBMP_avg", predictionCol="predicted_price", metricName="rmse").evaluate(price_predictions)
price_r2 = RegressionEvaluator(labelCol="LBMP_avg", predictionCol="predicted_price", metricName="r2").evaluate(price_predictions)
price_mae = RegressionEvaluator(labelCol="LBMP_avg", predictionCol="predicted_price", metricName="mae").evaluate(price_predictions)

results['price_prediction'] = {'r2': price_r2, 'rmse': price_rmse, 'mae': price_mae}
print(f"  R²: {price_r2:.4f}, RMSE: {price_rmse:.2f}, MAE: {price_mae:.2f}")

# Get feature importance
price_importance = dict(zip(price_features, price_model.stages[-1].featureImportances.toArray().tolist()))
results['price_feature_importance'] = price_importance

# Train Demand Forecaster
print("\n[7/8] Training Demand Forecast Model (GBT)...")
demand_assembler = VectorAssembler(inputCols=demand_features, outputCol="features_raw", handleInvalid="skip")
demand_scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
demand_gbt = GBTRegressor(labelCol="Load_MW", featuresCol="features", predictionCol="predicted_load", maxDepth=8, maxIter=50, seed=42)
demand_pipeline = Pipeline(stages=[demand_assembler, demand_scaler, demand_gbt])
demand_model = demand_pipeline.fit(train_df)

demand_predictions = demand_model.transform(test_df)
demand_rmse = RegressionEvaluator(labelCol="Load_MW", predictionCol="predicted_load", metricName="rmse").evaluate(demand_predictions)
demand_r2 = RegressionEvaluator(labelCol="Load_MW", predictionCol="predicted_load", metricName="r2").evaluate(demand_predictions)
demand_mae = RegressionEvaluator(labelCol="Load_MW", predictionCol="predicted_load", metricName="mae").evaluate(demand_predictions)

results['demand_forecast'] = {'r2': demand_r2, 'rmse': demand_rmse, 'mae': demand_mae}
print(f"  R²: {demand_r2:.4f}, RMSE: {demand_rmse:.2f}, MAE: {demand_mae:.2f}")

demand_importance = dict(zip(demand_features, demand_model.stages[-1].featureImportances.toArray().tolist()))
results['demand_feature_importance'] = demand_importance

# Train Spike Detector
print("\n[8/8] Training Spike Detection Model (Random Forest)...")
spike_assembler = VectorAssembler(inputCols=spike_features, outputCol="features_raw", handleInvalid="skip")
spike_scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
spike_rf = RandomForestClassifier(labelCol="is_price_spike", featuresCol="features", predictionCol="predicted_spike",
                                   probabilityCol="spike_probability", numTrees=100, maxDepth=10, seed=42)
spike_pipeline = Pipeline(stages=[spike_assembler, spike_scaler, spike_rf])
spike_model = spike_pipeline.fit(train_df)

spike_predictions = spike_model.transform(test_df)
spike_auc = BinaryClassificationEvaluator(labelCol="is_price_spike", rawPredictionCol="spike_probability", metricName="areaUnderROC").evaluate(spike_predictions)
spike_acc = MulticlassClassificationEvaluator(labelCol="is_price_spike", predictionCol="predicted_spike", metricName="accuracy").evaluate(spike_predictions)
spike_f1 = MulticlassClassificationEvaluator(labelCol="is_price_spike", predictionCol="predicted_spike", metricName="f1").evaluate(spike_predictions)

results['spike_detection'] = {'auc': spike_auc, 'accuracy': spike_acc, 'f1': spike_f1}
print(f"  AUC: {spike_auc:.4f}, Accuracy: {spike_acc:.4f}, F1: {spike_f1:.4f}")

spike_importance = dict(zip(spike_features, spike_model.stages[-1].featureImportances.toArray().tolist()))
results['spike_feature_importance'] = spike_importance

# Save predictions for dashboard
print("\n[*] Saving results for dashboard...")

# Combine all predictions
all_predictions = price_predictions.join(
    demand_predictions.select("hour_timestamp", "Name", "predicted_load"),
    on=["hour_timestamp", "Name"]
).join(
    spike_predictions.select("hour_timestamp", "Name", "predicted_spike"),
    on=["hour_timestamp", "Name"]
)

# Convert to pandas and save
pred_pd = all_predictions.select(
    "hour_timestamp", "Name", "LBMP_avg", "predicted_price",
    "Load_MW", "predicted_load", "is_price_spike", "predicted_spike",
    "hour", "day_of_week", "month", "is_weekend", "price_ma_24h",
    "load_ma_24h", "Marginal_Cost_Congestion_avg"
).toPandas()

pred_pd.to_parquet(RESULTS_DIR / "predictions.parquet")
print(f"  Predictions saved: {len(pred_pd):,} records")

# Save metrics
with open(RESULTS_DIR / "model_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("  Model metrics saved")

# Save full featured data
df.write.mode("overwrite").parquet(str(PROJECT_ROOT / "data" / "processed" / "full_featured_data"))
print("  Full featured data saved")

# Cleanup
train_df.unpersist()
test_df.unpersist()
df.unpersist()
spark.stop()

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print(f"\nResults saved to: {RESULTS_DIR}")
print("\nModel Performance Summary:")
print(f"  Price Prediction:  R² = {results['price_prediction']['r2']:.4f}")
print(f"  Demand Forecast:   R² = {results['demand_forecast']['r2']:.4f}")
print(f"  Spike Detection:   AUC = {results['spike_detection']['auc']:.4f}")
