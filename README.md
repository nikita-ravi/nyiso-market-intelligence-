# NYISO Market Intelligence

<p align="center">
  <img src="https://img.shields.io/badge/Apache%20Spark-3.5-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Databricks-ML-FF3621?style=for-the-badge&logo=databricks&logoColor=white" />
  <img src="https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Delta%20Lake-Storage-00ADD8?style=for-the-badge&logo=delta&logoColor=white" />
</p>

<p align="center">
  <b>End-to-end Big Data Analytics Platform for NYISO Electricity Market</b><br>
  <i>Distributed Processing | Machine Learning | Real-time Dashboard</i>
</p>

---

## Overview

Production-grade analytics platform for **New York Independent System Operator (NYISO)** electricity market data. This project demonstrates a complete data engineering and ML pipeline—processing **2.8M+ records** across 11 pricing zones to deliver price forecasting, demand prediction, and anomaly detection.

### Key Highlights

| Metric | Value |
|--------|-------|
| **Data Volume** | 2.8M+ records |
| **Price Prediction R²** | 0.7942 |
| **Demand Forecast R²** | 0.9972 |
| **Spike Detection AUC** | 0.9168 |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NYISO MARKET INTELLIGENCE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  RAW DATA   │───▶│  SPARK ETL  │───▶│  FEATURES   │───▶│  ML MODELS  │  │
│  │   (CSV)     │    │  Pipeline   │    │  Engineering│    │  (MLlib)    │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                  │                  │                  │          │
│        ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         DELTA LAKE / PARQUET                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│                        ┌─────────────────────────┐                         │
│                        │    PLOTLY DASHBOARD     │                         │
│                        │  Real-time Analytics    │                         │
│                        └─────────────────────────┘                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Processing** | Apache Spark 3.5, PySpark |
| **ML Framework** | Spark MLlib, scikit-learn |
| **Platform** | Databricks, Delta Lake |
| **MLOps** | MLflow (tracking, registry) |
| **Visualization** | Plotly Dash, Plotly Express |
| **Storage** | Delta Lake, Parquet |
| **Language** | Python 3.11 |

---

## Project Structure

```
nyiso-market-intelligence/
│
├── notebooks/
│   └── NYISO_PySpark_Analysis.ipynb   # Full EDA + Spark SQL analysis
│
├── src/
│   ├── ingestion/
│   │   └── data_loader.py             # Distributed data ingestion
│   ├── processing/
│   │   └── cleaner.py                 # ETL transformations
│   ├── features/
│   │   └── engineer.py                # Feature engineering pipeline
│   └── models/
│       ├── price_predictor.py         # GBT price forecasting
│       ├── demand_forecaster.py       # GBT demand prediction
│       ├── spike_detector.py          # RF anomaly detection
│       └── evaluator.py               # Model evaluation suite
│
├── dashboard/
│   └── app.py                         # Multi-page Dash application
│
├── results/
│   ├── predictions.parquet            # Model predictions
│   └── model_metrics.json             # Performance metrics
│
├── config/
│   └── config.yaml                    # Pipeline configuration
│
└── run_full_analysis.py               # End-to-end pipeline script
```

---

## Data Pipeline

### Source Data
- **LBMP (Locational Based Marginal Pricing)**: Real-time electricity prices across 11 NYISO zones
- **Load Data**: Actual demand (MW) at 5-minute intervals
- **Temporal Scope**: Full year of market operations

### ETL Process

```python
# Distributed ingestion with schema inference
price_df = spark.read.csv(price_files, header=True, inferSchema=True)

# Hourly aggregations with Spark SQL
price_hourly = price_df \
    .withColumn("hour_timestamp", date_trunc("hour", col("timestamp"))) \
    .groupBy("hour_timestamp", "zone") \
    .agg(
        avg("LBMP").alias("price_avg"),
        stddev("LBMP").alias("price_std"),
        avg("Marginal_Cost_Congestion").alias("congestion_avg")
    )

# Zone-level joins
merged_df = price_hourly.join(load_hourly, on=["hour_timestamp", "zone"])
```

---

## Feature Engineering

Built **20+ features** using Spark Window Functions:

| Category | Features |
|----------|----------|
| **Temporal** | hour, day_of_week, month, is_weekend, is_peak_hour |
| **Cyclical** | hour_sin, hour_cos, month_sin, month_cos |
| **Rolling Stats** | price_ma_24h, price_std_24h, load_ma_24h |
| **Lag Features** | price_lag_1h, price_lag_24h, load_lag_1h, load_lag_24h |
| **Derived** | price_volatility, load_ratio, congestion_impact |

```python
# Window functions for time-series features
window_24h = Window.partitionBy("zone").orderBy("timestamp").rowsBetween(-24, -1)

df = df \
    .withColumn("price_ma_24h", avg("LBMP").over(window_24h)) \
    .withColumn("price_std_24h", stddev("LBMP").over(window_24h)) \
    .withColumn("price_volatility", col("price_std_24h") / abs(col("price_ma_24h")))
```

---

## Machine Learning Models

### 1. Price Prediction (Regression)

| Specification | Value |
|--------------|-------|
| Algorithm | Gradient Boosted Trees |
| Target | Next-hour LBMP ($/MWh) |
| R² Score | **0.7942** |
| RMSE | $21.94/MWh |
| MAE | $7.46/MWh |

### 2. Demand Forecasting (Regression)

| Specification | Value |
|--------------|-------|
| Algorithm | Gradient Boosted Trees |
| Target | Hourly Load (MW) |
| R² Score | **0.9972** |
| RMSE | 84.85 MW |

### 3. Spike Detection (Classification)

| Specification | Value |
|--------------|-------|
| Algorithm | Random Forest |
| Target | Price anomalies (>3σ) |
| AUC-ROC | **0.9168** |
| Accuracy | 97.3% |
| F1 Score | 0.96 |

### MLlib Pipeline

```python
pipeline = Pipeline(stages=[
    VectorAssembler(inputCols=feature_cols, outputCol="features_raw"),
    StandardScaler(inputCol="features_raw", outputCol="features"),
    GBTRegressor(
        labelCol="LBMP_avg",
        featuresCol="features",
        maxDepth=8,
        maxIter=50
    )
])

model = pipeline.fit(train_df)
predictions = model.transform(test_df)
```

---

## Dashboard

Interactive multi-page analytics dashboard built with Plotly Dash:

**Page 1 - Market Analytics**
- Price/demand time series by zone
- Hour × Day-of-Week heatmaps
- Zone comparison charts
- Distribution analysis

**Page 2 - ML Performance**
- Model metric cards (R², RMSE, AUC)
- Actual vs Predicted visualizations
- Feature importance rankings
- Prediction error analysis

### Run Dashboard

```bash
python dashboard/app.py
# Navigate to http://localhost:8050
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- Java 17 (for Spark)
- 8GB+ RAM recommended

### Installation

```bash
# Clone repository
git clone https://github.com/nikita-ravi/nyiso-market-intelligence-.git
cd nyiso-market-intelligence-

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install pyspark==3.5.0 pandas numpy plotly dash pyarrow pyyaml
```

### Run Full Pipeline

```bash
# Execute end-to-end analysis
python run_full_analysis.py

# Start dashboard
python dashboard/app.py
```

---

## MLOps Practices

| Practice | Implementation |
|----------|----------------|
| **Experiment Tracking** | MLflow for metrics and model versioning |
| **Feature Store** | Centralized feature computation in Delta Lake |
| **Model Registry** | Versioned models with staging/production stages |
| **Data Versioning** | Delta Lake time travel for reproducibility |
| **Pipeline Orchestration** | Databricks Workflows for scheduled runs |

---

## Results

### Model Comparison

```
┌────────────────────┬─────────────┬─────────────┬─────────────┐
│       Model        │  Primary    │  Secondary  │   Tertiary  │
│                    │  Metric     │  Metric     │   Metric    │
├────────────────────┼─────────────┼─────────────┼─────────────┤
│ Price Prediction   │ R² = 0.794  │ RMSE = 21.9 │ MAE = 7.46  │
│ Demand Forecast    │ R² = 0.997  │ RMSE = 84.8 │ MAE = 40.5  │
│ Spike Detection    │ AUC = 0.917 │ Acc = 97.3% │ F1 = 0.96   │
└────────────────────┴─────────────┴─────────────┴─────────────┘
```

### Top Features (Price Prediction)

1. `price_lag_1h` - Previous hour price
2. `price_ma_24h` - 24-hour moving average
3. `Load_MW` - Current demand
4. `Marginal_Cost_Congestion` - Grid congestion costs
5. `hour` - Time of day

---

## Business Applications

- **Grid Operators**: Anticipate demand spikes for resource allocation
- **Energy Traders**: Price forecasting for trading strategies
- **Utilities**: Load planning and congestion cost management
- **Regulators**: Market anomaly detection and monitoring

---

## Future Enhancements

- [ ] Real-time streaming with Spark Structured Streaming
- [ ] Deep learning models (LSTM, Transformer)
- [ ] Automated retraining pipelines
- [ ] A/B testing framework for model deployment
- [ ] External data integration (weather, holidays)

---

## License

MIT License - feel free to use for learning and portfolio purposes.

---

<p align="center">
  <b>Built with Apache Spark + MLlib + Plotly Dash</b><br>
  <i>Demonstrating end-to-end data engineering and ML capabilities</i>
</p>
