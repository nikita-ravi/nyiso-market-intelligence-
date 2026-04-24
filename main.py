#!/usr/bin/env python3
"""Main CLI for NYISO PySpark Analytics Pipeline.

Usage:
    python main.py pipeline    # Run full data pipeline
    python main.py train       # Train all models
    python main.py evaluate    # Evaluate models
    python main.py dashboard   # Launch dashboard
    python main.py all         # Run everything
"""
import os
import sys
import argparse

# Set JAVA_HOME before importing PySpark
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"

from src.utils.spark_session import get_spark_session, stop_spark_session
from src.utils.config import CONFIG, get_project_root
from src.ingestion import NYISODataLoader
from src.processing.cleaner import NYISODataProcessor
from src.features.engineer import NYISOFeatureEngineer
from src.models import PricePredictor, DemandForecaster, SpikeDetector
from src.models.evaluator import ModelEvaluator


def run_pipeline():
    """Run the data processing pipeline."""
    print("\n" + "="*60)
    print("NYISO Data Pipeline")
    print("="*60)

    spark = get_spark_session()

    # Step 1: Load data
    print("\n[1/4] Loading data...")
    loader = NYISODataLoader(spark)
    price_df, load_df = loader.load_all()
    print(f"  - Price records: {price_df.count():,}")
    print(f"  - Load records: {load_df.count():,}")

    # Step 2: Process data
    print("\n[2/4] Processing data...")
    processor = NYISODataProcessor(spark)
    processed_df = processor.process_all(price_df, load_df)
    print(f"  - Processed records: {processed_df.count():,}")

    # Step 3: Engineer features
    print("\n[3/4] Engineering features...")
    engineer = NYISOFeatureEngineer()
    featured_df = engineer.engineer_all_features(processed_df)
    print(f"  - Featured records: {featured_df.count():,}")
    print(f"  - Feature columns: {len(featured_df.columns)}")

    # Step 4: Save processed data
    print("\n[4/4] Saving to parquet...")
    output_path = get_project_root() / "data" / "processed" / "featured_data"
    featured_df.write.mode("overwrite").parquet(str(output_path))
    print(f"  - Saved to: {output_path}")

    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)

    return featured_df


def train_models(df=None):
    """Train all ML models."""
    print("\n" + "="*60)
    print("Model Training")
    print("="*60)

    spark = get_spark_session()

    # Load data if not provided
    if df is None:
        data_path = get_project_root() / "data" / "processed" / "featured_data"
        if not data_path.exists():
            print("No processed data found. Running pipeline first...")
            df = run_pipeline()
        else:
            print(f"Loading processed data from {data_path}")
            df = spark.read.parquet(str(data_path))

    # Split data
    print("\nSplitting data (80/20)...")
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    print(f"  - Training samples: {train_df.count():,}")
    print(f"  - Test samples: {test_df.count():,}")

    # Cache for faster training
    train_df.cache()
    test_df.cache()

    models_dir = get_project_root() / "models"
    models_dir.mkdir(exist_ok=True)

    # Train Price Predictor
    print("\n[1/3] Training Price Predictor...")
    price_model = PricePredictor()
    price_model.train(train_df)
    price_model.save()
    print("  - Price Predictor trained and saved")

    # Train Demand Forecaster
    print("\n[2/3] Training Demand Forecaster...")
    demand_model = DemandForecaster()
    demand_model.train(train_df)
    demand_model.save()
    print("  - Demand Forecaster trained and saved")

    # Train Spike Detector
    print("\n[3/3] Training Spike Detector...")
    spike_model = SpikeDetector()
    spike_model.train(train_df)
    spike_model.save()
    print("  - Spike Detector trained and saved")

    print("\n" + "="*60)
    print("All models trained successfully!")
    print("="*60)

    return train_df, test_df


def evaluate_models(test_df=None):
    """Evaluate all trained models."""
    print("\n" + "="*60)
    print("Model Evaluation")
    print("="*60)

    spark = get_spark_session()

    # Load test data if not provided
    if test_df is None:
        data_path = get_project_root() / "data" / "processed" / "featured_data"
        if not data_path.exists():
            print("No processed data found. Running full pipeline...")
            _, test_df = train_models()
        else:
            df = spark.read.parquet(str(data_path))
            _, test_df = df.randomSplit([0.8, 0.2], seed=42)

    evaluator = ModelEvaluator()

    # Evaluate Price Predictor
    print("\n[1/3] Evaluating Price Predictor...")
    try:
        price_model = PricePredictor().load()
        price_result = evaluator.evaluate_model(price_model, test_df)
        evaluator.print_summary(price_result)
    except Exception as e:
        print(f"  - Error: {e}")

    # Evaluate Demand Forecaster
    print("\n[2/3] Evaluating Demand Forecaster...")
    try:
        demand_model = DemandForecaster().load()
        demand_result = evaluator.evaluate_model(demand_model, test_df)
        evaluator.print_summary(demand_result)
    except Exception as e:
        print(f"  - Error: {e}")

    # Evaluate Spike Detector
    print("\n[3/3] Evaluating Spike Detector...")
    try:
        spike_model = SpikeDetector().load()
        spike_result = evaluator.evaluate_model(spike_model, test_df)
        evaluator.print_summary(spike_result)
    except Exception as e:
        print(f"  - Error: {e}")

    # Save results
    evaluator.save_results()

    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


def launch_dashboard():
    """Launch the Dash dashboard."""
    print("\n" + "="*60)
    print("Launching Dashboard")
    print("="*60)

    dashboard_config = CONFIG.get("dashboard", {})
    host = dashboard_config.get("host", "127.0.0.1")
    port = dashboard_config.get("port", 8050)

    print(f"\nStarting dashboard at http://{host}:{port}")
    print("Press Ctrl+C to stop\n")

    # Import and run dashboard
    from dashboard.app import app
    app.run(host=host, port=port, debug=dashboard_config.get("debug", True))


def run_all():
    """Run full pipeline, train, evaluate, then launch dashboard."""
    print("\n" + "="*60)
    print("NYISO Full Analytics Pipeline")
    print("="*60)

    # Run pipeline
    df = run_pipeline()

    # Train models
    train_df, test_df = train_models(df)

    # Evaluate
    evaluate_models(test_df)

    # Stop Spark before dashboard (dashboard doesn't need it)
    stop_spark_session()

    # Launch dashboard
    launch_dashboard()


def main():
    parser = argparse.ArgumentParser(
        description="NYISO PySpark Analytics CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py pipeline    Run data processing pipeline
  python main.py train       Train ML models
  python main.py evaluate    Evaluate trained models
  python main.py dashboard   Launch web dashboard
  python main.py all         Run everything
        """
    )

    parser.add_argument(
        "command",
        choices=["pipeline", "train", "evaluate", "dashboard", "all"],
        help="Command to run"
    )

    args = parser.parse_args()

    try:
        if args.command == "pipeline":
            run_pipeline()
            stop_spark_session()
        elif args.command == "train":
            train_models()
            stop_spark_session()
        elif args.command == "evaluate":
            evaluate_models()
            stop_spark_session()
        elif args.command == "dashboard":
            launch_dashboard()
        elif args.command == "all":
            run_all()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        stop_spark_session()
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        stop_spark_session()
        sys.exit(1)


if __name__ == "__main__":
    main()
