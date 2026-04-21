import argparse
import sys
import logging
from src.interface import AstroAnomalyEngine

def main():
    parser = argparse.ArgumentParser(description="AstroAnomaly CLI - High-Performance Astronomical Anomaly Discovery")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train SOTA experts on astronomical data")
    train_parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    train_parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    train_parser.add_argument("--models", type=str, default="transformer,tranad,timesnet", help="Comma-separated model names")

    # Discover command
    discover_parser = subparsers.add_parser("discover", help="Run the discovery pipeline on a dataset")
    discover_parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    discover_parser.add_argument("--weights", type=str, default="latest", help="Suffix for weights to load")
    discover_parser.add_argument("--output", type=str, default="discovery_report.json", help="Filename for the report")

    args = parser.parse_args()

    engine = AstroAnomalyEngine()

    if args.command == "train":
        model_list = args.models.split(",")
        engine.train(data_path=args.data, model_types=model_list, epochs=args.epochs)
    elif args.command == "discover":
        results = engine.discover(data_path=args.data, weights_suffix=args.weights)
        engine.save_report(results, filename=args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
