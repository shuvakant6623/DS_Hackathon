#!/usr/bin/env python3
import argparse
import sys

from pipeline.config import PipelineConfig
from pipeline.train import train_all


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train models for the Causal Analysis Pipeline",
    )
    parser.add_argument(
        "--encoder-epochs", type=int, default=None,
        help="Override encoder training epochs",
    )
    parser.add_argument(
        "--gnn-epochs", type=int, default=None,
        help="Override GNN training epochs",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Directory for saving model checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device for training: cpu, cuda, or auto (default: auto)",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate a technical report after training",
    )
    args = parser.parse_args()

    config = PipelineConfig(device=args.device)

    try:
        history = train_all(
            config=config,
            checkpoint_dir=args.checkpoint_dir,
            encoder_epochs=args.encoder_epochs,
            gnn_epochs=args.gnn_epochs,
            verbose=True,
        )
    except FileNotFoundError as e:
        print(f"Error: Data file not found: {e}")
        print(f"Expected CSV at: {config.data.csv_path}")
        print(f"Expected JSON at: {config.data.json_path}")
        sys.exit(1)

    if args.report:
        from pipeline.report import generate_report
        report = generate_report(training_history=history)
        print(f"\nTechnical report saved to outputs/technical_report.md")


if __name__ == "__main__":
    main()