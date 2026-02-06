#!/usr/bin/env python3
import argparse
import logging
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
    parser.add_argument(
        "--train-encoder", action="store_true",
        help="Train only the encoder",
    )
    parser.add_argument(
        "--train-gnn", action="store_true",
        help="Train only the GNN",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from existing checkpoints",
    )
    parser.add_argument(
        "--force-train", action="store_true",
        help="Force retraining even if checkpoints exist",
    )
    parser.add_argument(
        "--skip-tests", action="store_true",
        help="Skip test-set evaluation after training",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = PipelineConfig(device=args.device)

    # Derive skip flags from --train-encoder / --train-gnn selectors
    skip_encoder = False
    skip_gnn = False
    if args.train_encoder and not args.train_gnn:
        skip_gnn = True
    elif args.train_gnn and not args.train_encoder:
        skip_encoder = True

    try:
        history = train_all(
            config=config,
            checkpoint_dir=args.checkpoint_dir,
            encoder_epochs=args.encoder_epochs,
            gnn_epochs=args.gnn_epochs,
            verbose=True,
            force_train=args.force_train,
            skip_encoder=skip_encoder,
            skip_gnn=skip_gnn,
            resume=args.resume,
            skip_tests=args.skip_tests,
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