#!/usr/bin/env python3
import argparse
import sys

from pipeline.config import PipelineConfig
from pipeline.evaluate import evaluate_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Causal Analysis Pipeline",
    )
    parser.add_argument(
        "--max-records", type=int, default=None,
        help="Limit evaluation to first N records (default: all)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device for evaluation: cpu, cuda, or auto (default: auto)",
    )

    parser.add_argument(
        "--report", action="store_true",
        help="Generate a technical report with evaluation results",
    )
    args = parser.parse_args()

    config = PipelineConfig(device=args.device)

    try:
        results = evaluate_pipeline(
            config=config,
            max_records=args.max_records,
            verbose=True,
        )
    except FileNotFoundError as e:
        print(f"Error: Data file not found: {e}")
        print(f"Expected CSV at: {config.data.csv_path}")
        print(f"Expected JSON at: {config.data.json_path}")
        sys.exit(1)

    if args.report:
        from pipeline.report import generate_report
        report = generate_report(evaluation_results=results)
        print(f"\nTechnical report saved to outputs/technical_report.md")


if __name__ == "__main__":
    main()