#!/usr/bin/env python3
import argparse
import sys

from pipeline.config import PipelineConfig
from pipeline.main import CausalAnalysisPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Causal Analysis Pipeline for Conversational Data",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=10,
        help="Number of conversations to analyse (default: 10)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Analyse all conversations",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device for inference: cpu, cuda, or auto (default: auto)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Interactive follow-up query",
    )
    args = parser.parse_args()

    config = PipelineConfig(device=args.device)
    pipe = CausalAnalysisPipeline(config)

    print("Loading data...")
    try:
        pipe.load_data()
    except FileNotFoundError as e:
        print(f"Error: Data file not found: {e}")
        print(f"Expected CSV at: {config.data.csv_path}")
        print(f"Expected JSON at: {config.data.json_path}")
        sys.exit(1)
    print(f"Loaded {len(pipe.records)} conversation records.\n")

    # Analyse conversations
    max_records = None if args.all else args.max_records
    print(f"Analysing {'all' if args.all else args.max_records} conversations...\n")

    # Show detailed analysis for first record
    if pipe.records:
        result = pipe.analyse_conversation(pipe.records[0])
        print(result["explanation"])
        print()

        # Show causal DAG
        print("Causal DAG (DOT format):")
        print(pipe.causal_dag.to_dot())
        print()

        # Counterfactual
        cf = result["causal"]["counterfactual"]
        print("Counterfactual analysis (do(delay=0)):")
        for var, val in cf.items():
            print(f"  {var}: {val:.4f}")
        print()

    # Batch analysis summary
    results = pipe.analyse_all(max_records=max_records)
    print(f"\n{'='*60}")
    print(f"BATCH ANALYSIS SUMMARY ({len(results)} conversations)")
    print(f"{'='*60}")
    for r in results:
        print(
            f"  {r['transcript_id']}: "
            f"outcome={r['outcome']}, "
            f"ATE={r['ate']:.4f}, "
            f"chain={' → '.join(r['causal_chain'])}"
        )

    # Interactive query
    if args.query:
        print(f"\nInteractive Query: {args.query}")
        print(pipe.interactive_query(args.query))


if __name__ == "__main__":
    main()
