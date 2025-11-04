#!/usr/bin/env python3
"""
RAGQA - Main Evaluation Script

This script provides the main entry point for RAG system evaluation.
"""

import argparse
import json
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src import SUPPORTED_DATASET, get_eval
from src.data import EvalResult
from src.logger import get_logger, set_verbose
from src.report import Runner

logger = get_logger()


def main():
    """Main function for RAG evaluation."""
    parser = argparse.ArgumentParser(
        description="PRGB - RAG System Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation with Qwen3 model
  python eval.py --model-name "Qwen3" --model-path "/path/to/model" --data-path "data/test.jsonl"
  
  # Evaluation with custom noise configuration
  python eval.py --model-name "Qwen3" --noise-config '{"noise_doc_level1":4,"noise_doc_level2":4,"noise_doc_level3":1}'
  
  # Batch evaluation with specific parameters
  python eval.py --model-name "Qwen3" --batch-size 32 --temperature 0.8 --shuffle True
        """,
    )

    # Model configuration
    parser.add_argument("--api-key", type=str, default=None, help="api key of chatgpt")
    parser.add_argument(
        "--total_doc_number",
        type=int,
        default=30,
        help="total doc number when infering",
    )
    parser.add_argument(
        "--model-name", type=str, default="Qwen3", help="Name of the model to evaluate"
    )
    parser.add_argument(
        "--inference-mode",
        type=bool,
        default=False,
        help="whether inference model or not",
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model or API url"
    )

    # Data configuration
    parser.add_argument(
        "--eval-dataset",
        nargs="+",
        choices=SUPPORTED_DATASET,
        default=SUPPORTED_DATASET,
        help="Specify which datasets to evaluate. Must be one or more of 'a', 'b', 'c'. (e.g., --eval-dataset a c)",
    )

    # Output configuration
    parser.add_argument(
        "--output-path",
        type=str,
        default="./results",
        help="Output directory for results",
    )

    # Evaluation parameters
    parser.add_argument(
        "--shuffle", type=bool, default=True, help="Whether to shuffle the data"
    )
    parser.add_argument(
        "--batch-size", type=int, default=5, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for text generation"
    )
    parser.add_argument(
        "--custom_config",
        type=str,
        default=None,
        help="custom prompt config path",
    )
    # Additional options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        set_verbose(True)

    # Create output directory if it doesn't exist
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting evaluation with model: {args.model_name}")
    logger.info(f"Eval Dataset: {args.eval_dataset}")
    logger.info(f"Output path: {args.output_path}")

    print("🚀 RAGQA Evaluation System Example")
    print("=" * 50)

    # Create runner
    runner = Runner(output_dir="reports")

    # Create sample results
    # sample_results = create_sample_results()
    results = get_eval(args)

    print("\n📊 Running evaluation for all datasets...")

    # Run evaluation for all datasets
    runner.run_all(results)

    # Print summary
    runner.print_summary()

    print("\n📝 Generating HTML report...")

    # Generate HTML report
    html_path = runner.generate_html_report()
    print(f"HTML report generated: {html_path}")

    print("\n💾 Saving JSON results...")

    # Save JSON results
    json_path = runner.save_json_results()
    print(f"JSON results saved: {json_path}")

    print("\n✅ Example completed successfully!")
    print(f"📁 Check the 'reports' directory for output files")

    # try:
    #     # Run evaluation
    #     get_eval(args)
    #     logger.info("Evaluation completed successfully!")

    # except Exception as e:
    #     logger.error(f"Evaluation failed: {e}")
    #     sys.exit(1)


if __name__ == "__main__":
    main()
