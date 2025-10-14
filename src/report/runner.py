"""
Unified execution entry point for RAGQA evaluation.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..data import EvalResult
from ..logger import get_logger
from .eval_functions import NAME_TO_EVAL_CLASS, evaluate_dataset, get_eval_class
from .html_reporter import HTMLReporter

logger = get_logger()


class Runner:
    """Main runner class for RAGQA evaluation."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        self.html_reporter = HTMLReporter(output_dir)
        self.results = {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def run_all(
        self,
        results_data: Dict[str, List[EvalResult]],
        eval_methods: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Run evaluation for all datasets.

        Args:
            results_data: Dictionary mapping dataset names to EvalResult lists
            eval_methods: Optional dictionary mapping dataset names to evaluation methods

        Returns:
            Dict containing evaluation results for all datasets
        """
        logger.info("🚀 Starting evaluation for all datasets...")

        if eval_methods is None:
            eval_methods = {}

        for dataset_name in results_data.keys():
            if dataset_name in NAME_TO_EVAL_CLASS:
                logger.info(f"📊 Evaluating {dataset_name}...")
                self._evaluate_single_dataset(
                    dataset_name,
                    results_data[dataset_name],
                    eval_methods.get(dataset_name),
                )
            else:
                logger.warning(f"⚠️  Unknown dataset: {dataset_name}")

        logger.info("✅ All datasets evaluated successfully!")
        return self.results

    def run_single(
        self,
        dataset_name: str,
        results: List[EvalResult],
        eval_methods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run evaluation for a single dataset.

        Args:
            dataset_name: Name of the dataset to evaluate
            results: List of EvalResult objects
            eval_methods: Optional list of evaluation methods to use

        Returns:
            Dict containing evaluation results for the dataset
        """
        if dataset_name not in NAME_TO_EVAL_CLASS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        logger.info(f"📊 Evaluating single dataset: {dataset_name}")

        self._evaluate_single_dataset(dataset_name, results, eval_methods)

        logger.info(f"✅ Dataset {dataset_name} evaluated successfully!")
        return self.results.get(dataset_name, {})

    def _evaluate_single_dataset(
        self,
        dataset_name: str,
        results: List[EvalResult],
        eval_methods: Optional[List[str]] = None,
    ):
        """Evaluate a single dataset and store results."""

        # Evaluate the dataset
        eval_result = evaluate_dataset(
            dataset_name,
            results,
            output_path=None,  # Don't save individual files
            eval_methods=eval_methods,
        )

        # Save results to our results dictionary
        self.results[dataset_name] = {
            "name": dataset_name,
            "class": eval_result.eval_type,
            "scores": eval_result.scores,
            "error_id": eval_result.get_error_ids(),
            # Include rich error details for interactive report
            "error_details": {
                er.id: {
                    "query": er.query,
                    "ground_truth": er.answer,
                    "prediction": er.prediction,
                }
                for er in eval_result.get_incorrect_answers()
            },
            "sample_count": len(results),
        }

    def generate_html_report(self, filename: str = None) -> str:
        """
        Generate HTML report from evaluation results.

        Args:
            filename: Optional filename for the report

        Returns:
            str: Path to the generated HTML file
        """
        if not self.results:
            logger.warning("No evaluation results to report. Run evaluation first.")
            return ""
        logger.info("📝 Generating HTML report...")
        report_path = self.html_reporter.generate_report(self.results, filename)
        logger.info(f"📄 HTML report generated: {report_path}")
        return report_path

    def save_json_results(self, filename: str = None) -> str:
        """
        Save evaluation results to JSON file.

        Args:
            filename: Optional filename for the JSON file

        Returns:
            str: Path to the saved JSON file
        """
        if not self.results:
            logger.warning("No evaluation results to save. Run evaluation first.")
            return ""

        if filename is None:
            filename = "evaluation_results.json"

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 JSON results saved: {filepath}")
        return filepath

    def make(self, target: str = "all", **kwargs):
        """
        Make-style interface for running evaluations.

        Args:
            target: Target to build ("all" or specific dataset name)
            **kwargs: Additional arguments passed to evaluation methods
        """
        if target == "all":
            logger.info("🎯 Building all datasets...")
            # This would typically load results from somewhere
            # For now, we'll use demo data
            demo_results = self._load_demo_results()
            self.run_all(demo_results)
        elif target in NAME_TO_EVAL_CLASS:
            logger.info(f"🎯 Building dataset: {target}")
            demo_results = self._load_demo_results()
            if target in demo_results:
                self.run_single(target, demo_results[target])
            else:
                logger.error(f"No results found for dataset: {target}")
        else:
            logger.error(f"Unknown target: {target}")
            logger.info(
                f"Available targets: all, {', '.join(NAME_TO_EVAL_CLASS.keys())}"
            )

    def _load_demo_results(self) -> Dict[str, List[EvalResult]]:
        """Load demo results for testing purposes."""
        # This is a placeholder - in real usage, you'd load actual results
        from ..data import EvalResult

        demo_data = {
            "hotpotqa": [
                EvalResult(
                    id="hotpot_test-1",
                    query="What is the capital of France?",
                    prompt="System: Answer the question.\nUser: What is the capital of France?",
                    answer="Paris",
                    prediction="Paris",
                ),
                EvalResult(
                    id="hotpot_test-2",
                    query="Who wrote Romeo and Juliet?",
                    prompt="System: Answer the question.\nUser: Who wrote Romeo and Juliet?",
                    answer="William Shakespeare",
                    prediction="Shakespeare",
                ),
            ],
            "popqa": [
                EvalResult(
                    id="pop_test-1",
                    query="What is 2+2?",
                    prompt="System: Answer the question.\nUser: What is 2+2?",
                    answer="4",
                    prediction="4",
                )
            ],
        }

        return demo_data

    def print_summary(self):
        """Print a summary of evaluation results."""
        if not self.results:
            print("No evaluation results available.")
            return

        print("\n" + "=" * 60)
        print("📊 RAGQA EVALUATION SUMMARY")
        print("=" * 60)

        for dataset_name, data in self.results.items():
            print(f"\n🔍 {dataset_name.upper()} ({data['class']})")
            print("-" * 40)

            if "scores" in data:
                for metric, score in data["scores"].items():
                    if score is not None and score != "":
                        print(f"  {metric.upper()}: {score:.3f}")

            if "error_id" in data and data["error_id"]:
                print(f"  Errors: {len(data['error_id'])}")
                if len(data["error_id"]) <= 3:
                    print(f"    {', '.join(data['error_id'])}")
                else:
                    print(
                        f"    {', '.join(data['error_id'][:3])}... and {len(data['error_id']) - 3} more"
                    )

        print("\n" + "=" * 60)


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="RAGQA Evaluation Runner")
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        help="Target to build (all or specific dataset name)",
    )
    parser.add_argument(
        "--output-dir", default="reports", help="Output directory for reports"
    )
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("--json", action="store_true", help="Save results to JSON file")

    args = parser.parse_args()

    # Create runner
    runner = Runner(output_dir=args.output_dir)

    try:
        # Run evaluation
        if args.target == "all":
            demo_results = runner._load_demo_results()
            runner.run_all(demo_results)
        else:
            demo_results = runner._load_demo_results()
            if args.target in demo_results:
                runner.run_single(args.target, demo_results[args.target])
            else:
                logger.error(f"Unknown target: {args.target}")
                return

        # Print summary
        runner.print_summary()

        # Generate outputs
        if args.html:
            runner.generate_html_report()

        if args.json:
            runner.save_json_results()

        logger.info("🎉 Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
