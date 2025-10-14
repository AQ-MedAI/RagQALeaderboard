#!/usr/bin/env python3
"""
Example usage of the RAGQA evaluation system.
"""

from src.data import EvalResult
from src.report import Runner


def create_sample_results():
    """Create sample evaluation results for testing."""
    return {
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
            EvalResult(
                id="hotpot_test-3",
                query="What is 2+2?",
                prompt="System: Answer the question.\nUser: What is 2+2?",
                answer="4",
                prediction="5",
            ),
        ],
        "popqa": [
            EvalResult(
                id="pop_test-1",
                query="What is the largest planet?",
                prompt="System: Answer the question.\nUser: What is the largest planet?",
                answer="Jupiter",
                prediction="Jupiter",
            ),
            EvalResult(
                id="pop_test-2",
                query="What year did World War II end?",
                prompt="System: Answer the question.\nUser: What year did World War II end?",
                answer="1945",
                prediction="1945",
            ),
        ],
        "musiqueqa": [
            EvalResult(
                id="musique_test-1",
                query="Who composed Beethoven's 5th Symphony?",
                prompt="System: Answer the question.\nUser: Who composed Beethoven's 5th Symphony?",
                answer="Ludwig van Beethoven",
                prediction="Beethoven",
            )
        ],
        "pubmedqa": [
            EvalResult(
                id="pubmed_test-1",
                query="Is smoking harmful to health?",
                prompt="System: Answer the question.\nUser: Is smoking harmful to health?",
                answer="Yes",
                prediction="Yes, confirmed, harmful",
            ),
            EvalResult(
                id="pubmed_test-2",
                query="Does exercise improve mental health?",
                prompt="System: Answer the question.\nUser: Does exercise improve mental health?",
                answer="Yes",
                prediction="No, not effective",
            ),
        ],
    }


def create_results_from_jsonl(save_dir):
    import glob
    import os

    from src.report import EvalBase

    jsonl_files = glob.glob(os.path.join(save_dir, "*.jsonl"))
    res = {}
    for jsonl_file in jsonl_files:
        if "hotpotqa" in jsonl_file:
            res["hotpotqa"] = EvalBase.load_from_jsonl(jsonl_file).results
        elif "popqa" in jsonl_file:
            res["popqa"] = EvalBase.load_from_jsonl(jsonl_file).results
        elif "musiqueqa" in jsonl_file:
            res["musiqueqa"] = EvalBase.load_from_jsonl(jsonl_file).results
        elif "triviaqa" in jsonl_file:
            res["triviaqa"] = EvalBase.load_from_jsonl(jsonl_file).results
        elif "2wiki" in jsonl_file:
            res["2wiki"] = EvalBase.load_from_jsonl(jsonl_file).results
        elif "nq" in jsonl_file:
            res["nq"] = EvalBase.load_from_jsonl(jsonl_file).results
        elif "pubmedqa" in jsonl_file:
            res["pubmedqa"] = EvalBase.load_from_jsonl(jsonl_file).results
    return res


def main(args):
    """Main example function."""
    print("🚀 RAGQA Evaluation System Example")
    print("=" * 50)

    # Create runner
    runner = Runner(output_dir="reports")

    # Create sample results
    # sample_results = create_sample_results()
    sample_results = create_results_from_jsonl(args.result_dir)

    print("\n📊 Running evaluation for all datasets...")

    # Run evaluation for all datasets
    runner.run_all(sample_results)

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PRGB - RAG System Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get Report From existing results
  python get_report.py --result-dir ./results
        """,
    )

    # Model configuration
    parser.add_argument(
        "--result-dir", type=str, default="./results", help="api key of chatgpt"
    )
    args = parser.parse_args()
    main(args)
