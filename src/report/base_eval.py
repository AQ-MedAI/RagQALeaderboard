"""
Base evaluation classes and shared utilities.
"""

import json
import os
from abc import ABC
from pathlib import Path
from typing import Dict, List

from ..data import EvalResult
from ..logger import get_logger
from .metrics import (
    _is_answer_correct,
    calculate_accuracy,
    calculate_exact_match,
    calculate_f1,
)

logger = get_logger()

# Dataset name to type mapping used by all evals
NAME_TO_TYPE = {
    "triviaqa": "singlehopQA",
    "popqa": "singlehopQA",
    "hotpotqa": "multihopQA",
    "musiqueqa": "multihopQA",
    "2wiki": "multihopQA",  
    "pubmedqa": "biomedicalQA",
}


class EvalBase(ABC):
    """Base evaluation class for all evaluation types."""

    def __init__(self, name: str, eval_methods: List[str] = None):
        """
        Initialize the evaluation base class.

        Args:
            name: Name of the dataset
            eval_methods: List of evaluation methods to use (acc, f1, em)
        """
        self.name = name
        self.eval_type = NAME_TO_TYPE.get(name, "unknown")
        self.eval_methods = eval_methods or ["acc", "f1", "em"]
        self.results: List[EvalResult] = []
        self.scores: Dict[str, float] = {}
        self.calculate_score_state: bool = False

    def add_result(self, result: EvalResult):
        """Add an evaluation result."""
        # breakpoint()
        if isinstance(result.answer, list):
            result.set_label(
                any(_is_answer_correct(ans, result.prediction) for ans in result.answer)
            )
        else:
            result.set_label(_is_answer_correct(result.answer, result.prediction))
        self.results.append(result)

    def add_results(self, results: List[EvalResult]):
        """Add multiple evaluation results."""
        for res in results:
            self.add_result(res)

    def calculate_scores(self) -> Dict[str, float]:
        """Calculate scores for all evaluation methods."""
        if not self.results:
            logger.warning(f"No results to evaluate for {self.name}")
            return {}

        scores = {}
        for method in self.eval_methods:
            if method == "acc":
                scores["acc"] = calculate_accuracy(self.results)
            elif method == "f1":
                scores["f1"] = calculate_f1(self.results)
            elif method == "em":
                scores["em"] = calculate_exact_match(self.results)

        self.scores = scores
        self.calculate_score_state = True
        return scores

    def get_correct_answers(self) -> List[EvalResult]:
        """Get all correct answers."""
        return [result for result in self.results if self._is_correct(result)]

    def get_incorrect_answers(self) -> List[EvalResult]:
        """Get all incorrect answers."""
        return [result for result in self.results if not self._is_correct(result)]

    def get_total_score(self) -> float:
        """Get the average score across all evaluation methods."""
        if not self.scores:
            self.calculate_scores()

        if not self.scores:
            return 0.0

        return sum(self.scores.values()) / len(self.scores)

    def get_error_ids(self) -> List[str]:
        """Get IDs of all incorrect answers."""
        return [result.id for result in self.get_incorrect_answers()]

    def get_corresponding_datapreprocess_type(self):
        """Get corresponding data preprocess type."""
        return self.name

    def save_results_score(self, file_path: str):
        """Save evaluation results to file."""
        scores_dict = {}
        for method in self.eval_methods:
            if method in self.scores:
                scores_dict[method] = self.scores[method]
            else:
                scores_dict[method] = None

        output_data = {
            "name": self.name,
            "class": self.eval_type,
            "scores": scores_dict,
            "error_id": self.get_error_ids(),
        }

        with open(file_path, "w", encoding="utf-8") as f:
            import json

            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation results saved to {file_path}")

    def save_results(
        self, output_path: str, append: bool = False, error_only: bool = False
    ):
        if not self.results:
            logger.error("No results to save")
            return

        mode = "a" if append else "w"

        # Filter results if error_only is True
        results_to_save = self.get_incorrect_results() if error_only else self.results

        if not results_to_save:
            logger.warning("No results to save after filtering")
            return

        try:
            with open(output_path, mode, encoding="utf-8") as f:
                # Use tqdm to show progress
                for result in results_to_save:
                    # Create data dictionary
                    data = {
                        "id": result.id,
                        "query": result.query,
                        "prompt": result.prompt,
                        "answer": result.answer,
                        "prediction": result.prediction,
                        "label": result.label,
                    }

                    try:
                        # Write one line
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    except Exception as e:
                        logger.error(f"Error writing line: {e}")
                        logger.error(f"Problematic data: {data}")
                        continue

        except Exception as e:
            logger.error(f"Error opening file {output_path}: {e}")
            raise

        logger.info(
            f"Successfully saved {len(results_to_save)} items to \033[31m{output_path}\033[0m"
        )

    @classmethod
    def load_from_jsonl(
        cls, file_path: str, name: str = None, eval_methods: List[str] = None
    ) -> "EvalBase":
        """
        Load evaluation results from a JSONL file saved by save_results method.

        Args:
            file_path: Path to the JSONL file
            name: Name of the dataset (if None, will be extracted from filename)
            eval_methods: List of evaluation methods to use (if None, will use default)

        Returns:
            EvalBase instance with loaded results

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Extract name from filename if not provided
        if name is None:
            name = Path(file_path).stem

        # Create instance
        instance = cls(name=name, eval_methods=eval_methods)

        loaded_results = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue

                    try:
                        data = json.loads(line)

                        # Validate required fields
                        required_fields = ["id", "query", "answer", "prediction"]
                        missing_fields = [
                            field for field in required_fields if field not in data
                        ]
                        if missing_fields:
                            logger.warning(
                                f"Line {line_num}: Missing required fields: {missing_fields}"
                            )
                            continue

                        # Create EvalResult object
                        result = EvalResult(
                            id=data["id"],
                            query=data["query"],
                            prompt=data.get("prompt", ""),  # prompt might be optional
                            answer=data["answer"],
                            prediction=data["prediction"],
                        )

                        # Set label if it exists in the data
                        if "label" in data and data["label"] is not None:
                            result.set_label(data["label"])

                        loaded_results.append(result)

                    except json.JSONDecodeError as e:
                        logger.error(f"Line {line_num}: Invalid JSON format - {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Line {line_num}: Error processing line - {e}")
                        continue

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise

        if not loaded_results:
            logger.warning(f"No valid results loaded from {file_path}")
        else:
            # Add all results to the instance
            # Note: add_results will automatically calculate labels if not already set
            instance.add_results(loaded_results)
            logger.info(
                f"Successfully loaded {len(loaded_results)} results from {file_path}"
            )

        return instance

    def print_summary(self):
        """Print evaluation summary."""
        if not self.calculate_score_state:
            logger.info(
                "Running calculate_scores() first as scores have not been calculated."
            )
            self.calculate_scores()
        print(f"\n=== Evaluation Summary for {self.name} ===")
        print(f"Dataset Type: {self.eval_type}")
        print(f"Total Samples: {len(self.results)}")
        print(f"Correct Answers: {len(self.get_correct_answers())}")
        print(f"Incorrect Answers: {len(self.get_incorrect_answers())}")

        if self.scores:
            print(f"Scores:")
            for method, score in self.scores.items():
                print(f"  {method.upper()}: {score:.4f}")
            print(f"Total Score: {self.get_total_score():.4f}")

        if self.get_error_ids():
            print(
                f"Error IDs: {self.get_error_ids()[:5]}{'...' if len(self.get_error_ids()) > 5 else ''}"
            )

    def _is_correct(self, result: EvalResult) -> bool:
        """Check if the prediction is correct (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement _is_correct method")
