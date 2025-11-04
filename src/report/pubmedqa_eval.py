"""
PubmedQA evaluation class.
"""

import logging
from typing import Dict, List

from ..data import EvalResult
from .base_eval import EvalBase
from .metrics import (
    _is_answer_correct,
    calculate_accuracy,
    calculate_exact_match,
    calculate_f1,
)

logger = logging.getLogger(__name__)


class PubmedQAEval(EvalBase):
    """PubmedQA evaluation class."""

    def __init__(self, eval_methods: List[str] = None):
        # Default to all methods for PubmedQA
        if eval_methods is None:
            eval_methods = ["acc", "f1", "em"]
        super().__init__("pubmedqa", eval_methods)

    def calculate_scores(self) -> Dict[str, float]:
        """Calculate scores for PubmedQA with processed predictions."""
        if not self.results:
            logger.warning(f"No results to evaluate for {self.name}")
            return {}

        # Process predictions: split by comma and take first element
        processed_results = []
        for result in self.results:
            prediction = result.prediction
            processed_result = EvalResult(
                id=result.id,
                query=result.query,
                prompt=result.prompt,
                answer=result.answer,
                prediction=prediction,
            )
            processed_results.append(processed_result)

        # Calculate scores using processed results
        scores = {}
        for method in self.eval_methods:
            if method == "acc":
                scores["acc"] = calculate_accuracy(processed_results)
            elif method == "f1":
                scores["f1"] = calculate_f1(processed_results)
            elif method == "em":
                processed_results = []
                for result in self.results:
                    prediction = (
                        result.prediction.split(",")[0].strip() if result.prediction else ""
                    )
                    processed_result = EvalResult(
                        id=result.id,
                        query=result.query,
                        prompt=result.prompt,
                        answer=result.answer,
                        prediction=prediction,
                    )
                    processed_results.append(processed_result)
                scores["em"] = calculate_exact_match(processed_results)

        self.scores = scores
        return scores

    def _is_correct(self, result: EvalResult) -> bool:
        """Check if the prediction is correct for PubmedQA."""
        # Split prediction by comma and take the first element
        # prediction = (
        #     result.prediction.split(",")[0].strip() if result.prediction else ""
        # )
        # Create a modified result with processed prediction
        modified_result = EvalResult(
            id=result.id,
            query=result.query,
            prompt=result.prompt,
            answer=result.answer,
            prediction=result.prediction,
        )
        return _is_answer_correct(modified_result.answer, modified_result.prediction)
