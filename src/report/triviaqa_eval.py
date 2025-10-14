"""
MusiqueQA evaluation class.
"""

from typing import List

from ..data import EvalResult
from .base_eval import EvalBase
from .metrics import _is_answer_correct


class TriviaQAEval(EvalBase):
    """MusiqueQA evaluation class."""

    def __init__(self, eval_methods: List[str] = None):
        # Default to accuracy and F1 for MusiqueQA
        if eval_methods is None:
            eval_methods = ["acc","f1","em"]
        super().__init__("triviaqa", eval_methods)

    def _is_correct(self, result: EvalResult) -> bool:
        """Check if the prediction is correct for MusiqueQA."""
        if isinstance(result.answer, list):
            return any(
                _is_answer_correct(ans, result.prediction) for ans in result.answer
            )
        else:
            return _is_answer_correct(result.answer, result.prediction)
