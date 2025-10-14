"""
PopQA evaluation class.
"""

from typing import List

from ..data import EvalResult
from .base_eval import EvalBase
from .metrics import _is_answer_correct


class PopQAEval(EvalBase):
    """PopQA evaluation class."""

    def __init__(self, eval_methods: List[str] = None):
        # Default to accuracy only for PopQA
        if eval_methods is None:
            eval_methods = ["acc","f1","em"]
        super().__init__("popqa", eval_methods)

    def _is_correct(self, result: EvalResult) -> bool:
        """Check if the prediction is correct for PopQA."""
        if isinstance(result.answer, list):
            return any(
                _is_answer_correct(ans, result.prediction) for ans in result.answer
            )
        else:
            return _is_answer_correct(result.answer, result.prediction)
