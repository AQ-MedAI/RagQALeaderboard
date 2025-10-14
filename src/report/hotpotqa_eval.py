"""
HotpotQA evaluation class.
"""

from typing import List

from ..data import EvalResult
from .base_eval import EvalBase
from .metrics import _is_answer_correct


class HotpotQAEval(EvalBase):
    """HotpotQA evaluation class."""

    def __init__(self, eval_methods: List[str] = None):
        # Default to all methods, but can be customized
        if eval_methods is None:
            eval_methods = ["acc", "f1", "em"]
        super().__init__("hotpotqa", eval_methods)

    def _is_correct(self, result: EvalResult) -> bool:
        """Check if the prediction is correct for HotpotQA."""
        return _is_answer_correct(result.answer, result.prediction)
