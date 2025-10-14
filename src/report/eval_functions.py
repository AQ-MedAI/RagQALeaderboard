from typing import List, Optional

from ..data import EvalResult
from .base_eval import EvalBase
from .hotpotqa_eval import HotpotQAEval
from .musiqueqa_eval import MusiqueQAEval
from .popqa_eval import PopQAEval
from .pubmedqa_eval import PubmedQAEval
from .triviaqa_eval import TriviaQAEval
from .twowiki_eval import TwoWIKIEval

# Mapping from dataset names to evaluation classes
NAME_TO_EVAL_CLASS = {
    "hotpotqa": HotpotQAEval,
    "popqa": PopQAEval,
    "musiqueqa": MusiqueQAEval,
    "pubmedqa": PubmedQAEval,
    "2wiki": TwoWIKIEval,
    "triviaqa": TriviaQAEval,
}


def get_eval_class(dataset_name: str, eval_methods: List[str] = None) -> EvalBase:
    """Get the appropriate evaluation class for a dataset."""
    if dataset_name not in NAME_TO_EVAL_CLASS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    eval_class = NAME_TO_EVAL_CLASS[dataset_name]
    return eval_class(eval_methods)


def evaluate_dataset(
    dataset_name: str,
    results: List[EvalResult],
    output_path: Optional[str] = None,
    eval_methods: Optional[List[str]] = None,
) -> EvalBase:
    """Evaluate a dataset and optionally save results.

    Returns the eval instance populated with scores and results.
    """
    evaluator = get_eval_class(dataset_name, eval_methods)
    evaluator.add_results(results)
    evaluator.calculate_scores()

    if output_path:
        evaluator.save_results(output_path)

    return evaluator
