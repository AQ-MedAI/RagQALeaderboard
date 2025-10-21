"""
Evaluation metrics for different types of questions.
"""

import re
from typing import List, Tuple

from ..data import EvalResult

blank_ct = 0

def calculate_accuracy(results: List[EvalResult]) -> float:
    """
    Calculate accuracy using regex pattern matching.

    Args:
        results: List of EvalResult objects

    Returns:
        float: Accuracy score between 0 and 1
    """
    if not results:
        return 0.0

    correct_count = 0
    for result in results:
        if isinstance(result.answer, list):
            if any(_is_answer_correct(ans, result.prediction) for ans in result.answer):
                correct_count += 1
        else:
            if _is_answer_correct(result.answer, result.prediction):
                correct_count += 1

    return correct_count / len(results)


def calculate_f1(results: List[EvalResult]) -> float:
    """
    Calculate F1 score.

    Args:
        results: List of EvalResult objects

    Returns:
        float: F1 score between 0 and 1
    """
    if not results:
        return 0.0

    total_f1 = 0.0
    for result in results:
        if isinstance(result.answer, list):
            temp_f1 = 0.0
            for ans in result.answer:
                if "</think>" in result.prediction:
                    pred_clean = result.prediction.split("</think>")[1]
                    temp_f1 = max(temp_f1, _calculate_single_f1(ans, pred_clean))
                else:
                    pred_clean = result.prediction
            total_f1 += temp_f1
        else:
            total_f1 += _calculate_single_f1(result.answer, result.prediction)

    return total_f1 / len(results)


def calculate_exact_match(results: List[EvalResult]) -> float:
    """
    Calculate exact match score.

    Args:
        results: List of EvalResult objects

    Returns:
        float: Exact match score between 0 and 1
    """
    if not results:
        return 0.0

    exact_matches = 0
    for result in results:
        if isinstance(result.answer, list):
            if any(_is_exact_match(ans, result.prediction) for ans in result.answer):
                exact_matches += 1
        else:
            if _is_exact_match(result.answer, result.prediction):
                exact_matches += 1

    return exact_matches / len(results)


def _is_answer_correct(ground_truth: str, prediction: str) -> bool:
    """
    Check if the prediction is correct using regex pattern matching.

    Args:
        ground_truth: Ground truth answer
        prediction: Model prediction

    Returns:
        bool: True if prediction is correct, False otherwise
    """
    # if "4,000" in ground_truth:
    #     breakpoint()
    global blank_ct
    if not ground_truth or not prediction:
        return False

    # Clean and normalize answers
    gt_clean = ground_truth.lower().strip()
    pred_clean = prediction.lower().strip().replace("**", "")
    pred_clean = re.sub("\*(.*?)\*", r"\1", pred_clean)

    if "</think>" in pred_clean:
        pred_clean = pred_clean.split("</think>")[1]
    
    else:
        if "<think>" in pred_clean:
            return False
    # Check exact match first
    if gt_clean == pred_clean:
        return True

    if gt_clean.strip()=="":
        breakpoint()
        return False
    # Use regex pattern matching for partial matches
    # Escape special characters in ground truth
    # escaped_gt = re.escape(gt_clean)
    # Look for the ground truth as a complete word/phrase
    # pattern = r"(\W|^)(" + escaped_gt + r")(\W|$)"

    # match_result = re.search(pattern, pred_clean, re.S)
    # return match_result is not None
    return gt_clean in pred_clean


def _is_exact_match(ground_truth: str, prediction: str) -> bool:
    """
    Check if the prediction exactly matches the ground truth.

    Args:
        ground_truth: Ground truth answer
        prediction: Model prediction

    Returns:
        bool: True if exact match, False otherwise
    """
    if not ground_truth or not prediction:
        return False

    return ground_truth.lower().strip() == prediction.lower().strip()


def _calculate_single_f1(ground_truth: str, prediction: str) -> float:
    """
    Calculate F1 score for a single answer-prediction pair.

    Args:
        ground_truth: Ground truth answer
        prediction: Model prediction

    Returns:
        float: F1 score between 0 and 1
    """
    if not ground_truth or not prediction:
        return 0.0

    # Check exact match first
    if _is_exact_match(ground_truth, prediction):
        return 1.0

    # Word-based F1 calculation
    gt_words = set(re.findall(r"\w+", ground_truth.lower()))
    pred_words = set(re.findall(r"\w+", prediction.lower()))

    if not gt_words or not pred_words:
        return 0.0

    intersection = gt_words.intersection(pred_words)
    precision = len(intersection) / len(pred_words) if pred_words else 0
    recall = len(intersection) / len(gt_words) if gt_words else 0

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)
