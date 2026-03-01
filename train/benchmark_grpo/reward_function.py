"""
Reward function for GRPO training on merged benchmark datasets.

Handles both math answers (compared via mathruler grade_answer) and
multiple-choice / text answers (compared via exact string match after
normalization).

Computes a combined reward from:
  - Accuracy (0 or 1): Does the model's boxed answer match ground truth?
  - Format (0, 0.5, or 1): Does the output contain \\boxed{} with reasoning?
  - Process reward (0 to 1): Quality of the reasoning steps.

The accuracy reward dominates (80%), with format (5%) and process (15%)
providing additional learning signal even when the answer is wrong.
"""

import re
from typing import Dict, List

from mathruler.grader import extract_boxed_content, grade_answer


# ============================================================================
# Answer Extraction & Comparison
# ============================================================================

def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison."""
    if answer is None:
        return ""
    answer = str(answer).strip()
    # Remove surrounding $ signs and whitespace
    answer = answer.strip("$").strip()
    # Normalize whitespace
    answer = " ".join(answer.split())
    return answer


def is_multichoice_answer(ground_truth: str) -> bool:
    """Check if the ground truth looks like a multiple choice answer (single letter A-Z)."""
    gt = ground_truth.strip()
    return len(gt) == 1 and gt.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def extract_multichoice_from_boxed(predict: str) -> str:
    """Extract a multiple choice letter from \\boxed{...} content."""
    content = extract_boxed_content(predict)
    if content is None:
        return None
    content = str(content).strip()
    # Try to find a single letter
    match = re.search(r'\b([A-Z])\b', content.upper())
    if match:
        return match.group(1)
    # If the entire content is a single letter
    if len(content) == 1 and content.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return content.upper()
    return content


def accuracy_reward(predict: str, ground_truth: str) -> float:
    """Check if the model's \\boxed{} answer matches the ground truth.

    Handles both mathematical expressions (via grade_answer) and
    multiple-choice / exact text answers.
    """
    gt = normalize_answer(ground_truth)

    if is_multichoice_answer(gt):
        # Multiple choice: extract letter from boxed content
        pred_letter = extract_multichoice_from_boxed(predict)
        if pred_letter is None:
            return 0.0
        return 1.0 if pred_letter.upper() == gt.upper() else 0.0

    # Math / text answer: use mathruler grader
    answer = extract_boxed_content(predict)
    try:
        return 1.0 if grade_answer(answer, ground_truth) else 0.0
    except Exception:
        # Fallback: exact normalized string comparison
        if answer is not None and normalize_answer(answer) == gt:
            return 1.0
        return 0.0


# ============================================================================
# Format Reward
# ============================================================================

def format_reward(predict: str) -> float:
    """Check if the output has proper format: reasoning followed by \\boxed{answer}.

    Awards 1.0 if:
      - Output contains \\boxed{...} with a non-empty answer
      - There is some reasoning text before the boxed answer
    Awards 0.5 if:
      - Output contains \\boxed{...} but with minimal/no reasoning before it
    Awards 0.0 if:
      - No \\boxed{...} found at all
    """
    boxed_match = re.search(r'\\boxed\{.+\}', predict, re.DOTALL)
    if not boxed_match:
        return 0.0

    reasoning_before = predict[:boxed_match.start()].strip()
    if len(reasoning_before.split()) >= 20:
        return 1.0
    else:
        return 0.5


# ============================================================================
# Step-level Process Reward (heuristic)
# ============================================================================

MATH_PATTERNS = {
    "equations": re.compile(r'[a-zA-Z_]\s*=\s*[\d\.\-\+\*/\(\)a-zA-Z\\]+'),
    "latex_math": re.compile(
        r'\\(?:frac|sqrt|sum|int|prod|lim|sin|cos|tan|log|ln|binom|cdot|times|div|pm|mp|leq|geq|neq|approx)\b'
    ),
    "calculations": re.compile(r'=\s*[\-]?\d+(?:\.\d+)?(?:\s*$|\s*[,\.])', re.MULTILINE),
    "step_markers": re.compile(
        r'(?:^|\n)\s*(?:step\s*\d+|first|second|third|next|then|therefore|thus|hence|so|'
        r'finally|consequently|since|because|given\s+that|we\s+(?:know|have|get|find|can|need|see|note))\b',
        re.IGNORECASE,
    ),
    "math_keywords": re.compile(
        r'\b(?:equation|substitut|simplif|factor|expand|combin|permut|probabilit|triangle|circle|'
        r'angle|area|volume|perimeter|hypotenuse|quadratic|polynomial|derivative|integral|limit|'
        r'sequence|series|modulo|remainder|divisible|prime|gcd|lcm|congruent|parallel|perpendicular)\b',
        re.IGNORECASE,
    ),
    "intermediate_results": re.compile(r'(?:=\s*[\-]?\d+(?:\.\d+)?(?:/\d+)?|\\frac\{[^}]+\}\{[^}]+\})'),
}


def extract_reasoning(predict: str) -> str:
    """Extract reasoning portion: <think>...</think>, or everything before \\boxed{}."""
    think_match = re.search(r'<think>(.*?)</think>', predict, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()

    boxed_match = re.search(r'\\boxed\{', predict)
    if boxed_match:
        reasoning = predict[:boxed_match.start()].strip()
        if reasoning:
            return reasoning

    return predict.strip()


def process_reward(predict: str) -> float:
    """Compute a heuristic process reward for reasoning quality. Returns [0, 1]."""
    reasoning = extract_reasoning(predict)
    if not reasoning:
        return 0.0

    words = reasoning.split()
    num_words = max(len(words), 1)

    # Math density
    math_elements = (
        len(MATH_PATTERNS["equations"].findall(reasoning))
        + len(MATH_PATTERNS["latex_math"].findall(reasoning))
        + len(MATH_PATTERNS["math_keywords"].findall(reasoning))
    )
    math_density = min(1.0, math_elements / max(1, num_words / 20))

    # Step structure
    step_markers = MATH_PATTERNS["step_markers"].findall(reasoning)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', reasoning) if p.strip()]
    num_steps = max(len(step_markers), len(paragraphs), 1)
    step_structure = min(1.0, num_steps / 5.0)

    # Logical flow
    logical_flow = min(1.0, len(step_markers) / 4.0)

    # Calculation presence
    calc_total = (
        len(MATH_PATTERNS["calculations"].findall(reasoning))
        + len(MATH_PATTERNS["intermediate_results"].findall(reasoning))
    )
    calculation_presence = min(1.0, calc_total / 3.0)

    raw_quality = (
        0.30 * math_density
        + 0.30 * step_structure
        + 0.20 * logical_flow
        + 0.20 * calculation_presence
    )

    # Length penalty
    if num_words < 50:
        length_mult = 0.5 + 0.5 * (num_words / 50.0)
    elif num_words > 800:
        length_mult = max(0.7, 1.0 - (num_words - 800) / 2000.0)
    else:
        length_mult = 1.0

    return raw_quality * length_mult


# ============================================================================
# Main entry point — called by verl reward manager
# ============================================================================

def compute_score(
    predicts: List[str],
    ground_truths: List[str],
    format_weight: float = 0.05,
    process_weight: float = 0.0,
) -> List[Dict[str, float]]:
    """Compute rewards for GRPO training.

    Args:
        predicts: Model-generated responses (one per rollout).
        ground_truths: Ground-truth answers (strings, lists, or letters).
        format_weight: Weight for format reward component.
        process_weight: Weight for process/reasoning reward component.

    Returns:
        List of reward dicts with keys: overall, format, accuracy, process.

    Reward formula:
        overall = accuracy_weight * accuracy + format_weight * format + process_weight * process
    Where accuracy_weight = 1 - format_weight - process_weight = 0.80 by default.
    """
    accuracy_weight = 1.0 - format_weight - process_weight
    scores = []

    for predict, ground_truth in zip(predicts, ground_truths):
        # Handle list-type ground_truth (e.g. OlympiadBench final_answer)
        if isinstance(ground_truth, (list, tuple)):
            ground_truth = ground_truth[0] if len(ground_truth) > 0 else ""
        ground_truth = str(ground_truth)

        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)  # normalize spacing

        acc_score = accuracy_reward(predict, ground_truth)
        fmt_score = format_reward(predict)
        proc_score = 0

        overall = (
            accuracy_weight * acc_score
            + format_weight * fmt_score
            + process_weight * proc_score
        )

        scores.append({
            "overall": overall,
            "format": fmt_score,
            "accuracy": acc_score,
            "process": proc_score,
        })

    return scores
