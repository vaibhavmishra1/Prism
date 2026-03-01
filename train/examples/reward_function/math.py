# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import math
from typing import Dict, List

from mathruler.grader import extract_boxed_content, grade_answer


# ============================================================================
# Step-level Process Reward Model (PRM) - Heuristic Implementation
# ============================================================================
# Instead of a binary accuracy reward, this provides a richer signal by
# evaluating the quality of reasoning steps inside <think>...</think> blocks.
# This helps the model learn structured multi-step reasoning even when
# the final answer is incorrect.
# ============================================================================

# Regex patterns for detecting mathematical reasoning quality
MATH_PATTERNS = {
    # Equations and expressions (e.g., "x = 5", "2x + 3 = 7", "f(x) = ...")
    "equations": re.compile(r'[a-zA-Z_]\s*=\s*[\d\.\-\+\*/\(\)a-zA-Z\\]+'),
    # LaTeX math expressions (\frac, \sqrt, \sum, \int, etc.)
    "latex_math": re.compile(r'\\(?:frac|sqrt|sum|int|prod|lim|sin|cos|tan|log|ln|binom|cdot|times|div|pm|mp|leq|geq|neq|approx)\b'),
    # Numeric calculations (e.g., "= 15", "= 3.14")
    "calculations": re.compile(r'=\s*[\-]?\d+(?:\.\d+)?(?:\s*$|\s*[,\.])', re.MULTILINE),
    # Step markers (e.g., "Step 1:", "First,", "Then,", "Therefore,", "So,", "Thus,")
    "step_markers": re.compile(r'(?:^|\n)\s*(?:step\s*\d+|first|second|third|next|then|therefore|thus|hence|so|finally|consequently|since|because|given\s+that|we\s+(?:know|have|get|find|can|need|see|note))\b', re.IGNORECASE),
    # Mathematical keywords
    "math_keywords": re.compile(r'\b(?:equation|substitut|simplif|factor|expand|combin|permut|probabilit|triangle|circle|angle|area|volume|perimeter|hypotenuse|quadratic|polynomial|derivative|integral|limit|sequence|series|modulo|remainder|divisible|prime|gcd|lcm|congruent|parallel|perpendicular)\b', re.IGNORECASE),
    # Boxed intermediate results or key expressions
    "intermediate_results": re.compile(r'(?:=\s*[\-]?\d+(?:\.\d+)?(?:/\d+)?|\\frac\{[^}]+\}\{[^}]+\})'),
}


def extract_reasoning(predict: str) -> str:
    """Extract the reasoning portion of the output.

    Tries in order:
    1. Content inside <think>...</think> tags (if present)
    2. Everything before the last \\boxed{...} occurrence
    3. The entire output as fallback
    """
    # Try <think> tags first
    think_match = re.search(r'<think>(.*?)</think>', predict, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()

    # Otherwise, use everything before the last \boxed{} as reasoning
    boxed_match = re.search(r'\\boxed\{', predict)
    if boxed_match:
        reasoning = predict[:boxed_match.start()].strip()
        if reasoning:
            return reasoning

    # Fallback: entire output
    return predict.strip()


def count_reasoning_steps(think_content: str) -> int:
    """Count the number of distinct reasoning steps in the think block.

    Steps are identified by:
    - Explicit step markers ("Step 1:", "First,", "Then,", etc.)
    - Paragraph breaks (double newlines)
    - Sentence-level transitions with mathematical content
    """
    if not think_content:
        return 0

    # Split by double newlines or explicit step markers
    step_markers = MATH_PATTERNS["step_markers"].findall(think_content)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', think_content) if p.strip()]

    # Use the max of explicit markers and paragraph count
    return max(len(step_markers), len(paragraphs), 1)


def compute_step_quality(think_content: str) -> Dict[str, float]:
    """Evaluate the quality of reasoning steps.

    Returns a dict with component scores (each in [0, 1]):
        - math_density: How much mathematical content is present
        - step_structure: How well-structured the reasoning is
        - logical_flow: Whether there are logical connectors
        - calculation_presence: Whether explicit calculations are shown
    """
    if not think_content:
        return {
            "math_density": 0.0,
            "step_structure": 0.0,
            "logical_flow": 0.0,
            "calculation_presence": 0.0,
        }

    words = think_content.split()
    num_words = max(len(words), 1)

    # 1. Math density: ratio of mathematical content
    equation_matches = len(MATH_PATTERNS["equations"].findall(think_content))
    latex_matches = len(MATH_PATTERNS["latex_math"].findall(think_content))
    math_keyword_matches = len(MATH_PATTERNS["math_keywords"].findall(think_content))
    math_elements = equation_matches + latex_matches + math_keyword_matches
    # Normalize: expect ~1 math element per 20 words for good reasoning
    math_density = min(1.0, math_elements / max(1, num_words / 20))

    # 2. Step structure: how many distinct reasoning steps
    num_steps = count_reasoning_steps(think_content)
    # Expect 3-10 steps for good math reasoning; sigmoid-like scaling
    step_structure = min(1.0, num_steps / 5.0)

    # 3. Logical flow: presence of logical connectors and transitions
    step_marker_matches = len(MATH_PATTERNS["step_markers"].findall(think_content))
    logical_flow = min(1.0, step_marker_matches / 4.0)

    # 4. Calculation presence: whether explicit numeric calculations are shown
    calculation_matches = len(MATH_PATTERNS["calculations"].findall(think_content))
    intermediate_matches = len(MATH_PATTERNS["intermediate_results"].findall(think_content))
    calc_total = calculation_matches + intermediate_matches
    calculation_presence = min(1.0, calc_total / 3.0)

    return {
        "math_density": math_density,
        "step_structure": step_structure,
        "logical_flow": logical_flow,
        "calculation_presence": calculation_presence,
    }


def compute_length_penalty(think_content: str) -> float:
    """Penalize extremely short reasoning (likely guessing) or excessively long/repetitive reasoning.

    Returns a multiplier in [0.5, 1.0]:
        - Very short (<50 words): penalty (down to 0.5)
        - Normal (50-800 words): no penalty (1.0)
        - Too long (>800 words): slight penalty (down to 0.7)
    """
    if not think_content:
        return 0.5

    num_words = len(think_content.split())

    if num_words < 50:
        # Linear ramp from 0.5 to 1.0 for 0-50 words
        return 0.5 + 0.5 * (num_words / 50.0)
    elif num_words > 800:
        # Gentle penalty for very long outputs (possible repetition)
        excess = num_words - 800
        return max(0.7, 1.0 - excess / 2000.0)
    else:
        return 1.0


def compute_repetition_penalty(think_content: str) -> float:
    """Detect and penalize repetitive reasoning patterns.

    Returns a multiplier in [0.3, 1.0].
    """
    if not think_content or len(think_content) < 100:
        return 1.0

    # Split into sentences
    sentences = re.split(r'[.!?\n]+', think_content)
    sentences = [s.strip().lower() for s in sentences if len(s.strip()) > 10]

    if len(sentences) < 3:
        return 1.0

    # Check for duplicate sentences
    unique_sentences = set(sentences)
    uniqueness_ratio = len(unique_sentences) / len(sentences)

    # Also check for n-gram repetition (trigrams)
    words = think_content.lower().split()
    if len(words) > 20:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        unique_trigrams = set(trigrams)
        trigram_ratio = len(unique_trigrams) / max(len(trigrams), 1)
    else:
        trigram_ratio = 1.0

    # Combined repetition score
    rep_score = min(uniqueness_ratio, trigram_ratio)

    # Map to [0.3, 1.0]
    return max(0.3, rep_score)


def process_reward(predict: str) -> float:
    """Compute a process reward score for the reasoning quality.

    This is a heuristic PRM that evaluates the structure and quality
    of mathematical reasoning without needing a separate trained model.

    Works on the full output — uses <think> block if present, otherwise
    uses everything before \\boxed{} as the reasoning portion.

    Returns a score in [0.0, 1.0].
    """
    reasoning = extract_reasoning(predict)

    if not reasoning:
        return 0.0

    # Compute component scores
    quality = compute_step_quality(reasoning)

    # Weighted combination of quality components
    raw_quality = (
        0.30 * quality["math_density"]
        + 0.30 * quality["step_structure"]
        + 0.20 * quality["logical_flow"]
        + 0.20 * quality["calculation_presence"]
    )

    # Apply length and repetition penalties
    length_mult = compute_length_penalty(reasoning)
    rep_mult = compute_repetition_penalty(reasoning)

    return raw_quality * length_mult * rep_mult


# ============================================================================
# Original reward functions (kept for compatibility)
# ============================================================================

def format_reward(predict: str) -> float:
    """Check if the output has proper format: reasoning followed by \\boxed{answer}.

    Awards 1.0 if:
      - Output contains \\boxed{...} with a non-empty answer
      - There is some reasoning text before the boxed answer (not just the answer alone)
    Awards 0.5 if:
      - Output contains \\boxed{...} but with minimal/no reasoning before it
    Awards 0.0 if:
      - No \\boxed{...} found at all
    """
    # Check for \boxed{...} with non-empty content
    boxed_match = re.search(r'\\boxed\{.+\}', predict, re.DOTALL)
    if not boxed_match:
        return 0.0

    # Check if there's meaningful reasoning before the boxed answer
    reasoning_before = predict[:boxed_match.start()].strip()
    if len(reasoning_before.split()) >= 20:
        return 1.0  # Good: has reasoning + boxed answer
    else:
        return 0.5  # Has boxed answer but minimal reasoning


def accuracy_reward(predict: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict)
    try:
        return 1.0 if grade_answer(answer, ground_truth) else 0.0
    except:
        return 0.0


# ============================================================================
# Main reward function
# ============================================================================

def compute_score(
    predicts: List[str],
    ground_truths: List[str],
    format_weight: float = 0.05,
    process_weight: float = 0.15,
) -> List[Dict[str, float]]:
    """Compute rewards with step-level process reward model.

    The overall reward combines three components:
        1. Accuracy reward (0 or 1): Did the model get the right answer?
        2. Format reward (0, 0.5, or 1): Does the output have \\boxed{} with reasoning?
        3. Process reward (0 to 1): How good is the reasoning quality?

    Process and format rewards work on the FULL output — no <think> tags needed.
    Reasoning is extracted as everything before the last \\boxed{}.

    Weights:
        overall = (1 - format_weight - process_weight) * accuracy
                + format_weight * format
                + process_weight * process_reward

    With defaults (format=0.05, process=0.15):
        - Correct + good reasoning + boxed: 0.80 + 0.05 + ~0.15 = ~1.0
        - Correct + poor reasoning: 0.80 + 0.025 + ~0.03 ≈ 0.86
        - Wrong + excellent reasoning: 0.0 + 0.05 + ~0.15 = ~0.20
        - Wrong + poor reasoning: 0.0 + 0.025 + ~0.02 ≈ 0.05
        - No boxed answer at all: 0.0 + 0.0 + ~0.02 = ~0.02
    """
    accuracy_weight = 1.0 - format_weight - process_weight
    scores = []

    for predict, ground_truth in zip(predicts, ground_truths):
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)  # handle qwen2.5vl-32b format

        format_score = format_reward(predict)
        acc_score = accuracy_reward(predict, ground_truth)
        proc_score = process_reward(predict)

        overall = (
            accuracy_weight * acc_score
            + format_weight * format_score
            + process_weight * proc_score
        )

        scores.append({
            "overall": overall,
            "format": format_score,
            "accuracy": acc_score,
            "process": proc_score,
        })

    return scores
