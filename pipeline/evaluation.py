from typing import Any, Dict, List, Set

import numpy as np

from .constants import CAUSAL_VAR_TO_FEATURE


def id_recall(
    predicted_causes: List[str],
    ground_truth_causes: List[str],
) -> float:
    if not ground_truth_causes:
        return 1.0
    gt: Set[str] = set(ground_truth_causes)
    pred: Set[str] = set(predicted_causes)
    return len(gt & pred) / len(gt)


def faithfulness_score(
    explanation_variables: List[str],
    evidence_turns: Dict[str, List[dict]],
    turn_features: List[dict],
) -> float:
    if not explanation_variables:
        return 1.0

    grounded = 0
    for var in explanation_variables:
        feat_key = CAUSAL_VAR_TO_FEATURE.get(var)
        if feat_key is None:
            continue
        var_evs = evidence_turns.get(var, [])
        if any(
            turn_features[ev["turn_idx"]].get(feat_key, 0.0) > 0
            for ev in var_evs
            if ev["turn_idx"] < len(turn_features)
        ):
            grounded += 1

    return grounded / len(explanation_variables)


def relevancy_score(
    retrieved_turns: List[dict],
    relevant_turn_indices: List[int],
) -> float:
    if not retrieved_turns:
        return 0.0
    relevant: Set[int] = set(relevant_turn_indices)
    hits = sum(1 for t in retrieved_turns if t["turn_idx"] in relevant)
    return hits / len(retrieved_turns)


def outcome_accuracy(
    predicted_outcomes: List[int],
    true_outcomes: List[int],
) -> float:
    """Simple accuracy for outcome classification."""
    if not true_outcomes:
        return 0.0
    correct = sum(p == t for p, t in zip(predicted_outcomes, true_outcomes))
    return correct / len(true_outcomes)


def compute_all_metrics(
    predicted_causes: List[str],
    ground_truth_causes: List[str],
    explanation_variables: List[str],
    evidence_turns: Dict[str, List[dict]],
    turn_features: List[dict],
    retrieved_turns: List[dict],
    relevant_turn_indices: List[int],
    predicted_outcomes: List[int],
    true_outcomes: List[int],
) -> Dict[str, float]:
    """Compute every evaluation metric and return a summary dict."""
    return {
        "id_recall": id_recall(predicted_causes, ground_truth_causes),
        "faithfulness": faithfulness_score(
            explanation_variables, evidence_turns, turn_features,
        ),
        "relevancy": relevancy_score(retrieved_turns, relevant_turn_indices),
        "outcome_accuracy": outcome_accuracy(predicted_outcomes, true_outcomes),
    }
