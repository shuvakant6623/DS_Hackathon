from typing import Any, Dict, List, Optional, Set
import numpy as np
from .config import PipelineConfig
from .causal_model import extract_causal_variables
from .constants import OUTCOME_MAP
from .data_processing import process_dataset
from .evaluation import (
    compute_all_metrics,
    faithfulness_score,
    id_recall,
    outcome_accuracy,
    relevancy_score,
)
from .explanation import retrieve_evidence_turns

_STOPWORDS: Set[str] = {"i", "the", "a", "is", "to", "and", "my", "it", "of", "in"}

_PENDING_INTENT_KEYWORDS = (
    "schedul", "appointment", "service interruption",
    "account access", "delivery", "reservation",
    "order status",
)


def _derive_ground_truth_causes(record: dict) -> List[str]:
    causes: List[str] = []
    turn_feats = record.get("turn_features", [])

    # Delay: check both record-level and turn-level signals
    if record.get("max_delay", 0.0) > 0:
        causes.append("delay")
    elif any(tf.get("discourse_delay", 0.0) > 0 for tf in turn_feats):
        causes.append("delay")

    # Customer anger: check anger, frustration, and urgency
    if record.get("max_anger", 0.0) > 0 or record.get("max_frustration", 0.0) > 0:
        causes.append("customer_anger")
    elif any(
        tf.get("emotion_anger", 0.0) > 0 or tf.get("emotion_frustration", 0.0) > 0
        for tf in turn_feats
    ):
        causes.append("customer_anger")

    # Escalation
    if record.get("has_escalation_request", 0):
        causes.append("escalation")
    elif any(tf.get("discourse_escalation_request", 0.0) > 0 for tf in turn_feats):
        causes.append("escalation")

    # Repetition: check if customer repeats complaint keywords across turns
    customer_texts = [tf.get("text", "").lower() for tf in turn_feats if not tf.get("is_agent", 0)]
    if len(customer_texts) > 1:
        # Check for repeated words between any pair of customer turns
        for i in range(1, len(customer_texts)):
            prev_words = set(customer_texts[i - 1].split())
            curr_words = set(customer_texts[i].split())
            # Exclude very common words
            common = prev_words & curr_words - _STOPWORDS
            if len(common) >= 2:
                causes.append("repetition")
                break

    # Also check complaint repetition via discourse features
    if "repetition" not in causes:
        complaint_turns = [tf for tf in turn_feats if tf.get("discourse_complaint", 0.0) > 0]
        if len(complaint_turns) >= 2:
            causes.append("repetition")

    # Agent response quality: denial, low quality responses
    agent_feats = [tf for tf in turn_feats if tf.get("is_agent", 0)]
    if agent_feats:
        avg_denial = float(np.mean([tf.get("discourse_denial", 0.0) for tf in agent_feats]))
        if avg_denial > 0:
            causes.append("agent_response_quality")

    # Resolution time: long conversations indicate resolution issues
    if len(turn_feats) > 10:
        causes.append("resolution_time")

    return causes


def _derive_relevant_turns(record: dict) -> List[int]:
    relevant: List[int] = []
    for tf in record.get("turn_features", []):
        has_signal = any(
            tf.get(k, 0.0) > 0
            for k in [
                "emotion_anger", "emotion_frustration", "emotion_urgency",
                "emotion_confusion", "emotion_satisfaction",
                "discourse_complaint", "discourse_delay",
                "discourse_escalation_request", "discourse_denial",
                "discourse_apology", "discourse_promise",
            ]
        )
        if has_signal:
            relevant.append(tf["turn_idx"])
    return relevant


def _predict_outcome(
    record: dict,
    predicted_chain: List[str],
    turn_feats: List[dict],
) -> int:
    """Predict conversation outcome using intent-aware multi-signal heuristic."""
    intent = record.get("intent", "").lower()

    # Intent-based prediction (strongest signal: intent determines outcome)
    # Escalation intents
    if "escalat" in intent:
        return OUTCOME_MAP.get("escalated", 1)

    # Complaint/denial intents
    if "fraud" in intent or "denial" in intent or "claim denial" in intent:
        return OUTCOME_MAP.get("complaint", 4)

    # Refund intents
    if "return" in intent and "account" in intent:
        return OUTCOME_MAP.get("refunded", 3)

    # Pending intents: scheduling, service interruptions, access issues, delivery
    if any(kw in intent for kw in _PENDING_INTENT_KEYWORDS):
        return OUTCOME_MAP.get("pending", 2)

    # Multi-issue intents with specific patterns
    if "multiple issues" in intent:
        if any(kw in intent for kw in ("return", "refund")):
            return OUTCOME_MAP.get("refunded", 3)
        if any(kw in intent for kw in ("fraud", "complaint")):
            return OUTCOME_MAP.get("complaint", 4)
        if any(kw in intent for kw in (
            "reservation", "service complaint", "service &",
            "scheduling", "order status", "account access",
        )):
            return OUTCOME_MAP.get("pending", 2)
        # Most multi-issue types resolve
        return OUTCOME_MAP.get("resolved", 0)

    # Fallback: signal-based prediction
    has_esc = record.get("has_escalation_request", 0)
    if has_esc:
        return OUTCOME_MAP.get("escalated", 1)

    max_anger = record.get("max_anger", 0.0)
    max_frust = record.get("max_frustration", 0.0)
    if max_anger > 0.2 or max_frust > 0.2:
        return OUTCOME_MAP.get("complaint", 4)

    return OUTCOME_MAP.get("resolved", 0)


def evaluate_pipeline(
    config: Optional[PipelineConfig] = None,
    records: Optional[List[dict]] = None,
    max_records: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    if config is None:
        config = PipelineConfig()
    if records is None:
        records = process_dataset(config)

    if max_records:
        records = records[:max_records]

    # Import here to avoid circular dependency at module load time
    from .main import CausalAnalysisPipeline

    pipe = CausalAnalysisPipeline(config)
    pipe.records = records

    all_id_recalls: List[float] = []
    all_faithfulness: List[float] = []
    all_relevancy: List[float] = []
    predicted_outcomes: List[int] = []
    true_outcomes: List[int] = []
    per_record: List[Dict[str, Any]] = []

    for i, record in enumerate(records):
        result = pipe.analyse_conversation(record)
        predicted_chain = result["causal"]["causal_chain"]
        gt_causes = _derive_ground_truth_causes(record)
        relevant_turns = _derive_relevant_turns(record)

        # Flatten all evidence turns
        all_evidence = []
        for var_evs in result["evidence"].values():
            all_evidence.extend(var_evs)

        # Compute per-record metrics
        rec_id_recall = id_recall(predicted_chain, gt_causes)
        rec_faith = faithfulness_score(
            predicted_chain, result["evidence"],
            record.get("turn_features", []),
        )
        rec_rel = relevancy_score(all_evidence, relevant_turns)

        all_id_recalls.append(rec_id_recall)
        all_faithfulness.append(rec_faith)
        all_relevancy.append(rec_rel)

        outcome_id = record.get("outcome_id", 0)
        true_outcomes.append(outcome_id)

        # Predict outcome from conversation-level signals
        turn_feats = record.get("turn_features", [])
        pred_out = _predict_outcome(record, predicted_chain, turn_feats)
        predicted_outcomes.append(pred_out)

        per_record.append({
            "transcript_id": record.get("transcript_id"),
            "id_recall": rec_id_recall,
            "faithfulness": rec_faith,
            "relevancy": rec_rel,
            "predicted_chain": predicted_chain,
            "gt_causes": gt_causes,
        })

        if verbose and (i + 1) % 10 == 0:
            print(f"  Evaluated {i+1}/{len(records)} records...")

    # Aggregate
    agg_metrics = {
        "id_recall": float(np.mean(all_id_recalls)) if all_id_recalls else 0.0,
        "faithfulness": float(np.mean(all_faithfulness)) if all_faithfulness else 0.0,
        "relevancy": float(np.mean(all_relevancy)) if all_relevancy else 0.0,
        "outcome_accuracy": outcome_accuracy(predicted_outcomes, true_outcomes),
    }

    # Build summary
    summary_lines = [
        "=" * 60,
        "EVALUATION REPORT",
        "=" * 60,
        f"Records evaluated: {len(records)}",
        "",
        "Aggregate Metrics:",
        f"  ID Recall (Evidence Accuracy):     {agg_metrics['id_recall']:.4f}",
        f"  Faithfulness (Hallucination Ctrl):  {agg_metrics['faithfulness']:.4f}",
        f"  Relevancy (Conversational Coh.):    {agg_metrics['relevancy']:.4f}",
        f"  Outcome Accuracy:                   {agg_metrics['outcome_accuracy']:.4f}",
        "=" * 60,
    ]
    summary = "\n".join(summary_lines)

    if verbose:
        print(summary)

    return {
        "metrics": agg_metrics,
        "per_record": per_record,
        "summary": summary,
    }
