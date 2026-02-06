import json
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .constants import EMOTION_KEYWORDS, DISCOURSE_KEYWORDS, OUTCOME_MAP

_INTENT_TO_OUTCOME: Dict[str, str] = {}


def _infer_outcome(intent: str) -> str:
    """Heuristically map an intent string to a coarse outcome label."""
    intent_lower = intent.lower()
    if "escalat" in intent_lower:
        return "escalated"
    if "fraud" in intent_lower or "denial" in intent_lower:
        return "complaint"
    if "refund" in intent_lower or "return" in intent_lower:
        return "refunded"
    if any(kw in intent_lower for kw in ("schedul", "access", "delivery", "service")):
        return "pending"
    return "resolved"


# ── public API ─────────────────────────────────────────────────────────────

def load_data(cfg: PipelineConfig) -> Tuple[pd.DataFrame, Dict[str, list]]:
    """Load CSV metadata and conversation JSON."""
    df = pd.read_csv(cfg.data.csv_path)
    with open(cfg.data.json_path, "r") as f:
        conversations = json.load(f)
    return df, conversations


def _keyword_score(text: str, keywords: List[str]) -> float:
    """Return fraction of keywords that appear in *text*."""
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw in text_lower)
    return hits / max(len(keywords), 1)


def extract_turn_features(turn: dict, turn_idx: int, total_turns: int) -> dict:
    """Build a feature dict for a single dialogue turn."""
    text = turn["text"]
    speaker = turn["speaker"]

    features: dict = {
        "text": text,
        "speaker": speaker,
        "is_agent": int(speaker.lower() == "agent"),
        "turn_idx": turn_idx,
        "turn_position": turn_idx / max(total_turns - 1, 1),
        "word_count": len(text.split()),
        "char_count": len(text),
        "question_marks": text.count("?"),
        "exclamation_marks": text.count("!"),
    }

    # emotion keyword scores
    for emotion, keywords in EMOTION_KEYWORDS.items():
        features[f"emotion_{emotion}"] = _keyword_score(text, keywords)

    # discourse keyword scores
    for relation, keywords in DISCOURSE_KEYWORDS.items():
        features[f"discourse_{relation}"] = _keyword_score(text, keywords)

    return features


def build_conversation_features(
    transcript_id: str,
    turns: list,
    intent: str,
) -> dict:
    outcome = _infer_outcome(intent)
    outcome_id = OUTCOME_MAP.get(outcome, 0)
    turn_feats = [
        extract_turn_features(t, i, len(turns))
        for i, t in enumerate(turns)
    ]

    anger_scores = [tf["emotion_anger"] for tf in turn_feats]
    frustration_scores = [tf["emotion_frustration"] for tf in turn_feats]
    esc_scores = [tf["discourse_escalation_request"] for tf in turn_feats]
    delay_scores = [tf["discourse_delay"] for tf in turn_feats]

    return {
        "transcript_id": transcript_id,
        "outcome": outcome,
        "outcome_id": outcome_id,
        "intent": intent,
        "num_turns": len(turns),
        "avg_turn_len": float(np.mean([tf["word_count"] for tf in turn_feats])),
        "max_anger": float(max(anger_scores)) if anger_scores else 0.0,
        "max_frustration": float(max(frustration_scores)) if frustration_scores else 0.0,
        "has_escalation_request": int(any(s > 0 for s in esc_scores)),
        "max_delay": float(max(delay_scores)) if delay_scores else 0.0,
        "turn_features": turn_feats,
    }


def process_dataset(cfg: PipelineConfig) -> List[dict]:
    """
    End-to-end data processing: load → segment → featurise → return.

    Returns a list of conversation-level feature dicts ready for downstream
    layers.
    """
    df, conversations = load_data(cfg)

    # Build a fast look-up from transcript_id → intent
    id_to_intent: Dict[str, str] = dict(
        zip(df["transcript_id"].astype(str), df["intent"])
    )

    records: List[dict] = []
    for tid, turns in conversations.items():
        intent = id_to_intent.get(tid, "Unknown")
        record = build_conversation_features(tid, turns, intent)
        records.append(record)

    return records
