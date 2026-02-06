from typing import Any, Dict, List, Optional

import numpy as np

from .config import ExplanationConfig
from .constants import CAUSAL_VAR_TO_FEATURE


# ── Evidence retrieval ────────────────────────────────────────────────────

def retrieve_evidence_turns(
    turn_features: List[dict],
    causal_variable: str,
    top_k: int = 3,
) -> List[dict]:
    feature_key = CAUSAL_VAR_TO_FEATURE.get(causal_variable, "discourse_complaint")

    scored: List[dict] = []
    for i, tf in enumerate(turn_features):
        score = tf.get(feature_key, 0.0)
        # For anger, also include frustration
        if causal_variable == "customer_anger":
            score = max(score, tf.get("emotion_frustration", 0.0))
        scored.append({
            "turn_idx": i,
            "text": tf["text"],
            "speaker": tf["speaker"],
            "score": score,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def rank_evidence_by_faithfulness(
    evidence: List[dict],
    causal_chain: List[str],
    turn_features: List[dict],
) -> List[dict]:
    for ev in evidence:
        idx = ev["turn_idx"]
        if idx < len(turn_features):
            tf = turn_features[idx]
            faith_score = sum(
                tf.get(CAUSAL_VAR_TO_FEATURE.get(v, ""), 0.0)
                for v in causal_chain
            )
            ev["faithfulness_score"] = faith_score
        else:
            ev["faithfulness_score"] = 0.0

    evidence.sort(key=lambda x: x["faithfulness_score"], reverse=True)
    return evidence


def generate_explanation(
    causal_chain: List[str],
    evidence: Dict[str, List[dict]],
    ate_results: Dict[str, Any],
    conversation_record: dict,
) -> str:
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("CAUSAL EXPLANATION REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Transcript: {conversation_record.get('transcript_id', 'N/A')}")
    lines.append(f"Outcome: {conversation_record.get('outcome', 'N/A')}")
    lines.append(f"Intent: {conversation_record.get('intent', 'N/A')}")
    lines.append("")

    # Causal chain summary
    chain_str = " → ".join(causal_chain)
    lines.append(f"Causal Chain: {chain_str}")
    lines.append("")

    # ATE if available
    if ate_results:
        lines.append(f"Estimated Causal Effect (ATE): {ate_results.get('ate', 0.0):.4f}")
        lines.append(
            f"95% CI: [{ate_results.get('ci_lower', 0.0):.4f}, "
            f"{ate_results.get('ci_upper', 0.0):.4f}]"
        )
        lines.append("")

    # Evidence per variable
    lines.append("SUPPORTING EVIDENCE")
    lines.append("-" * 40)
    for var in causal_chain:
        var_evidence = evidence.get(var, [])
        lines.append(f"\n  Variable: {var}")
        if not var_evidence:
            lines.append("    No direct evidence found.")
            continue
        for ev in var_evidence[:3]:
            turn_ref = f"Turn {ev['turn_idx']}"
            speaker = ev["speaker"]
            snippet = ev["text"][:120]
            score = ev.get("score", 0.0)
            lines.append(
                f"    [{turn_ref}] ({speaker}, relevance={score:.2f}): "
                f'"{snippet}..."'
            )

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


class InteractionContext:

    def __init__(self, config: ExplanationConfig):
        self.config = config
        self.history: List[Dict[str, str]] = []
        self.current_transcript_id: Optional[str] = None
        self.cached_evidence: Dict[str, List[dict]] = {}
        self.cached_causal_chain: List[str] = []

    def add_turn(self, query: str, response: str) -> None:
        """Record a query-response pair."""
        self.history.append({"query": query, "response": response})
        # Keep only the last N turns
        if len(self.history) > self.config.context_window:
            self.history = self.history[-self.config.context_window:]

    def set_context(
        self,
        transcript_id: str,
        evidence: Dict[str, List[dict]],
        causal_chain: List[str],
    ) -> None:
        """Set the active conversation context."""
        self.current_transcript_id = transcript_id
        self.cached_evidence = evidence
        self.cached_causal_chain = causal_chain

    def get_context_summary(self) -> str:
        """Return a brief summary of the current interaction state."""
        lines = [
            f"Active transcript: {self.current_transcript_id or 'None'}",
            f"Causal chain: {' → '.join(self.cached_causal_chain) if self.cached_causal_chain else 'Not set'}",
            f"History length: {len(self.history)} turns",
        ]
        return "\n".join(lines)

    def handle_query(self, query: str, conversation_record: dict) -> str:
        query_lower = query.lower()

        if "why" in query_lower:
            response = generate_explanation(
                self.cached_causal_chain,
                self.cached_evidence,
                {},
                conversation_record,
            )
        elif "evidence" in query_lower or "show" in query_lower:
            lines = ["Evidence for current analysis:"]
            for var, evs in self.cached_evidence.items():
                lines.append(f"\n  {var}:")
                for ev in evs[:2]:
                    lines.append(
                        f"    Turn {ev['turn_idx']} ({ev['speaker']}): "
                        f"\"{ev['text'][:80]}...\""
                    )
            response = "\n".join(lines)
        elif "what if" in query_lower or "counterfactual" in query_lower:
            response = (
                "Counterfactual analysis: If the identified root cause had "
                "been addressed earlier, the model estimates the escalation "
                "probability would have decreased based on the causal chain: "
                + " → ".join(self.cached_causal_chain)
            )
        else:
            response = (
                f"Current analysis context:\n"
                f"{self.get_context_summary()}\n\n"
                f"Please ask about 'why', 'evidence', or 'what if' scenarios."
            )

        self.add_turn(query, response)
        return response
