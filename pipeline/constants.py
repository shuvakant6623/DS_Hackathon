from typing import Dict, List


# ── Emotion keyword lexicon ───────────────────────────────────────────────

EMOTION_KEYWORDS: Dict[str, List[str]] = {
    "anger": [
        "angry", "furious", "unacceptable", "ridiculous", "outrageous",
        "terrible", "worst", "hate", "disgusted", "infuriated",
    ],
    "frustration": [
        "frustrated", "annoyed", "disappointed", "sick of", "fed up",
        "again", "still", "not resolved", "waiting", "multiple times",
    ],
    "satisfaction": [
        "thank", "appreciate", "great", "excellent", "wonderful",
        "helpful", "resolved", "happy", "pleased", "perfect",
    ],
    "confusion": [
        "confused", "don't understand", "unclear", "what do you mean",
        "not sure", "explain", "lost",
    ],
    "urgency": [
        "urgent", "immediately", "asap", "right now", "emergency",
        "critical", "cannot wait", "hurry",
    ],
}



DISCOURSE_KEYWORDS: Dict[str, List[str]] = {
    "complaint": [
        "not working", "broken", "issue", "problem", "wrong",
        "failed", "error", "unacceptable",
    ],
    "denial": [
        "cannot", "unable", "not possible", "unfortunately",
        "don't have", "can't do", "not available",
    ],
    "delay": [
        "wait", "hold", "delay", "business days", "processing time",
        "taking long", "still waiting",
    ],
    "apology": [
        "sorry", "apologize", "apologies", "regret", "understand your frustration",
    ],
    "clarification": [
        "could you clarify", "let me explain", "to clarify", "what I mean",
        "can you confirm", "just to verify",
    ],
    "promise": [
        "I will", "we'll make sure", "guarantee", "promise", "rest assured",
        "going to", "I'll ensure",
    ],
    "escalation_request": [
        "supervisor", "manager", "escalate", "higher authority",
        "someone else", "complaint department", "legal",
    ],
}



CAUSAL_VAR_TO_FEATURE: Dict[str, str] = {
    "delay": "discourse_delay",
    "repetition": "discourse_complaint",
    "agent_response_quality": "discourse_denial",
    "customer_anger": "emotion_anger",
    "escalation": "discourse_escalation_request",
    "resolution_time": "discourse_delay",
}




OUTCOME_MAP: Dict[str, int] = {
    "resolved": 0,
    "escalated": 1,
    "pending": 2,
    "refunded": 3,
    "complaint": 4,
}
