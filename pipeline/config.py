from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Paths and data-processing settings."""

    csv_path: str = "Datasets/processed/transcript_dataset.csv"
    json_path: str = "Datasets/processed/conversation_transcript_map.json"
    max_turns: int = 64
    max_token_len: int = 128
    val_size: float = 0.1
    test_size: float = 0.1
    random_seed: int = 42


@dataclass
class EncoderConfig:
    """Layer-1 encoder hyper-parameters."""

    model_name: str = "bert-base-uncased"
    hidden_dim: int = 768
    num_emotion_classes: int = 6  # neutral, anger, frustration, satisfaction, confusion, urgency
    num_outcome_classes: int = 5  # resolved, escalated, pending, refunded, complaint
    dropout: float = 0.3
    learning_rate: float = 2e-5
    epochs: int = 10
    batch_size: int = 16


@dataclass
class DiscourseConfig:
    """Layer-2 discourse graph hyper-parameters."""

    edge_types: List[str] = field(
        default_factory=lambda: [
            "complaint",
            "denial",
            "delay",
            "apology",
            "clarification",
            "promise",
            "escalation_request",
        ]
    )
    gnn_hidden_dim: int = 256
    gnn_num_layers: int = 3
    gnn_heads: int = 4
    dropout: float = 0.3
    learning_rate: float = 1e-3
    epochs: int = 30
    batch_size: int = 32


@dataclass
class CausalConfig:
    """Layer-3 causal model settings."""

    causal_variables: List[str] = field(
        default_factory=lambda: [
            "delay",
            "repetition",
            "agent_response_quality",
            "customer_anger",
            "resolution_time",
            "escalation",
        ]
    )
    treatment: str = "delay"
    outcome: str = "escalation"
    n_bootstrap: int = 100
    significance_level: float = 0.05


@dataclass
class ExplanationConfig:
    """Layer-4 explanation / interaction settings."""

    max_evidence_turns: int = 5
    temperature: float = 0.3
    max_generation_len: int = 512
    context_window: int = 10

def _resolve_device(device: str) -> str:
    """Resolve ``'auto'`` to the best available device."""
    if device == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

@dataclass
class PipelineConfig:
    """Root configuration that aggregates every layer."""

    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    discourse: DiscourseConfig = field(default_factory=DiscourseConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    explanation: ExplanationConfig = field(default_factory=ExplanationConfig)
    device: str = "auto"  # "cpu", "cuda", or "auto" (auto-detect)

    def __post_init__(self) -> None:
        self.device = _resolve_device(self.device)
