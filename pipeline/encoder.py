import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EncoderConfig


class TurnEncoder(nn.Module):

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # Lazy import so the module can be loaded even without transformers
        from transformers import AutoModel

        self.transformer = AutoModel.from_pretrained(config.model_name)
        hidden = config.hidden_dim

        # ── Classification heads ──────────────────────────────────────
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden // 2, config.num_emotion_classes),
        )

        self.outcome_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden // 2, config.num_outcome_classes),
        )

        # Evidence span detection: binary per-token tag
        self.evidence_head = nn.Linear(hidden, 2)

    # ─── forward ──────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # CLS token embedding → turn-level representation
        cls_emb = outputs.last_hidden_state[:, 0, :]       # (B, H)
        token_emb = outputs.last_hidden_state               # (B, L, H)

        emotion_logits = self.emotion_head(cls_emb)          # (B, E)
        outcome_logits = self.outcome_head(cls_emb)          # (B, O)
        evidence_logits = self.evidence_head(token_emb)      # (B, L, 2)

        return {
            "turn_embeddings": cls_emb,
            "emotion_logits": emotion_logits,
            "outcome_logits": outcome_logits,
            "evidence_logits": evidence_logits,
        }


class EncoderLoss(nn.Module):

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        emotion_logits: torch.Tensor,
        emotion_labels: torch.Tensor,
        outcome_logits: torch.Tensor,
        outcome_labels: torch.Tensor,
        evidence_logits: torch.Tensor,
        evidence_labels: torch.Tensor,
    ) -> dict:
        """Return total loss and per-task breakdown."""
        l_emo = self.ce(emotion_logits, emotion_labels)
        l_out = self.ce(outcome_logits, outcome_labels)
        # evidence: flatten (B, L, 2) → (B*L, 2)
        l_evi = self.ce(
            evidence_logits.view(-1, 2),
            evidence_labels.view(-1),
        )
        total = self.alpha * l_emo + self.beta * l_out + self.gamma * l_evi
        return {
            "loss": total,
            "emotion_loss": l_emo,
            "outcome_loss": l_out,
            "evidence_loss": l_evi,
        }


def encode_turns(
    texts: list,
    tokenizer,
    model: TurnEncoder,
    max_length: int = 128,
    device: str = "cpu",
) -> dict:
    encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    model.eval()
    with torch.no_grad():
        return model(input_ids, attention_mask)
