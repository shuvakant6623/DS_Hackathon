import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from .config import PipelineConfig
from .constants import OUTCOME_MAP
from .data_processing import process_dataset
from .discourse_graph import (
    DiscourseGNN,
    DiscourseGraphLoss,
    build_discourse_graph,
)
from .model_io import (
    default_paths,
    save_encoder,
    save_gnn,
    save_training_history,
)



class _FeatureEncoder(nn.Module):

    def __init__(self, input_dim: int = 17, hidden_dim: int = 64,
                 num_emotion_classes: int = 6, num_outcome_classes: int = 5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_emotion_classes),
        )
        self.outcome_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_outcome_classes),
        )

    def forward(self, x: torch.Tensor) -> dict:
        h = self.encoder(x)
        return {
            "turn_embeddings": h,
            "emotion_logits": self.emotion_head(h),
            "outcome_logits": self.outcome_head(h),
        }


class _ConversationDataset(Dataset):
    """Dataset wrapper over processed conversation records."""

    def __init__(self, records: List[dict]):
        self.samples: List[Tuple[torch.Tensor, int, int]] = []
        for rec in records:
            outcome_id = rec.get("outcome_id", 0)
            for tf in rec.get("turn_features", []):
                fvec = self._feature_vector(tf)
                emotion_label = self._emotion_label(tf)
                self.samples.append((fvec, emotion_label, outcome_id))

    @staticmethod
    def _feature_vector(tf: dict) -> torch.Tensor:
        vec = [
            tf.get("is_agent", 0),
            tf.get("turn_position", 0.0),
            tf.get("word_count", 0) / 100.0,
            tf.get("question_marks", 0) / 5.0,
            tf.get("exclamation_marks", 0) / 5.0,
            tf.get("emotion_anger", 0.0),
            tf.get("emotion_frustration", 0.0),
            tf.get("emotion_satisfaction", 0.0),
            tf.get("emotion_confusion", 0.0),
            tf.get("emotion_urgency", 0.0),
            tf.get("discourse_complaint", 0.0),
            tf.get("discourse_denial", 0.0),
            tf.get("discourse_delay", 0.0),
            tf.get("discourse_apology", 0.0),
            tf.get("discourse_clarification", 0.0),
            tf.get("discourse_promise", 0.0),
            tf.get("discourse_escalation_request", 0.0),
        ]
        return torch.tensor(vec, dtype=torch.float32)

    @staticmethod
    def _emotion_label(tf: dict) -> int:
        """Derive dominant emotion label from keyword scores."""
        emotions = [
            tf.get("emotion_anger", 0.0),
            tf.get("emotion_frustration", 0.0),
            tf.get("emotion_satisfaction", 0.0),
            tf.get("emotion_confusion", 0.0),
            tf.get("emotion_urgency", 0.0),
        ]
        max_score = max(emotions)
        if max_score == 0:
            return 0  # neutral
        return int(np.argmax(emotions)) + 1  # 1-based (0 = neutral)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def train_encoder(
    config: PipelineConfig,
    records: Optional[List[dict]] = None,
    checkpoint_dir: str = "checkpoints",
    epochs: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    if records is None:
        records = process_dataset(config)
    device = torch.device(config.device)
    n_epochs = epochs or config.encoder.epochs
    lr = config.encoder.learning_rate

    # Train / val split
    np.random.seed(config.data.random_seed)
    n = len(records)
    indices = np.random.permutation(n)
    split = int(n * (1 - config.data.test_size))
    train_records = [records[i] for i in indices[:split]]
    val_records = [records[i] for i in indices[split:]]

    train_ds = _ConversationDataset(train_records)
    val_ds = _ConversationDataset(val_records)
    train_loader = DataLoader(train_ds, batch_size=config.encoder.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.encoder.batch_size)

    model = _FeatureEncoder(
        input_dim=17,
        hidden_dim=64,
        num_emotion_classes=config.encoder.num_emotion_classes,
        num_outcome_classes=config.encoder.num_outcome_classes,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    emotion_loss_fn = nn.CrossEntropyLoss()
    outcome_loss_fn = nn.CrossEntropyLoss()

    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for features, emo_labels, out_labels in train_loader:
            features = features.to(device)
            emo_labels = emo_labels.to(device)
            out_labels = out_labels.to(device)
            optimizer.zero_grad()
            out = model(features)
            loss = emotion_loss_fn(out["emotion_logits"], emo_labels) + \
                   outcome_loss_fn(out["outcome_logits"], out_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        n_val = 0
        with torch.no_grad():
            for features, emo_labels, out_labels in val_loader:
                features = features.to(device)
                emo_labels = emo_labels.to(device)
                out_labels = out_labels.to(device)
                out = model(features)
                loss = emotion_loss_fn(out["emotion_logits"], emo_labels) + \
                       outcome_loss_fn(out["outcome_logits"], out_labels)
                val_loss += loss.item()
                preds = out["outcome_logits"].argmax(dim=1)
                correct += (preds == out_labels).sum().item()
                total += len(out_labels)
                n_val += 1

        avg_val_loss = val_loss / max(n_val, 1)
        val_acc = correct / max(total, 1)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(val_acc)

        if verbose:
            print(
                f"  Encoder Epoch {epoch+1}/{n_epochs}  "
                f"train_loss={avg_train_loss:.4f}  "
                f"val_loss={avg_val_loss:.4f}  "
                f"val_acc={val_acc:.4f}"
            )

    # Save checkpoint
    model_cpu = model.cpu()
    paths = default_paths(checkpoint_dir)
    save_encoder(model_cpu, paths["encoder"], metadata={
        "epochs": n_epochs,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_accuracy": history["val_accuracy"][-1] if history["val_accuracy"] else None,
        "device": config.device,
    })
    if verbose:
        print(f"  Encoder saved to {paths['encoder']}")

    return history

def _build_turn_embeddings(turn_features: List[dict], embed_dim: int = 32) -> torch.Tensor:
    """Build feature-based turn embeddings (same as CausalAnalysisPipeline._encode_turns)."""
    embeddings = []
    for tf in turn_features:
        vec = [
            tf.get("is_agent", 0),
            tf.get("turn_position", 0.0),
            tf.get("word_count", 0) / 100.0,
            tf.get("question_marks", 0) / 5.0,
            tf.get("exclamation_marks", 0) / 5.0,
            tf.get("emotion_anger", 0.0),
            tf.get("emotion_frustration", 0.0),
            tf.get("emotion_satisfaction", 0.0),
            tf.get("emotion_confusion", 0.0),
            tf.get("emotion_urgency", 0.0),
            tf.get("discourse_complaint", 0.0),
            tf.get("discourse_denial", 0.0),
            tf.get("discourse_delay", 0.0),
            tf.get("discourse_apology", 0.0),
            tf.get("discourse_clarification", 0.0),
            tf.get("discourse_promise", 0.0),
            tf.get("discourse_escalation_request", 0.0),
        ]
        vec.extend([0.0] * (embed_dim - len(vec)))
        embeddings.append(vec[:embed_dim])
    return torch.tensor(embeddings, dtype=torch.float32)


def train_gnn(
    config: PipelineConfig,
    records: Optional[List[dict]] = None,
    checkpoint_dir: str = "checkpoints",
    epochs: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    if records is None:
        records = process_dataset(config)
        device = torch.device(config.device)

    n_epochs = epochs or config.discourse.epochs
    lr = config.discourse.learning_rate
    embed_dim = 32

    # Build graph data for all conversations
    graphs: List[dict] = []
    for rec in records:
        tf = rec.get("turn_features", [])
        if len(tf) < 2:
            continue
        turn_emb = _build_turn_embeddings(tf, embed_dim)
        g = build_discourse_graph(tf, turn_emb, config.discourse.edge_types)
        if g["edge_index"].shape[1] > 0:
            graphs.append(g)

    if not graphs:
        if verbose:
            print("  No graphs with edges found; skipping GNN training.")
        return {"train_loss": []}

    # Train / val split
    np.random.seed(config.data.random_seed)
    idx = np.random.permutation(len(graphs))
    split = int(len(graphs) * (1 - config.data.test_size))
    train_graphs = [graphs[i] for i in idx[:split]]
    val_graphs = [graphs[i] for i in idx[split:]]

    model = DiscourseGNN(config.discourse, input_dim=embed_dim).to(device)
    loss_fn = DiscourseGraphLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for g in train_graphs:
            node_feat = g["node_features"].to(device)
            edge_idx = g["edge_index"].to(device)
            edge_attr = g["edge_attr"].to(device)
            optimizer.zero_grad()
            out = model(node_feat, edge_idx)
            edge_logits = model.classify_edges(out["node_embeddings"], edge_idx)
            loss = loss_fn(edge_logits, edge_attr)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        n_val = 0
        with torch.no_grad():
            for g in val_graphs:
                node_feat = g["node_features"].to(device)
                edge_idx = g["edge_index"].to(device)
                edge_attr = g["edge_attr"].to(device)
                out = model(node_feat, edge_idx)
                edge_logits = model.classify_edges(out["node_embeddings"], edge_idx)
                loss = loss_fn(edge_logits, edge_attr)
                val_loss += loss.item()
                preds = edge_logits.argmax(dim=1)
                correct += (preds == edge_attr).sum().item()
                total += len(edge_attr)
                n_val += 1

        avg_val = val_loss / max(n_val, 1)
        val_acc = correct / max(total, 1)
        history["val_loss"].append(avg_val)
        history["val_accuracy"].append(val_acc)

        if verbose:
            print(
                f"  GNN Epoch {epoch+1}/{n_epochs}  "
                f"train_loss={avg_train:.4f}  "
                f"val_loss={avg_val:.4f}  "
                f"val_acc={val_acc:.4f}"
            )

    # Save
    model_cpu = model.cpu()
    paths = default_paths(checkpoint_dir)
    save_gnn(model_cpu, paths["gnn"], metadata={
        "epochs": n_epochs,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_accuracy": history["val_accuracy"][-1] if history["val_accuracy"] else None,
        "device": config.device,
    })
    if verbose:
        print(f"  GNN saved to {paths['gnn']}")

    return history


def train_all(
    config: Optional[PipelineConfig] = None,
    checkpoint_dir: str = "checkpoints",
    encoder_epochs: Optional[int] = None,
    gnn_epochs: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    if config is None:
        config = PipelineConfig()

    if verbose:
        print("=" * 60)
        print("TRAINING PIPELINE")
        print(f"Device: {config.device}")
        print("=" * 60)

    # Load data once
    if verbose:
        print("\n[1/3] Loading data...")
    records = process_dataset(config)
    if verbose:
        print(f"  Loaded {len(records)} conversation records.")

    # Stage 1: Encoder
    if verbose:
        print("\n[2/3] Training feature encoder...")
    enc_hist = train_encoder(
        config, records, checkpoint_dir,
        epochs=encoder_epochs, verbose=verbose,
    )

    # Stage 2: GNN
    if verbose:
        print("\n[3/3] Training discourse GNN...")
    gnn_hist = train_gnn(
        config, records, checkpoint_dir,
        epochs=gnn_epochs, verbose=verbose,
    )

    # Save combined history
    combined = {
        "encoder_history": enc_hist,
        "gnn_history": gnn_hist,
    }
    paths = default_paths(checkpoint_dir)
    save_training_history(combined, paths["history"])
    if verbose:
        print(f"\n  Training history saved to {paths['history']}")
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)

    return combined