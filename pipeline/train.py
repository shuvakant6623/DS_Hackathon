import logging
import os
import random
import tempfile
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
    load_encoder as _load_encoder_ckpt,
    load_gnn as _load_gnn_ckpt,
    save_encoder,
    save_gnn,
    save_training_history,
)

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility in one place."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



class _FeatureEncoder(nn.Module):

    def __init__(self, input_dim: int = 17, hidden_dim: int = 64,
                 num_emotion_classes: int = 6, num_outcome_classes: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
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
        # Attention pooling for conversation-level outcome prediction
        self.attn_pool = nn.Linear(hidden_dim, 1)
        self.outcome_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_outcome_classes),
        )

    def forward(self, x: torch.Tensor) -> dict:
        h = self.encoder(x)
        return {
            "turn_embeddings": h,
            "emotion_logits": self.emotion_head(h),
            "outcome_logits": self.outcome_head(h),
        }

    def forward_conversation(self, turn_features: torch.Tensor) -> dict:
        """Process all turns of a conversation and predict outcome at conversation level.

        Args:
            turn_features: (num_turns, input_dim) tensor of turn-level features
        Returns:
            dict with turn_embeddings, emotion_logits (per-turn), and
            outcome_logits (single conversation-level prediction)
        """
        h = self.encoder(turn_features)  # (num_turns, hidden_dim)
        emotion_logits = self.emotion_head(h)  # (num_turns, num_emotion_classes)

        # Attention pooling: pool turn embeddings into conversation embedding
        attn_weights = torch.softmax(self.attn_pool(h), dim=0)  # (num_turns, 1)
        conv_embedding = (attn_weights * h).sum(dim=0, keepdim=True)  # (1, hidden_dim)

        outcome_logits = self.outcome_head(conv_embedding)  # (1, num_outcome_classes)
        return {
            "turn_embeddings": h,
            "emotion_logits": emotion_logits,
            "outcome_logits": outcome_logits,
            "conversation_embedding": conv_embedding,
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


class _ConversationLevelDataset(Dataset):
    """Dataset that returns all turns per conversation with one outcome label."""

    def __init__(self, records: List[dict], max_turns: int = 64):
        self.conversations: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
        self.max_turns = max_turns
        for rec in records:
            turn_feats = rec.get("turn_features", [])
            if not turn_feats:
                continue
            outcome_id = rec.get("outcome_id", 0)
            # Build feature matrix and emotion labels for all turns
            features = []
            emotion_labels = []
            for tf in turn_feats[:max_turns]:
                features.append(_ConversationDataset._feature_vector(tf))
                emotion_labels.append(_ConversationDataset._emotion_label(tf))
            feat_tensor = torch.stack(features)  # (num_turns, 17)
            emo_tensor = torch.tensor(emotion_labels, dtype=torch.long)
            self.conversations.append((feat_tensor, emo_tensor, outcome_id))

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int):
        return self.conversations[idx]


def train_encoder(
    config: PipelineConfig,
    records: Optional[List[dict]] = None,
    checkpoint_dir: str = "checkpoints",
    epochs: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    set_seed(config.data.random_seed)
    device = torch.device(config.device)
    if records is None:
        records = process_dataset(config)
    n_epochs = epochs or config.encoder.epochs
    lr = config.encoder.learning_rate

    # Train / val / test split
    np.random.seed(config.data.random_seed)
    n = len(records)
    indices = np.random.permutation(n)
    val_split = int(n * (1 - config.data.val_size - config.data.test_size))
    test_split = int(n * (1 - config.data.test_size))
    train_records = [records[i] for i in indices[:val_split]]
    val_records = [records[i] for i in indices[val_split:test_split]]
    test_records = [records[i] for i in indices[test_split:]]

    if verbose:
        print(f"  Data split: {len(train_records)} train / "
              f"{len(val_records)} val / {len(test_records)} test "
              f"(total {n})")

    # Conversation-level datasets for conversation-level outcome prediction
    train_conv_ds = _ConversationLevelDataset(train_records)
    val_conv_ds = _ConversationLevelDataset(val_records)

    # Compute class weights for balanced training
    outcome_counts = {}
    for rec in train_records:
        oid = rec.get("outcome_id", 0)
        outcome_counts[oid] = outcome_counts.get(oid, 0) + 1
    total_samples = sum(outcome_counts.values())
    n_classes = config.encoder.num_outcome_classes
    class_weights = torch.ones(n_classes, dtype=torch.float32)
    for cls_id, count in outcome_counts.items():
        if cls_id < n_classes:
            class_weights[cls_id] = total_samples / (n_classes * max(count, 1))
    class_weights = class_weights.to(device)

    model = _FeatureEncoder(
        input_dim=17,
        hidden_dim=64,
        num_emotion_classes=config.encoder.num_emotion_classes,
        num_outcome_classes=config.encoder.num_outcome_classes,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    emotion_loss_fn = nn.CrossEntropyLoss()
    outcome_loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_convs = 0

        # Train conversation by conversation for conversation-level outcome
        for conv_feats, conv_emo_labels, outcome_id in train_conv_ds:
            conv_feats = conv_feats.to(device)
            conv_emo_labels = conv_emo_labels.to(device)
            outcome_label = torch.tensor([outcome_id], dtype=torch.long, device=device)

            optimizer.zero_grad()
            out = model.forward_conversation(conv_feats)

            # Emotion loss: per-turn
            loss_emo = emotion_loss_fn(out["emotion_logits"], conv_emo_labels)
            # Outcome loss: conversation-level (single prediction)
            loss_out = outcome_loss_fn(out["outcome_logits"], outcome_label)
            loss = loss_emo + loss_out
            loss.backward()
            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_convs += 1

        avg_train_loss = epoch_loss / max(n_convs, 1)
        history["train_loss"].append(avg_train_loss)

        # Validation (conversation-level)
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        n_val = 0
        with torch.no_grad():
            for conv_feats, conv_emo_labels, outcome_id in val_conv_ds:
                conv_feats = conv_feats.to(device)
                conv_emo_labels = conv_emo_labels.to(device)
                outcome_label = torch.tensor([outcome_id], dtype=torch.long, device=device)

                out = model.forward_conversation(conv_feats)
                loss_emo = emotion_loss_fn(out["emotion_logits"], conv_emo_labels)
                loss_out = outcome_loss_fn(out["outcome_logits"], outcome_label)
                val_loss += (loss_emo + loss_out).item()
                pred = out["outcome_logits"].argmax(dim=1)
                correct += (pred == outcome_label).sum().item()
                total += 1
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

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

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
    history["split"] = {
        "train": len(train_records),
        "val": len(val_records),
        "test": len(test_records),
    }
    history["_test_state"] = {
        "model": model_cpu,
        "test_records": test_records,
    }
    return history

def _evaluate_encoder_test(
    model: nn.Module,
    test_records: List[dict],
    emotion_loss_fn: nn.Module,
    outcome_loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the encoder on a held-out test set using conversation-level predictions."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    test_ds = _ConversationLevelDataset(test_records)
    with torch.no_grad():
        for conv_feats, conv_emo_labels, outcome_id in test_ds:
            conv_feats = conv_feats.to(device)
            conv_emo_labels = conv_emo_labels.to(device)
            outcome_label = torch.tensor([outcome_id], dtype=torch.long, device=device)

            out = model.forward_conversation(conv_feats)
            loss_emo = emotion_loss_fn(out["emotion_logits"], conv_emo_labels)
            loss_out = outcome_loss_fn(out["outcome_logits"], outcome_label)
            test_loss += (loss_emo + loss_out).item()
            pred = out["outcome_logits"].argmax(dim=1)
            correct += (pred == outcome_label).sum().item()
            total += 1
    return {
        "test_loss": test_loss / max(total, 1),
        "test_accuracy": correct / max(total, 1),
    }

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
    device: str = "cpu"

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

    # Train / val / test split
    np.random.seed(config.data.random_seed)
    idx = np.random.permutation(len(graphs))
    val_split = int(len(graphs) * (1 - config.data.val_size - config.data.test_size))
    test_split = int(len(graphs) * (1 - config.data.test_size))
    train_graphs = [graphs[i] for i in idx[:val_split]]
    val_graphs = [graphs[i] for i in idx[val_split:test_split]]
    test_graphs = [graphs[i] for i in idx[test_split:]]

    if verbose:
        print(f"  Data split: {len(train_graphs)} train / "
              f"{len(val_graphs)} val / {len(test_graphs)} test "
              f"(total {len(graphs)} graphs)")

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

    history["split"] = {
        "train": len(train_graphs),
        "val": len(val_graphs),
        "test": len(test_graphs),
    }
    history["_test_state"] = {
        "model": model_cpu,
        "test_graphs": test_graphs,
    }

    return history

def _evaluate_gnn_test(
    model: DiscourseGNN,
    test_graphs: List[dict],
    loss_fn: DiscourseGraphLoss,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the GNN on a held-out test set and return loss/accuracy."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0
    with torch.no_grad():
        for g in test_graphs:
            node_feat = g["node_features"].to(device)
            edge_idx = g["edge_index"].to(device)
            edge_attr = g["edge_attr"].to(device)
            out = model(node_feat, edge_idx)
            edge_logits = model.classify_edges(out["node_embeddings"], edge_idx)
            loss = loss_fn(edge_logits, edge_attr)
            test_loss += loss.item()
            preds = edge_logits.argmax(dim=1)
            correct += (preds == edge_attr).sum().item()
            total += len(edge_attr)
            n_batches += 1
    return {
        "test_loss": test_loss / max(n_batches, 1),
        "test_accuracy": correct / max(total, 1),
    }



def train_all(
    config: Optional[PipelineConfig] = None,
    checkpoint_dir: str = "checkpoints",
    encoder_epochs: Optional[int] = None,
    gnn_epochs: Optional[int] = None,
    verbose: bool = True,
    force_train: bool = False,
    skip_encoder: bool = False,
    skip_gnn: bool = False,
    resume: bool = False,
    skip_tests: bool = False,
) -> Dict[str, Any]:
    if config is None:
        config = PipelineConfig()

    set_seed(config.data.random_seed)

    if verbose:
        print("=" * 60)
        print("TRAINING PIPELINE")
        print(f"Device: {config.device}")
        print("=" * 60)

    paths = default_paths(checkpoint_dir)

    # Load data once
    if verbose:
        print("\n[1/4] Loading data...")
    records = process_dataset(config)
    if verbose:
        print(f"  Loaded {len(records)} conversation records.")

    # Stage 1: Encoder
    enc_hist: Dict[str, Any] = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    if skip_encoder:
        if verbose:
            print("\n[2/4] Skipping encoder training (--skip-encoder).")
        logger.info("Encoder training skipped via --skip-encoder flag.")
    elif not force_train and os.path.exists(paths["encoder"]) and not resume:
        if verbose:
            print(f"\n[2/4] Encoder checkpoint found at {paths['encoder']}; skipping training.")
            print("  Use --force-train to retrain from scratch.")
        logger.info("Encoder checkpoint exists; skipping training.")
    else:
        if verbose:
            print("\n[2/4] Training feature encoder...")
        enc_hist = train_encoder(
            config, records, checkpoint_dir,
            epochs=encoder_epochs, verbose=verbose,
        )

    # Stage 2: GNN
    gnn_hist: Dict[str, Any] = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    if skip_gnn:
        if verbose:
            print("\n[3/4] Skipping GNN training (--skip-gnn).")
        logger.info("GNN training skipped via --skip-gnn flag.")
    elif not force_train and os.path.exists(paths["gnn"]) and not resume:
        if verbose:
            print(f"\n[3/4] GNN checkpoint found at {paths['gnn']}; skipping training.")
            print("  Use --force-train to retrain from scratch.")
        logger.info("GNN checkpoint exists; skipping training.")
    else:
        if verbose:
            print("\n[3/4] Training discourse GNN...")
        gnn_hist = train_gnn(
            config, records, checkpoint_dir,
            epochs=gnn_epochs, verbose=verbose,
        )
    device = torch.device(config.device)

    if not skip_tests:
        if verbose:
            print("\n[4/4] Running test-set evaluation ...")
            print("-" * 40)

        enc_test_state = enc_hist.pop("_test_state", None)
        if enc_test_state is not None:
            enc_model = enc_test_state["model"].to(device)
            emotion_loss_fn = nn.CrossEntropyLoss()
            outcome_loss_fn = nn.CrossEntropyLoss()
            enc_test = _evaluate_encoder_test(
                enc_model, enc_test_state["test_records"],
                emotion_loss_fn, outcome_loss_fn, device,
            )
            enc_hist["test_loss"] = enc_test["test_loss"]
            enc_hist["test_accuracy"] = enc_test["test_accuracy"]
            if verbose:
                print(f"  Encoder  test_loss={enc_test['test_loss']:.4f}"
                      f"  test_acc={enc_test['test_accuracy']:.4f}")

        gnn_test_state = gnn_hist.pop("_test_state", None)
        if gnn_test_state is not None:
            gnn_model = gnn_test_state["model"].to(device)
            gnn_loss_fn = DiscourseGraphLoss()
            gnn_test = _evaluate_gnn_test(
                gnn_model, gnn_test_state["test_graphs"], gnn_loss_fn, device,
            )
            gnn_hist["test_loss"] = gnn_test["test_loss"]
            gnn_hist["test_accuracy"] = gnn_test["test_accuracy"]
            if verbose:
                print(f"  GNN      test_loss={gnn_test['test_loss']:.4f}"
                      f"  test_acc={gnn_test['test_accuracy']:.4f}")

        # Print train-vs-test summary so users can verify generalisation
        if verbose:
            print("-" * 40)
            print("\n  Train vs Test Summary:")
            if enc_hist.get("train_loss") and "test_loss" in enc_hist:
                enc_train_final = enc_hist["train_loss"][-1]
                enc_test_final = enc_hist["test_loss"]
                print(f"    Encoder  train_loss={enc_train_final:.4f}  "
                      f"test_loss={enc_test_final:.4f}  "
                      f"test_acc={enc_hist['test_accuracy']:.4f}")
            if gnn_hist.get("train_loss") and "test_loss" in gnn_hist:
                gnn_train_final = gnn_hist["train_loss"][-1]
                gnn_test_final = gnn_hist["test_loss"]
                print(f"    GNN      train_loss={gnn_train_final:.4f}  "
                      f"test_loss={gnn_test_final:.4f}  "
                      f"test_acc={gnn_hist['test_accuracy']:.4f}")

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