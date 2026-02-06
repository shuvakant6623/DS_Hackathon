import json
import os
from typing import Any, Dict, Optional

import torch

from .config import PipelineConfig, EncoderConfig, DiscourseConfig
DEFAULT_CHECKPOINT_DIR = "checkpoints"

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_encoder(model: torch.nn.Module, path: str, metadata: Optional[dict] = None) -> None:
    """Save encoder weights and optional metadata."""
    _ensure_dir(os.path.dirname(path) or ".")
    payload: Dict[str, Any] = {"model_state_dict": model.state_dict()}
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, path)


def load_encoder(model: torch.nn.Module, path: str) -> dict:
    """Load encoder weights into *model* and return metadata."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint.get("metadata", {})


def save_gnn(model: torch.nn.Module, path: str, metadata: Optional[dict] = None) -> None:
    """Save GNN weights and optional metadata."""
    _ensure_dir(os.path.dirname(path) or ".")
    payload: Dict[str, Any] = {"model_state_dict": model.state_dict()}
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, path)


def load_gnn(model: torch.nn.Module, path: str) -> dict:
    """Load GNN weights into *model* and return metadata."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint.get("metadata", {})


def save_training_history(history: dict, path: str) -> None:
    """Persist training metrics as JSON."""
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(history, f, indent=2, default=str)


def load_training_history(path: str) -> dict:
    """Load training metrics from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def default_paths(checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR) -> dict:
    """Return canonical file paths for all artifacts."""
    return {
        "encoder": os.path.join(checkpoint_dir, "encoder.pt"),
        "gnn": os.path.join(checkpoint_dir, "discourse_gnn.pt"),
        "history": os.path.join(checkpoint_dir, "training_history.json"),
    }