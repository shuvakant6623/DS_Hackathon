import json
import logging
import os
import tempfile
from typing import Any, Dict, Optional

import torch

from .config import PipelineConfig, EncoderConfig, DiscourseConfig

DEFAULT_CHECKPOINT_DIR = "checkpoints"
logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _atomic_save(payload: dict, path: str) -> None:
    """Save a checkpoint atomically via a temporary file + rename."""
    _ensure_dir(os.path.dirname(path) or ".")
    fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(path) or ".", suffix=".tmp"
    )
    try:
        os.close(fd)
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def save_encoder(model: torch.nn.Module, path: str, metadata: Optional[dict] = None) -> None:
    """Save encoder weights and optional metadata."""
    payload: Dict[str, Any] = {"model_state_dict": model.state_dict()}
    if metadata:
        payload["metadata"] = metadata
    logger.info("Saving encoder checkpoint to %s", path)
    _atomic_save(payload, path)


def load_encoder(
    model: torch.nn.Module,
    path: str,
    device: str = "cpu",
) -> dict:
    """Load encoder weights into *model* with device mapping and validation."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Encoder checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    # Shape validation
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    if model_keys != ckpt_keys:
        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys
        logger.warning(
            "Checkpoint key mismatch - missing: %s, unexpected: %s",
            missing, unexpected,
        )
    model.load_state_dict(state_dict)
    model.to(device)
    meta = checkpoint.get("metadata", {})
    logger.info("Loaded encoder from %s (metadata: %s)", path, meta)
    return meta


def save_gnn(model: torch.nn.Module, path: str, metadata: Optional[dict] = None) -> None:
    """Save GNN weights and optional metadata."""
    payload: Dict[str, Any] = {"model_state_dict": model.state_dict()}
    if metadata:
        payload["metadata"] = metadata
    logger.info("Saving GNN checkpoint to %s", path)
    _atomic_save(payload, path)


def load_gnn(
    model: torch.nn.Module,
    path: str,
    device: str = "cpu",
) -> dict:
    """Load GNN weights into *model* with device mapping and validation."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"GNN checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    # Shape validation
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    if model_keys != ckpt_keys:
        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys
        logger.warning(
            "Checkpoint key mismatch - missing: %s, unexpected: %s",
            missing, unexpected,
        )
    model.load_state_dict(state_dict)
    model.to(device)
    meta = checkpoint.get("metadata", {})
    logger.info("Loaded GNN from %s (metadata: %s)", path, meta)
    return meta


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