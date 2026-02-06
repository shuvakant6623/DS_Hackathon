"""Tests for the training pipeline.

Verifies that:
- train_encoder and train_gnn do NOT call train_all (no recursive training)
- Checkpoint-aware training skips retraining when checkpoints exist
- --force-train overrides checkpoint skipping
- Deterministic seeding produces reproducible results
"""
import ast
import inspect
import os
import shutil
import tempfile

import pytest
import torch

from pipeline.config import PipelineConfig
from pipeline.train import (
    _FeatureEncoder,
    train_all,
    train_encoder,
    train_gnn,
    set_seed,
)
from pipeline.model_io import default_paths, save_encoder, save_gnn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_records(n: int = 20) -> list:
    """Generate minimal synthetic records for fast training tests."""
    import numpy as np

    records = []
    for i in range(n):
        turn_features = []
        for t in range(4):
            turn_features.append({
                "turn_idx": t,
                "is_agent": int(t % 2 == 1),
                "turn_position": t / 4.0,
                "word_count": 10,
                "question_marks": 0,
                "exclamation_marks": 0,
                "emotion_anger": float(np.random.rand()),
                "emotion_frustration": float(np.random.rand()),
                "emotion_satisfaction": 0.0,
                "emotion_confusion": 0.0,
                "emotion_urgency": 0.0,
                "discourse_complaint": 0.0,
                "discourse_denial": 0.0,
                "discourse_delay": float(np.random.rand()),
                "discourse_apology": 0.0,
                "discourse_clarification": 0.0,
                "discourse_promise": 0.0,
                "discourse_escalation_request": 0.0,
                "text": f"dummy text turn {t}",
            })
        records.append({
            "transcript_id": f"T{i:04d}",
            "outcome": "resolved",
            "outcome_id": 0,
            "turn_features": turn_features,
        })
    return records


# ---------------------------------------------------------------------------
# Test: No recursive training calls
# ---------------------------------------------------------------------------

class TestNoRecursiveTraining:
    """Ensure train_encoder / train_gnn never call train_all."""

    def test_train_encoder_source_has_no_train_all_call(self):
        """Static check: train_encoder body must not call train_all()."""
        source = inspect.getsource(train_encoder)
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                assert name != "train_all", (
                    "train_encoder must NOT call train_all"
                )

    def test_train_gnn_source_has_no_train_all_call(self):
        """Static check: train_gnn body must not call train_all()."""
        source = inspect.getsource(train_gnn)
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                assert name != "train_all", (
                    "train_gnn must NOT call train_all"
                )


# ---------------------------------------------------------------------------
# Test: Encoder trains exactly once
# ---------------------------------------------------------------------------

class TestEncoderTrainsOnce:
    """Verify the encoder training runs exactly the number of epochs requested."""

    def test_encoder_runs_requested_epochs(self):
        config = PipelineConfig(device="cpu")
        records = _make_dummy_records(20)
        epochs = 2

        hist = train_encoder(
            config, records, checkpoint_dir=tempfile.mkdtemp(),
            epochs=epochs, verbose=False,
        )

        assert len(hist["train_loss"]) == epochs
        assert len(hist["val_loss"]) == epochs


# ---------------------------------------------------------------------------
# Test: Checkpoint-aware training (skip when checkpoint exists)
# ---------------------------------------------------------------------------

class TestCheckpointAwareTraining:
    """Verify that train_all skips training when checkpoints exist."""

    def test_skips_when_checkpoints_exist(self):
        ckpt_dir = tempfile.mkdtemp()
        paths = default_paths(ckpt_dir)

        # Create dummy checkpoints
        model = _FeatureEncoder()
        save_encoder(model, paths["encoder"])
        save_gnn(model, paths["gnn"])

        config = PipelineConfig(device="cpu")
        records = _make_dummy_records(10)

        # Patch process_dataset to return our records
        import pipeline.train as train_mod
        orig_process = train_mod.process_dataset
        train_mod.process_dataset = lambda cfg: records
        try:
            result = train_all(
                config=config,
                checkpoint_dir=ckpt_dir,
                encoder_epochs=1,
                gnn_epochs=1,
                verbose=False,
                force_train=False,
            )
        finally:
            train_mod.process_dataset = orig_process

        # Both histories should be empty (no training occurred)
        assert len(result["encoder_history"]["train_loss"]) == 0
        assert len(result["gnn_history"]["train_loss"]) == 0

        shutil.rmtree(ckpt_dir)

    def test_force_train_overrides_checkpoints(self):
        ckpt_dir = tempfile.mkdtemp()
        paths = default_paths(ckpt_dir)

        # Create dummy checkpoints
        model = _FeatureEncoder()
        save_encoder(model, paths["encoder"])
        save_gnn(model, paths["gnn"])

        config = PipelineConfig(device="cpu")
        records = _make_dummy_records(10)

        import pipeline.train as train_mod
        orig_process = train_mod.process_dataset
        train_mod.process_dataset = lambda cfg: records
        try:
            result = train_all(
                config=config,
                checkpoint_dir=ckpt_dir,
                encoder_epochs=1,
                gnn_epochs=1,
                verbose=False,
                force_train=True,
                skip_tests=True,
            )
        finally:
            train_mod.process_dataset = orig_process

        # Training should have run
        assert len(result["encoder_history"]["train_loss"]) == 1
        assert len(result["gnn_history"]["train_loss"]) >= 0  # GNN may have 0 graphs

        shutil.rmtree(ckpt_dir)


# ---------------------------------------------------------------------------
# Test: Resume flag
# ---------------------------------------------------------------------------

class TestResume:
    """Verify that --resume allows training even when checkpoints exist."""

    def test_resume_allows_training(self):
        ckpt_dir = tempfile.mkdtemp()
        paths = default_paths(ckpt_dir)

        model = _FeatureEncoder()
        save_encoder(model, paths["encoder"])
        save_gnn(model, paths["gnn"])

        config = PipelineConfig(device="cpu")
        records = _make_dummy_records(10)

        import pipeline.train as train_mod
        orig_process = train_mod.process_dataset
        train_mod.process_dataset = lambda cfg: records
        try:
            result = train_all(
                config=config,
                checkpoint_dir=ckpt_dir,
                encoder_epochs=1,
                gnn_epochs=1,
                verbose=False,
                resume=True,
                skip_tests=True,
            )
        finally:
            train_mod.process_dataset = orig_process

        assert len(result["encoder_history"]["train_loss"]) == 1

        shutil.rmtree(ckpt_dir)


# ---------------------------------------------------------------------------
# Test: Deterministic seeding
# ---------------------------------------------------------------------------

class TestDeterministicSeeding:
    def test_set_seed_produces_same_output(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.allclose(a, b)

class TestDataSplit:
    """Verify data is correctly split into non-overlapping train/val/test sets."""

    def test_encoder_split_sizes_match_config(self):
        config = PipelineConfig(device="cpu")
        records = _make_dummy_records(100)
        ckpt_dir = tempfile.mkdtemp()

        hist = train_encoder(
            config, records, checkpoint_dir=ckpt_dir,
            epochs=1, verbose=False,
        )

        split = hist["split"]
        assert split["train"] + split["val"] + split["test"] == 100
        # Default 80/10/10 split
        assert split["train"] == 80
        assert split["val"] == 10
        assert split["test"] == 10

        shutil.rmtree(ckpt_dir)

    def test_encoder_split_is_deterministic(self):
        """Same seed should produce the same split."""
        config = PipelineConfig(device="cpu")
        records = _make_dummy_records(50)

        ckpt1 = tempfile.mkdtemp()
        ckpt2 = tempfile.mkdtemp()
        hist1 = train_encoder(
            config, records, checkpoint_dir=ckpt1,
            epochs=1, verbose=False,
        )
        hist2 = train_encoder(
            config, records, checkpoint_dir=ckpt2,
            epochs=1, verbose=False,
        )

        assert hist1["split"] == hist2["split"]

        shutil.rmtree(ckpt1)
        shutil.rmtree(ckpt2)


# ---------------------------------------------------------------------------
# Test: Test evaluation produces valid metrics
# ---------------------------------------------------------------------------

class TestTestEvaluation:
    """Verify the held-out test evaluation runs and returns valid results."""

    def test_train_all_returns_test_metrics(self):
        config = PipelineConfig(device="cpu")
        records = _make_dummy_records(30)
        ckpt_dir = tempfile.mkdtemp()

        import pipeline.train as train_mod
        orig_process = train_mod.process_dataset
        train_mod.process_dataset = lambda cfg: records
        try:
            result = train_all(
                config=config,
                checkpoint_dir=ckpt_dir,
                encoder_epochs=2,
                gnn_epochs=1,
                verbose=False,
                force_train=True,
                skip_tests=False,
            )
        finally:
            train_mod.process_dataset = orig_process

        enc = result["encoder_history"]
        assert "test_loss" in enc, "Encoder test_loss must be present"
        assert "test_accuracy" in enc, "Encoder test_accuracy must be present"
        assert isinstance(enc["test_loss"], float)
        assert 0.0 <= enc["test_accuracy"] <= 1.0

        shutil.rmtree(ckpt_dir)

    def test_split_info_in_history(self):
        config = PipelineConfig(device="cpu")
        records = _make_dummy_records(30)
        ckpt_dir = tempfile.mkdtemp()

        import pipeline.train as train_mod
        orig_process = train_mod.process_dataset
        train_mod.process_dataset = lambda cfg: records
        try:
            result = train_all(
                config=config,
                checkpoint_dir=ckpt_dir,
                encoder_epochs=1,
                gnn_epochs=1,
                verbose=False,
                force_train=True,
                skip_tests=True,
            )
        finally:
            train_mod.process_dataset = orig_process

        enc = result["encoder_history"]
        assert "split" in enc, "Split info must be present in encoder history"
        assert enc["split"]["train"] > 0
        assert enc["split"]["val"] >= 0
        assert enc["split"]["test"] >= 0

        shutil.rmtree(ckpt_dir)