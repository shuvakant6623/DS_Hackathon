"""Tests for the inference pipeline.

Verifies that:
- Inference does NOT trigger any training
- No optimizer is created during inference
- No training imports exist in the inference entrypoint
"""
import ast
import inspect

import pytest

from pipeline.main import CausalAnalysisPipeline


class TestInferenceNoTraining:
    """Ensure inference never triggers training."""

    def test_main_module_does_not_import_train(self):
        """Ensure pipeline.main does not import from pipeline.train."""
        import pipeline.main as main_mod

        source = inspect.getsource(main_mod)
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert "train" not in node.module, (
                        f"pipeline.main must NOT import from {node.module}"
                    )

    def test_run_pipeline_does_not_import_train(self):
        """Ensure run_pipeline.py does not import from pipeline.train."""
        with open("run_pipeline.py", "r") as f:
            source = f.read()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert "train" not in node.module, (
                        f"run_pipeline.py must NOT import from {node.module}"
                    )

    def test_no_optimizer_in_inference_pipeline(self):
        """Ensure CausalAnalysisPipeline does not create optimizers."""
        source = inspect.getsource(CausalAnalysisPipeline)
        assert "optim." not in source, (
            "CausalAnalysisPipeline must NOT use torch.optim"
        )
        assert "Optimizer" not in source, (
            "CausalAnalysisPipeline must NOT reference Optimizer"
        )
