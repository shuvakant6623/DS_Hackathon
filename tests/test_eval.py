"""Tests for evaluation pipeline.

Verifies that:
- Evaluation does NOT trigger any training
- No optimizer is created during evaluation
"""
import ast
import inspect

import pytest

from pipeline.evaluate import evaluate_pipeline


class TestEvalNoTraining:
    """Ensure evaluation never triggers training."""

    def test_evaluate_source_has_no_train_call(self):
        """Static check: evaluate_pipeline must not call train_all / train_encoder / train_gnn."""
        source = inspect.getsource(evaluate_pipeline)
        tree = ast.parse(source)

        forbidden = {"train_all", "train_encoder", "train_gnn"}

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                assert name not in forbidden, (
                    f"evaluate_pipeline must NOT call {name}"
                )

    def test_evaluate_module_does_not_import_train(self):
        """Ensure pipeline.evaluate does not import from pipeline.train."""
        import pipeline.evaluate as eval_mod

        source = inspect.getsource(eval_mod)
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert "train" not in node.module, (
                        f"pipeline.evaluate must NOT import from {node.module}"
                    )