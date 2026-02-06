import os
from datetime import datetime
from typing import Any, Dict, Optional


def generate_report(
    training_history: Optional[Dict[str, Any]] = None,
    evaluation_results: Optional[Dict[str, Any]] = None,
    output_path: str = "outputs/technical_report.md",
) -> str:
    lines = []

    lines.append("# Technical Report: Causal Analysis and Interactive Reasoning")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## 1. Approach Overview\n")
    lines.append(
        "This system implements a 4-layer causal analysis pipeline for "
        "conversational data:\n"
    )
    lines.append("1. **Layer 0 – Data Processing**: Load transcripts, extract turn-level features")
    lines.append("   (speaker, position, emotion keywords, discourse keywords).")
    lines.append("2. **Layer 1 – Encoder**: MLP-based feature encoder with emotion and outcome")
    lines.append("   classification heads. Produces fixed-size turn embeddings.")
    lines.append("3. **Layer 2 – Discourse Graph**: Builds a directed discourse graph over turns")
    lines.append("   and runs a multi-layer Graph Attention Network (GAT) for structural reasoning.")
    lines.append("4. **Layer 3 – Causal Model**: Constructs a causal DAG with domain-defined edges,")
    lines.append("   estimates Average Treatment Effects (ATE) via back-door adjustment, and")
    lines.append("   supports counterfactual queries.")
    lines.append("5. **Layer 4 – Explanation & Interaction**: Retrieves evidence turns, ranks by")
    lines.append("   faithfulness, generates structured explanations, and maintains multi-turn")
    lines.append("   interaction context.\n")

    lines.append("## 2. Model Architecture\n")
    lines.append("### Feature Encoder (Layer 1)\n")
    lines.append("- Input: 17-dimensional feature vector per turn")
    lines.append("- Architecture: 2-layer MLP (17 → 64 → 64) with ReLU + Dropout")
    lines.append("- Heads: Emotion (6 classes), Outcome (5 classes)")
    lines.append("- Training: Cross-entropy loss, Adam optimizer\n")
    lines.append("### Discourse GNN (Layer 2)\n")
    lines.append("- Architecture: 3-layer GAT with residual connections + LayerNorm")
    lines.append("- Hidden dimension: 256, Attention heads: 4")
    lines.append("- Pooling: Attention-weighted graph-level embedding")
    lines.append("- Edge classifier for discourse relation types\n")
    lines.append("### Causal DAG (Layer 3)\n")
    lines.append("- Variables: delay, repetition, agent_response_quality, customer_anger,")
    lines.append("  resolution_time, escalation")
    lines.append("- 8 directed edges encoding domain causal knowledge")
    lines.append("- ATE estimation via median-split back-door adjustment")
    lines.append("- Bootstrap confidence intervals (n=100)\n")

    lines.append("## 3. Training Results\n")
    if training_history:
        enc = training_history.get("encoder_history", {})
        gnn = training_history.get("gnn_history", {})

        if enc.get("train_loss"):
            lines.append("### Encoder Training\n")
            lines.append(f"- Epochs: {len(enc['train_loss'])}")
            lines.append(f"- Final training loss: {enc['train_loss'][-1]:.4f}")
            if enc.get("val_loss"):
                lines.append(f"- Final validation loss: {enc['val_loss'][-1]:.4f}")
            if enc.get("val_accuracy"):
                lines.append(f"- Final validation accuracy: {enc['val_accuracy'][-1]:.4f}")
            lines.append("")

        if gnn.get("train_loss"):
            lines.append("### GNN Training\n")
            lines.append(f"- Epochs: {len(gnn['train_loss'])}")
            lines.append(f"- Final training loss: {gnn['train_loss'][-1]:.4f}")
            if gnn.get("val_loss"):
                lines.append(f"- Final validation loss: {gnn['val_loss'][-1]:.4f}")
            if gnn.get("val_accuracy"):
                lines.append(f"- Final validation accuracy: {gnn['val_accuracy'][-1]:.4f}")
            lines.append("")
    else:
        lines.append("Training history not available.\n")

    lines.append("## 4. Evaluation Results\n")
    if evaluation_results:
        metrics = evaluation_results.get("metrics", {})
        lines.append(f"| Metric | Score |")
        lines.append(f"|--------|-------|")
        lines.append(f"| ID Recall (Evidence Accuracy) | {metrics.get('id_recall', 0):.4f} |")
        lines.append(f"| Faithfulness (Hallucination Control) | {metrics.get('faithfulness', 0):.4f} |")
        lines.append(f"| Relevancy (Conversational Coherence) | {metrics.get('relevancy', 0):.4f} |")
        lines.append(f"| Outcome Accuracy | {metrics.get('outcome_accuracy', 0):.4f} |")
        lines.append("")
    else:
        lines.append("Evaluation results not available.\n")

    lines.append("## 5. Task Coverage\n")
    lines.append("### Task 1: Query-Driven Causal Explanation\n")
    lines.append("- Accepts natural-language queries about conversation outcomes")
    lines.append("- Analyses relevant conversations using the causal DAG")
    lines.append("- Extracts evidence turns and ranks by faithfulness")
    lines.append("- Produces structured explanations with inline citations\n")
    lines.append("### Task 2: Multi-Turn Context-Aware Interaction\n")
    lines.append("- InteractionContext maintains sliding window of prior queries")
    lines.append("- Supports 'why', 'evidence', and 'what if' follow-up types")
    lines.append("- Preserves causal chain and evidence across turns")
    lines.append("- Deterministic context handling with explicit state tracking\n")

    lines.append("## 6. Reproducibility\n")
    lines.append("- All random seeds are fixed (default: 42)")
    lines.append("- Training runs on CPU by default")
    lines.append("- Dependencies listed in `requirements.txt`")
    lines.append("- Checkpoints saved to `checkpoints/` directory")
    lines.append("- Training: `python run_training.py`")
    lines.append("- Evaluation: `python run_evaluate.py`")
    lines.append("- Inference: `python run_pipeline.py`\n")

    report_text = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_text)

    return report_text