# Technical Report: Causal Analysis and Interactive Reasoning

Generated: 2026-02-07 06:03:42

## 1. Approach Overview

This system implements a 4-layer causal analysis pipeline for conversational data:

1. **Layer 0 – Data Processing**: Load transcripts, extract turn-level features
   (speaker, position, emotion keywords, discourse keywords).
2. **Layer 1 – Encoder**: MLP-based feature encoder with emotion and outcome
   classification heads. Produces fixed-size turn embeddings.
3. **Layer 2 – Discourse Graph**: Builds a directed discourse graph over turns
   and runs a multi-layer Graph Attention Network (GAT) for structural reasoning.
4. **Layer 3 – Causal Model**: Constructs a causal DAG with domain-defined edges,
   estimates Average Treatment Effects (ATE) via back-door adjustment, and
   supports counterfactual queries.
5. **Layer 4 – Explanation & Interaction**: Retrieves evidence turns, ranks by
   faithfulness, generates structured explanations, and maintains multi-turn
   interaction context.

## 2. Model Architecture

### Feature Encoder (Layer 1)

- Input: 17-dimensional feature vector per turn
- Architecture: 2-layer MLP (17 → 64 → 64) with ReLU + Dropout
- Heads: Emotion (6 classes), Outcome (5 classes)
- Training: Cross-entropy loss, Adam optimizer

### Discourse GNN (Layer 2)

- Architecture: 3-layer GAT with residual connections + LayerNorm
- Hidden dimension: 256, Attention heads: 4
- Pooling: Attention-weighted graph-level embedding
- Edge classifier for discourse relation types

### Causal DAG (Layer 3)

- Variables: delay, repetition, agent_response_quality, customer_anger,
  resolution_time, escalation
- 8 directed edges encoding domain causal knowledge
- ATE estimation via median-split back-door adjustment
- Bootstrap confidence intervals (n=100)

## 3. Training Results

Training history not available.

## 4. Evaluation Results

| Metric | Score |
|--------|-------|
| ID Recall (Evidence Accuracy) | 0.5087 |
| Faithfulness (Hallucination Control) | 0.6890 |
| Relevancy (Conversational Coherence) | 0.6753 |
| Outcome Accuracy | 0.9130 |

## 5. Task Coverage

### Task 1: Query-Driven Causal Explanation

- Accepts natural-language queries about conversation outcomes
- Analyses relevant conversations using the causal DAG
- Extracts evidence turns and ranks by faithfulness
- Produces structured explanations with inline citations

### Task 2: Multi-Turn Context-Aware Interaction

- InteractionContext maintains sliding window of prior queries
- Supports 'why', 'evidence', and 'what if' follow-up types
- Preserves causal chain and evidence across turns
- Deterministic context handling with explicit state tracking

## 6. Reproducibility

- All random seeds are fixed (default: 42)
- Training runs on CPU by default
- Dependencies listed in `requirements.txt`
- Checkpoints saved to `checkpoints/` directory
- Training: `python run_training.py`
- Evaluation: `python run_evaluate.py`
- Inference: `python run_pipeline.py`
