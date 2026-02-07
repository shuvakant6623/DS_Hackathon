# Causal Analysis Pipeline for Conversational Data
A production-grade causal analysis system for customer service conversations. It identifies causal relationships between discourse patterns, emotional states, and escalation outcomes using a 4-layer architecture — feature extraction, BERT encoding, graph neural networks, and causal inference — to generate explainable, actionable recommendations.
---
## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Running the Pipeline](#running-the-pipeline)
- [Evaluation & Benchmarking](#evaluation--benchmarking)
- [Visualization & EDA](#visualization--eda)
- [Configuration Reference](#configuration-reference)
- [Testing](#testing)
---

# Team Name : Titanic swimming team
## Shuvakant Patra
## Shreyans Behera
## Subodh Kumar Swain

## Architecture Overview
The system is organized into four layers:

| Layer | Component | Description |
|-------|-----------|-------------|
| **Layer 1** | Feature Encoder | 17-dim turn features → MLP + BERT transformer with emotion/outcome classification heads |
| **Layer 2** | Discourse GNN | Multi-layer Graph Attention Network (GAT) over directed discourse graphs |
| **Layer 3** | Causal Model | Causal DAG construction + Average Treatment Effect (ATE) estimation via DoWhy |
| **Layer 4** | Explanation | Evidence retrieval, counterfactual analysis, and causal chain generation |
**End-to-end flow:**
Raw transcripts → Feature extraction → Encoding → Graph construction → Causal inference → Explainable results
---
## Prerequisites
- **Python** 3.9 or higher
- **pip** (Python package manager)
- **Git**
- **(Optional)** NVIDIA GPU with CUDA for accelerated training and inference
---
## Environment Setup
### 1. Clone the repository
```bash
git clone https://github.com/shuvakant6623/DS_Hackathon.git
cd DS_Hackathon
```
### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> **Note:** `torch` and `torch-geometric` may require platform-specific installation. Refer to the [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) docs if the default install does not work for your system.
---
## Data Preprocessing
The dataset consists of 5,037 customer service conversations across 7 domains with outcome labels.
### Raw → Processed (one-time step)
Open and run the data preparation notebook:
```bash
jupyter notebook Modeling/processData.ipynb
```
This reads the raw JSON transcript data from `Datasets/raw/` and produces two processed files used by the pipeline:

| Output File | Description |
|-------------|-------------|
| `Datasets/processed/transcript_dataset.csv` | Flat CSV with conversation metadata and outcome labels |
| `Datasets/processed/conversation_transcript_map.json` | Structured JSON mapping transcript IDs to dialogue turns |
### Feature extraction (automatic)
When the pipeline runs, `pipeline/data_processing.py` automatically extracts 17-dimensional features per dialogue turn:
- **Structural:** speaker role, position, word/character counts, punctuation
- **Emotion scores:** anger, frustration, satisfaction, confusion, urgency
- **Discourse edge types:** complaint, denial, delay, apology, clarification, promise, escalation request
No manual step is needed — feature extraction is performed on-the-fly during training and inference.
---
## Model Training
### Train all models (encoder + GNN + causal)
```bash
python -m pipeline.run_training
```
### Common training options
```bash
# Train with a GPU
python -m pipeline.run_training --device cuda
# Override epoch counts
python -m pipeline.run_training --encoder-epochs 5 --gnn-epochs 15
# Train only the encoder
python -m pipeline.run_training --train-encoder
# Train only the GNN
python -m pipeline.run_training --train-gnn
# Resume from existing checkpoints
python -m pipeline.run_training --resume
# Force retraining (ignore existing checkpoints)
python -m pipeline.run_training --force-train
# Generate a technical report after training
python -m pipeline.run_training --report
```
### Training defaults
| Component | Epochs | Batch Size | Learning Rate | Optimizer |
|-----------|--------|------------|---------------|-----------|
| Encoder (BERT) | 10 | 16 | 2 × 10⁻⁵ | Adam |
| Discourse GNN | 30 | 32 | 1 × 10⁻³ | Adam |
| Causal Model | — | — | — | 100 bootstrap samples |
Model checkpoints are saved to the `checkpoints/` directory and automatically reused in subsequent runs.
---
## Running the Pipeline
### Analyse a subset of conversations
```bash
python run_pipeline.py --max-records 10
```
### Analyse all conversations
```bash
python run_pipeline.py --all
```
### Use GPU acceleration
```bash
python run_pipeline.py --device cuda
```
### Ask an interactive follow-up query
```bash
python run_pipeline.py --query "Why did this conversation escalate?"
```
### Example output
```
Causal DAG (DOT format):
  digraph { delay -> escalation; customer_anger -> escalation; ... }
Counterfactual analysis (do(delay=0)):
  escalation: 0.1234
BATCH ANALYSIS SUMMARY (10 conversations)
  T001: outcome=escalated, ATE=0.3201, chain=delay → customer_anger → escalation
  T002: outcome=resolved, ATE=0.0412, chain=apology → satisfaction → resolution
  ...
```
---
## Evaluation & Benchmarking
### Run the evaluation suite
```bash
python -m pipeline.run_evaluate
```
### Evaluation options
```bash
# Evaluate on a subset
python -m pipeline.run_evaluate --max-records 50
# Evaluate on GPU and generate a report
python -m pipeline.run_evaluate --device cuda --report
```
Metrics computed include faithfulness, recall, and causal consistency of generated explanations.
### Generate benchmark queries
```bash
python generate_queries.py
```
This produces `queries.csv` with complex conversational analysis queries and expected outputs for benchmarking.
---
## Visualization & EDA
### Exploratory Data Analysis notebook
Launch the EDA notebook for comprehensive visualizations of the dataset:
```bash
jupyter notebook EDA.ipynb
```
The notebook includes:
- **Outcome distribution** — bar charts showing conversation resolution vs. escalation rates
- **Domain analysis** — breakdowns across the 7 service domains
- **Emotion patterns** — heatmaps of emotion keyword frequency across turns
- **Turn-level statistics** — distributions of word counts, response times, and conversation lengths
- **Discourse patterns** — frequency of complaint, denial, delay, and other discourse acts
- **Correlation analysis** — relationships between features and escalation outcomes
### Causal DAG visualization
When running the pipeline, a causal DAG is printed in DOT format. To render it as an image:
```bash
python run_pipeline.py --max-records 1 > output.txt
# Extract the DOT block and render with Graphviz:
dot -Tpng causal_dag.dot -o causal_dag.png
```
### Technical reports
Training and evaluation can generate detailed Markdown reports:
```bash
python -m pipeline.run_training --report    # outputs/technical_report.md
python -m pipeline.run_evaluate --report    # outputs/technical_report.md
```
---
## Configuration Reference
All hyperparameters are managed via dataclasses in `pipeline/config.py`. Override them programmatically:
```python
from pipeline.config import PipelineConfig
config = PipelineConfig(device="cuda")
config.encoder.epochs = 20
config.encoder.learning_rate = 1e-5
config.discourse.gnn_num_layers = 4
config.causal.treatment = "delay"
config.causal.outcome = "escalation"
```
### Key configuration groups
| Group | Key Parameters |
|-------|----------------|
| `DataConfig` | `csv_path`, `json_path`, `max_turns`, `val_size`, `test_size`, `random_seed` |
| `EncoderConfig` | `model_name`, `hidden_dim`, `dropout`, `learning_rate`, `epochs`, `batch_size` |
| `DiscourseConfig` | `edge_types`, `gnn_hidden_dim`, `gnn_num_layers`, `gnn_heads`, `epochs` |
| `CausalConfig` | `causal_variables`, `treatment`, `outcome`, `n_bootstrap`, `significance_level` |
| `ExplanationConfig` | `max_evidence_turns`, `temperature`, `max_generation_len`, `context_window` |
---
## Testing
Run the full test suite:
```bash
python -m pytest tests/ -v
```
Run individual test modules:
```bash
python -m pytest tests/test_training.py -v      # Training tests
python -m pytest tests/test_inference.py -v      # Inference tests
python -m pytest tests/test_eval.py -v           # Evaluation tests
python -m pytest tests/test_generate_queries.py -v  # Query generation tests
```