import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DiscourseConfig
from .constants import DISCOURSE_KEYWORDS


# ── edge-type detection (keyword-based bootstrap) ────────────────────────

_EDGE_KEYWORDS = DISCOURSE_KEYWORDS


def detect_edge_type(source_text: str, target_text: str) -> Optional[str]:
    combined = (source_text + " " + target_text).lower()
    best_type: Optional[str] = None
    best_score = 0.0
    for etype, keywords in _EDGE_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in combined)
        score = hits / len(keywords)
        if score > best_score and score > 0:
            best_score = score
            best_type = etype
    return best_type


def build_discourse_graph(
    turns: List[dict],
    turn_embeddings: torch.Tensor,
    edge_types: List[str],
) -> dict:
    etype_to_idx = {et: i for i, et in enumerate(edge_types)}
    src_list, tgt_list, attr_list, label_list = [], [], [], []

    for i in range(len(turns)):
        # Connect to next and +2 turns (captures adjacency + skip relations)
        for offset in (1, 2):
            j = i + offset
            if j >= len(turns):
                break
            etype = detect_edge_type(turns[i]["text"], turns[j]["text"])
            if etype is None:
                etype = "clarification"  # default relation for adjacent turns
            src_list.append(i)
            tgt_list.append(j)
            attr_list.append(etype_to_idx.get(etype, 0))
            label_list.append(etype)

    edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long)
    edge_attr = torch.tensor(attr_list, dtype=torch.long)

    return {
        "node_features": turn_embeddings,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_labels": label_list,
    }


# ── GNN model ────────────────────────────────────────────────────────────

class GraphAttentionLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, in_dim)
        edge_index : (2, E)

        Returns
        -------
        (N, out_dim)
        """
        h = self.W(x)  # (N, out_dim)
        src, tgt = edge_index  # each (E,)

        # Compute attention scores
        edge_h = torch.cat([h[src], h[tgt]], dim=-1)  # (E, 2*out_dim)
        e = self.leaky_relu(self.attn(edge_h)).squeeze(-1)  # (E,)

        # Softmax over neighbours
        num_nodes = x.size(0)
        alpha = torch.zeros(num_nodes, num_nodes, device=x.device)
        alpha[src, tgt] = e
        alpha = F.softmax(alpha, dim=-1)
        alpha = self.dropout(alpha)

        # Aggregate
        out = torch.matmul(alpha, h)  # (N, out_dim)
        return out


class DiscourseGNN(nn.Module):

    def __init__(self, config: DiscourseConfig, input_dim: int):
        super().__init__()
        self.config = config
        hidden = config.gnn_hidden_dim

        self.input_proj = nn.Linear(input_dim, hidden)

        self.gat_layers = nn.ModuleList()
        for _ in range(config.gnn_num_layers):
            self.gat_layers.append(GraphAttentionLayer(hidden, hidden, config.dropout))

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(config.gnn_num_layers)]
        )

        # Attention pooling
        self.pool_gate = nn.Linear(hidden, 1)

        # Edge-type classifier (optional supervised signal)
        self.edge_classifier = nn.Linear(2 * hidden, len(config.edge_types))

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict:
        """
        Parameters
        ----------
        node_features : (N, D_in)
        edge_index : (2, E)

        Returns
        -------
        dict with keys:
            node_embeddings  – (N, H)
            graph_embedding  – (H,)
        """
        h = self.input_proj(node_features)  # (N, H)

        for gat, ln in zip(self.gat_layers, self.layer_norms):
            h_new = gat(h, edge_index)
            h = ln(h + h_new)  # residual + layer-norm
            h = F.elu(h)

        # Attention pooling → single graph-level embedding
        gate = torch.sigmoid(self.pool_gate(h))  # (N, 1)
        graph_emb = (gate * h).sum(dim=0)          # (H,)

        return {
            "node_embeddings": h,
            "graph_embedding": graph_emb,
        }

    def classify_edges(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Predict edge types from node embeddings.  Returns (E, num_edge_types)."""
        src, tgt = edge_index
        edge_repr = torch.cat([node_embeddings[src], node_embeddings[tgt]], dim=-1)
        return self.edge_classifier(edge_repr)


class DiscourseGraphLoss(nn.Module):
    """Loss for edge-type classification on the discourse graph."""

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        edge_logits: torch.Tensor,
        edge_labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.ce(edge_logits, edge_labels)
