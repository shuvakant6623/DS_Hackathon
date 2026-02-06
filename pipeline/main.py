from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .config import PipelineConfig
from .data_processing import process_dataset, build_conversation_features
from .discourse_graph import build_discourse_graph, DiscourseGNN
from .causal_model import (
    CausalDAG,
    extract_causal_variables,
    estimate_causal_effect,
    counterfactual_query,
    identify_root_causes,
)
from .explanation import (
    retrieve_evidence_turns,
    rank_evidence_by_faithfulness,
    generate_explanation,
    InteractionContext,
)
from .evaluation import compute_all_metrics

class CausalAnalysisPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.records: List[dict] = []
        self.causal_dag = CausalDAG(config.causal.causal_variables)
        self.interaction_ctx = InteractionContext(config.explanation)

        # Discourse GNN (initialised with default input dim; adjusted after encoding)
        self.discourse_gnn: Optional[DiscourseGNN] = None

    # ── Layer 0: data loading ─────────────────────────────────────────

    def load_data(self) -> None:
        """Load and preprocess the dataset."""
        self.records = process_dataset(self.config)

    # ── Layer 1: encoding (feature-based, no GPU needed) ──────────────

    def _encode_turns(self, turn_features: List[dict]) -> torch.Tensor:
        embed_dim = 32  # lightweight feature embedding
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
            # Pad to embed_dim
            vec.extend([0.0] * (embed_dim - len(vec)))
            embeddings.append(vec[:embed_dim])
        return torch.tensor(embeddings, dtype=torch.float32)

    def _build_graph(
        self,
        turn_features: List[dict],
        turn_embeddings: torch.Tensor,
    ) -> dict:
        graph = build_discourse_graph(
            turns=turn_features,
            turn_embeddings=turn_embeddings,
            edge_types=self.config.discourse.edge_types,
        )

        if self.discourse_gnn is None:
            self.discourse_gnn = DiscourseGNN(
                self.config.discourse,
                input_dim=turn_embeddings.shape[1],
            )

        with torch.no_grad():
            gnn_out = self.discourse_gnn(
                graph["node_features"],
                graph["edge_index"],
            )

        graph.update(gnn_out)
        return graph

    def _run_causal_analysis(
        self,
        record: dict,
        all_causal_data: Optional[List[Dict[str, float]]] = None,
    ) -> dict:
        cv = extract_causal_variables(record)

        # If we have population-level data, estimate ATE
        if all_causal_data is None:
            all_causal_data = [
                extract_causal_variables(r) for r in self.records
            ]

        ate = estimate_causal_effect(
            data=all_causal_data,
            treatment=self.config.causal.treatment,
            outcome=self.config.causal.outcome,
            dag=self.causal_dag,
            config=self.config.causal,
        )

        root_causes = identify_root_causes(
            data=all_causal_data,
            outcome=self.config.causal.outcome,
            dag=self.causal_dag,
            config=self.config.causal,
        )

        # Determine most likely causal chain
        chain = [rc["variable"] for rc in root_causes[:3]] + [self.config.causal.outcome]

        # Counterfactual: what if treatment had been zero?
        cf = counterfactual_query(
            observation=cv,
            intervention={self.config.causal.treatment: 0.0},
            dag=self.causal_dag,
        )

        return {
            "causal_variables": cv,
            "ate": ate,
            "root_causes": root_causes,
            "causal_chain": chain,
            "counterfactual": cf,
        }

    def _generate_explanation(
        self,
        record: dict,
        causal_result: dict,
    ) -> dict:
        """Retrieve evidence and generate explanation."""
        turn_features = record.get("turn_features", [])
        chain = causal_result["causal_chain"]

        # Evidence retrieval per causal variable
        evidence: Dict[str, list] = {}
        for var in chain:
            ev = retrieve_evidence_turns(
                turn_features,
                var,
                top_k=self.config.explanation.max_evidence_turns,
            )
            ev = rank_evidence_by_faithfulness(ev, chain, turn_features)
            evidence[var] = ev

        explanation_text = generate_explanation(
            causal_chain=chain,
            evidence=evidence,
            ate_results=causal_result["ate"],
            conversation_record=record,
        )

        return {
            "evidence": evidence,
            "explanation": explanation_text,
        }

    def analyse_conversation(self, record: dict) -> Dict[str, Any]:
        turn_features = record.get("turn_features", [])

        # Layer 1: Encode
        turn_embeddings = self._encode_turns(turn_features)

        # Layer 2: Discourse graph
        graph = self._build_graph(turn_features, turn_embeddings)

        # Layer 3: Causal analysis
        causal_result = self._run_causal_analysis(record)

        # Layer 4: Explanation
        expl_result = self._generate_explanation(record, causal_result)

        # Update interaction context
        self.interaction_ctx.set_context(
            transcript_id=record.get("transcript_id", ""),
            evidence=expl_result["evidence"],
            causal_chain=causal_result["causal_chain"],
        )

        return {
            "transcript_id": record.get("transcript_id"),
            "turn_embeddings": turn_embeddings,
            "graph": {
                "num_nodes": graph["node_features"].shape[0],
                "num_edges": graph["edge_index"].shape[1],
                "edge_labels": graph["edge_labels"],
                "graph_embedding_shape": graph["graph_embedding"].shape,
            },
            "causal": causal_result,
            "explanation": expl_result["explanation"],
            "evidence": expl_result["evidence"],
        }

    def analyse_all(self, max_records: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Run analysis on all (or first *max_records*) conversations.
        """
        records = self.records[:max_records] if max_records else self.records

        # Pre-compute causal data for ATE estimation
        all_causal = [extract_causal_variables(r) for r in self.records]

        results: List[Dict[str, Any]] = []
        for record in records:
            turn_features = record.get("turn_features", [])
            turn_embeddings = self._encode_turns(turn_features)
            graph = self._build_graph(turn_features, turn_embeddings)
            causal_result = self._run_causal_analysis(record, all_causal)
            expl_result = self._generate_explanation(record, causal_result)

            results.append({
                "transcript_id": record.get("transcript_id"),
                "outcome": record.get("outcome"),
                "causal_chain": causal_result["causal_chain"],
                "ate": causal_result["ate"]["ate"],
                "explanation": expl_result["explanation"],
            })

        return results

    def interactive_query(self, query: str) -> str:
        """Handle a follow-up query using the interaction context."""
        if not self.records:
            return "No data loaded. Call load_data() first."

        # Use the last analysed record as context
        if self.interaction_ctx.current_transcript_id:
            record = next(
                (r for r in self.records
                 if r["transcript_id"] == self.interaction_ctx.current_transcript_id),
                self.records[0],
            )
        else:
            record = self.records[0]

        return self.interaction_ctx.handle_query(query, record)
