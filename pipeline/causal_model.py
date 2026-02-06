from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import CausalConfig


# ── DAG definition ────────────────────────────────────────────────────────

# Edges represent direct causal relationships.
DEFAULT_DAG_EDGES: List[Tuple[str, str]] = [
    ("delay", "agent_response_quality"),
    ("delay", "customer_anger"),
    ("repetition", "customer_anger"),
    ("repetition", "resolution_time"),
    ("agent_response_quality", "customer_anger"),
    ("agent_response_quality", "resolution_time"),
    ("customer_anger", "escalation"),
    ("resolution_time", "escalation"),
]


class CausalDAG:

    def __init__(
        self,
        variables: List[str],
        edges: Optional[List[Tuple[str, str]]] = None,
    ):
        self.variables = variables
        self.edges = edges if edges is not None else list(DEFAULT_DAG_EDGES)
        self._adj: Dict[str, List[str]] = {v: [] for v in variables}
        for src, tgt in self.edges:
            if src in self._adj:
                self._adj[src].append(tgt)

    def parents(self, node: str) -> List[str]:
        """Return direct parents of *node*."""
        return [src for src, tgt in self.edges if tgt == node]

    def children(self, node: str) -> List[str]:
        """Return direct children of *node*."""
        return self._adj.get(node, [])

    def ancestors(self, node: str) -> set:
        """Return all ancestors of *node* via BFS."""
        visited: set = set()
        queue = self.parents(node)
        while queue:
            current = queue.pop(0)
            if current not in visited:
                visited.add(current)
                queue.extend(self.parents(current))
        return visited

    def topological_sort(self) -> List[str]:
        """Kahn's algorithm for topological ordering."""
        in_degree = {v: 0 for v in self.variables}
        for _, tgt in self.edges:
            in_degree[tgt] = in_degree.get(tgt, 0) + 1
        queue = [v for v in self.variables if in_degree[v] == 0]
        order: List[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in self.children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return order

    def to_dot(self) -> str:
        """Return a Graphviz DOT string for visualisation."""
        lines = ["digraph CausalDAG {", "  rankdir=LR;"]
        for src, tgt in self.edges:
            lines.append(f'  "{src}" -> "{tgt}";')
        lines.append("}")
        return "\n".join(lines)


# ── Feature extraction for causal variables ───────────────────────────────

def extract_causal_variables(conversation_record: dict) -> Dict[str, float]:
    turn_feats = conversation_record.get("turn_features", [])
    num_turns = max(len(turn_feats), 1)

    # Delay: average delay-keyword score across turns
    delay_scores = [tf.get("discourse_delay", 0.0) for tf in turn_feats]
    delay = float(np.mean(delay_scores)) if delay_scores else 0.0

    # Repetition: fraction of customer turns that repeat prior complaint keywords
    customer_texts = [
        tf["text"].lower() for tf in turn_feats if not tf.get("is_agent", 0)
    ]
    repetition = 0.0
    if len(customer_texts) > 1:
        repeat_count = sum(
            1 for i in range(1, len(customer_texts))
            if any(w in customer_texts[i] for w in customer_texts[0].split()[:5])
        )
        repetition = repeat_count / max(len(customer_texts) - 1, 1)

    # Agent response quality: inverse of denial + delay + low word count
    agent_feats = [tf for tf in turn_feats if tf.get("is_agent", 0)]
    if agent_feats:
        avg_denial = float(np.mean([tf.get("discourse_denial", 0.0) for tf in agent_feats]))
        avg_wc = float(np.mean([tf.get("word_count", 0) for tf in agent_feats]))
        quality = max(0.0, 1.0 - avg_denial) * min(avg_wc / 50.0, 1.0)
    else:
        quality = 0.5

    # Customer anger
    anger = conversation_record.get("max_anger", 0.0)
    frustration = conversation_record.get("max_frustration", 0.0)
    customer_anger = min(1.0, (anger + frustration) / 2.0)

    # Resolution time proxy: normalised turn count (more turns → longer)
    resolution_time = min(num_turns / 30.0, 1.0)

    # Escalation: binary from label
    escalation = float(conversation_record.get("has_escalation_request", 0))

    return {
        "delay": delay,
        "repetition": repetition,
        "agent_response_quality": quality,
        "customer_anger": customer_anger,
        "resolution_time": resolution_time,
        "escalation": escalation,
    }


# ── Causal effect estimation ─────────────────────────────────────────────

def estimate_causal_effect(
    data: List[Dict[str, float]],
    treatment: str,
    outcome: str,
    dag: CausalDAG,
    config: CausalConfig,
) -> Dict[str, Any]:
    if not data:
        return {"ate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                "confounders": [], "n_samples": 0}

    # Identify confounders via back-door criterion:
    # parents of treatment ∩ ancestors of outcome
    outcome_ancestors = dag.ancestors(outcome)
    treatment_parents = set(dag.parents(treatment))
    confounders = list(treatment_parents & (outcome_ancestors | {outcome}))

    t_vals = np.array([d[treatment] for d in data])
    y_vals = np.array([d[outcome] for d in data])

    # Median split for treatment → treated / control
    median_t = float(np.median(t_vals))
    treated_mask = t_vals >= median_t
    control_mask = ~treated_mask

    if treated_mask.sum() == 0 or control_mask.sum() == 0:
        ate = 0.0
    else:
        ate = float(y_vals[treated_mask].mean() - y_vals[control_mask].mean())

    # Bootstrap confidence interval
    bootstrap_rng = np.random.RandomState(config.n_bootstrap)
    boot_ates: List[float] = []
    n = len(data)
    for _ in range(config.n_bootstrap):
        idx = bootstrap_rng.choice(n, size=n, replace=True)
        t_b = t_vals[idx]
        y_b = y_vals[idx]
        med = float(np.median(t_b))
        tr = t_b >= med
        ct = ~tr
        if tr.sum() > 0 and ct.sum() > 0:
            boot_ates.append(float(y_b[tr].mean() - y_b[ct].mean()))

    if boot_ates:
        ci_lower = float(np.percentile(boot_ates, 2.5))
        ci_upper = float(np.percentile(boot_ates, 97.5))
    else:
        ci_lower, ci_upper = ate, ate

    return {
        "ate": ate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confounders": confounders,
        "n_samples": len(data),
    }


def counterfactual_query(
    observation: Dict[str, float],
    intervention: Dict[str, float],
    dag: CausalDAG,
) -> Dict[str, float]:
    cf = dict(observation)
    cf.update(intervention)

    for var in dag.topological_sort():
        if var in intervention:
            cf[var] = intervention[var]
            continue
        parents = dag.parents(var)
        if parents:
            parent_vals = [cf.get(p, 0.0) for p in parents]
            # Simple linear propagation (mean of parents)
            cf[var] = float(np.mean(parent_vals))

    return cf


def identify_root_causes(
    data: List[Dict[str, float]],
    outcome: str,
    dag: CausalDAG,
    config: CausalConfig,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for var in dag.variables:
        if var == outcome:
            continue
        effect = estimate_causal_effect(data, var, outcome, dag, config)
        results.append({"variable": var, **effect})
    results.sort(key=lambda r: abs(r["ate"]), reverse=True)
    return results
