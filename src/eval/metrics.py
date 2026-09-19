import re
from typing import List, Dict, Any, Optional

def compute_accuracy_matrix(results_matrix: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Computes standard Continual Learning metrics from an evaluation matrix R[t][i]
    where R[t][i] is accuracy on regime i after training on regime t.
    """
    regimes = list(results_matrix.keys())
    T = len(regimes)
    if T == 0:
        return {"avg_accuracy": 0.0, "bwt": 0.0, "fwt": 0.0}

    last_regime = regimes[-1]
    acc_seen = [results_matrix[last_regime].get(r, 0.0) for r in regimes]
    avg_acc = sum(acc_seen) / len(acc_seen) if acc_seen else 0.0

    # Backward Transfer (BWT)
    if T > 1:
        bwt_sum = 0.0
        for i_idx, r_i in enumerate(regimes[:-1]):
            acc_ti = results_matrix[last_regime].get(r_i, 0.0)
            acc_ii = results_matrix[r_i].get(r_i, 0.0)
            bwt_sum += (acc_ti - acc_ii)
        bwt = bwt_sum / (T - 1)
    else:
        bwt = 0.0

    # Forward Transfer (FWT)
    if T > 1:
        fwt_sum = 0.0
        for i_idx in range(1, T):
            r_prev = regimes[i_idx - 1]
            r_curr = regimes[i_idx]
            acc_prev_curr = results_matrix[r_prev].get(r_curr, 0.0)
            baseline = 0.10 # Zero-shot prior baseline
            fwt_sum += (acc_prev_curr - baseline)
        fwt = fwt_sum / (T - 1)
    else:
        fwt = 0.0

    return {
        "regimes_evaluated": regimes,
        "current_regime": last_regime,
        "avg_accuracy": round(avg_acc, 4),
        "bwt": round(bwt, 4),
        "fwt": round(fwt, 4),
        "matrix": results_matrix
    }

def evaluate_citation_hallucination(answer: str, valid_clauses: List[str], ground_truth_regime: str) -> Dict[str, Any]:
    """
    Evaluates precedent-citation hallucination specifically:
    1. Did the model cite non-existent clauses?
    2. Did the model misattribute clauses across regimes?
    """
    # Regex to capture patterns like 'Clause AI-101', 'Clause AUD-201', 'Clause INC-301', 'Clause SVR-401'
    clause_pattern = re.compile(r'\bClause\s+([A-Z]{2,4}-\d{3})\b', re.IGNORECASE)
    cited_clauses = clause_pattern.findall(answer)
    cited_normalized = [f"Clause {c.upper()}" for c in cited_clauses]

    if not cited_normalized:
        return {
            "has_citation": False,
            "cited_clauses": [],
            "hallucinated_citations": [],
            "hallucination_rate": 0.0,
            "is_grounded": True
        }

    hallucinated = []
    for citation in cited_normalized:
        if citation not in valid_clauses:
            hallucinated.append(citation)

    hallucination_rate = len(hallucinated) / len(cited_normalized) if cited_normalized else 0.0

    return {
        "has_citation": True,
        "cited_clauses": cited_normalized,
        "hallucinated_citations": hallucinated,
        "hallucination_rate": round(hallucination_rate, 4),
        "is_grounded": len(hallucinated) == 0
    }

def compute_drift_metric(pre_answers: List[str], post_answers: List[str]) -> float:
    """
    Computes the fraction of changed conclusions between pre-refresh and post-refresh golden replay sets.
    """
    if not pre_answers or not post_answers:
        return 0.0
    
    total = min(len(pre_answers), len(post_answers))
    if total == 0:
        return 0.0

    changed = 0
    for a1, a2 in zip(pre_answers[:total], post_answers[:total]):
        # Extract main numeric / clause tokens or normalized text
        norm1 = " ".join(a1.lower().split())
        norm2 = " ".join(a2.lower().split())
        if norm1 != norm2:
            changed += 1

    return round(changed / total, 4)
