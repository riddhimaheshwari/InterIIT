import copy
import numpy as np
from typing import Dict, List, Any, Tuple
from src.offline.validate import BenchmarkValidationGate

class DPOPolisher:
    """
    Direct Preference Optimization (DPO) polisher on grounded vs hallucinated pairs.
    Config-gated post-hoc alignment pass.
    """
    def __init__(
        self,
        revalidation_mode: str = "option_a",
        bwt_threshold: float = 0.03
    ):
        self.revalidation_mode = revalidation_mode
        self.validator = BenchmarkValidationGate(bwt_degradation_threshold=bwt_threshold)

    def generate_contrastive_pairs(
        self,
        held_out_slice: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Creates preference pairs: chosen = grounded answer, rejected = hallucinated/ungrounded answer.
        """
        pairs = []
        for item in held_out_slice:
            chosen = item.get("reference_answer", "")
            # Rejected introduces an ungrounded clause or wrong deadline
            rejected = chosen.replace("15 seconds", "45 minutes").replace("72 hours", "30 days")
            pairs.append({
                "prompt": item.get("query", ""),
                "chosen": chosen,
                "rejected": rejected
            })
        return pairs

    def apply_dpo_polish(
        self,
        candidate_adapter: Dict[str, Any],
        seen_regimes: List[str],
        historical_baselines: Dict[str, float],
        held_out_slice: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        """
        Runs DPO polishing and executes Option (a) re-validation.
        If post-DPO adapter degrades BWT, falls back to pre-DPO candidate.
        """
        # Create a modified copy of adapter matrices
        post_dpo_adapter = copy.deepcopy(candidate_adapter)
        post_dpo_adapter["adapter_id"] = f"{candidate_adapter['adapter_id']}_dpo_polished"
        
        # Simulate slight optimization adjustment
        A_matrix = post_dpo_adapter.get("A_matrix")
        if A_matrix is not None:
            post_dpo_adapter["A_matrix"] = A_matrix + 0.001 * np.random.randn(*A_matrix.shape)

        if self.revalidation_mode == "option_a":
            # Re-run Step D benchmark gate on post-DPO adapter
            passed, gate_summary = self.validator.validate_candidate_adapter(
                post_dpo_adapter,
                seen_regimes,
                historical_baselines
            )
            if passed:
                return post_dpo_adapter, True, gate_summary
            else:
                # Reject DPO pass and safely fall back to pre-DPO adapter
                gate_summary["dpo_fallback"] = True
                gate_summary["reason"] = "Post-DPO adapter degraded BWT beyond threshold; rolled back to pre-DPO."
                return candidate_adapter, False, gate_summary

        # Default fallback
        return candidate_adapter, True, {"verdict": "ACCEPTED_NO_GATE"}
