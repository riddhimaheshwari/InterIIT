import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from src.offline.validate import BenchmarkValidationGate
from src.offline.replay_gen import ReplayGenerator

class TIESConsolidator:
    """
    TIES-Merge Consolidation for LoRA Adapter Stacks.
    1. Trim: Keep top k% magnitude parameters.
    2. Elect: Compute majority sign (+1, -1, 0) per parameter.
    3. Merge: Disjoint mean of values agreeing with majority sign.
    """
    def __init__(
        self,
        trim_threshold: float = 0.20,
        adapters_dir: str = "artifacts/adapters",
        bwt_threshold: float = 0.03
    ):
        self.trim_threshold = trim_threshold # Fraction of top magnitude parameters to keep (e.g. 0.20)
        self.adapters_dir = Path(adapters_dir)
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        self.validator = BenchmarkValidationGate(bwt_degradation_threshold=bwt_threshold)

    def ties_merge_matrices(
        self,
        matrices: List[np.ndarray]
    ) -> np.ndarray:
        if not matrices:
            raise ValueError("No matrices provided for TIES merge.")
        if len(matrices) == 1:
            return matrices[0]

        stack = np.stack(matrices, axis=0) # [num_adapters, dim1, dim2]
        num_adapters = stack.shape[0]

        # 1. Trim smallest (1 - k)% magnitude parameters per adapter
        trimmed_stack = np.zeros_like(stack)
        for i in range(num_adapters):
            mat = stack[i]
            abs_mat = np.abs(mat)
            cutoff = np.percentile(abs_mat, (1.0 - self.trim_threshold) * 100.0)
            mask = abs_mat >= cutoff
            trimmed_stack[i] = np.where(mask, mat, 0.0)

        # 2. Elect majority sign
        signs = np.sign(trimmed_stack) # values in {-1, 0, 1}
        sign_sums = np.sum(signs, axis=0)
        majority_sign = np.sign(sign_sums) # +1 or -1, 0 if tied

        # 3. Disjoint Mean: Average only values agreeing with majority sign
        merged = np.zeros_like(stack[0])
        counts = np.zeros_like(stack[0])

        for i in range(num_adapters):
            agree_mask = (np.sign(trimmed_stack[i]) == majority_sign) & (majority_sign != 0)
            merged += np.where(agree_mask, trimmed_stack[i], 0.0)
            counts += agree_mask.astype(np.float32)

        # Avoid division by zero
        counts_safe = np.where(counts > 0, counts, 1.0)
        final_merged = np.where(counts > 0, merged / counts_safe, 0.0)

        return final_merged

    def consolidate_adapters(
        self,
        adapter_stack: List[Dict[str, Any]],
        merged_regime_label: str,
        seen_regimes: List[str],
        historical_baselines: Dict[str, float],
        replay_generator: Optional[ReplayGenerator] = None,
        golden_refresh_enabled: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Step E: Consolidates accumulated adapter stack into one merged adapter via TIES.
        Validates performance; on success resets basis and triggers golden refresh if enabled.
        """
        if len(adapter_stack) < 2:
            return False, {"verdict": "SKIPPED_INSUFFICIENT_STACK", "reason": "Fewer than 2 adapters to merge"}

        merged_id = f"adapter_ties_merged_{merged_regime_label}"
        A_list = [a["A_matrix"] for a in adapter_stack]
        B_list = [a["B_matrix"] for a in adapter_stack]

        merged_A = self.ties_merge_matrices(A_list)
        merged_B = self.ties_merge_matrices(B_list)

        candidate_merged = {
            "adapter_id": merged_id,
            "regime_id": merged_regime_label,
            "A_matrix": merged_A,
            "B_matrix": merged_B,
            "is_merged": True,
            "merged_parents": [a["adapter_id"] for a in adapter_stack]
        }

        # Validate merged adapter
        passed, val_summary = self.validator.validate_candidate_adapter(
            candidate_merged,
            seen_regimes,
            historical_baselines
        )

        if not passed:
            val_summary["verdict"] = "REJECTED_MERGE_UNDERPERFORMED"
            return False, val_summary

        # Save merged adapter artifacts
        save_path = self.adapters_dir / merged_id
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path / "lora_A.npy", merged_A)
        np.save(save_path / "lora_B.npy", merged_B)

        meta = {
            "adapter_id": merged_id,
            "regime_id": merged_regime_label,
            "is_merged": True,
            "merged_parents": [a["adapter_id"] for a in adapter_stack],
            "trim_threshold": self.trim_threshold,
            "validation_metrics": val_summary
        }
        with open(save_path / "adapter_config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Trigger golden refresh cycle if enabled
        drift_logs = {}
        if golden_refresh_enabled and replay_generator is not None:
            drift_logs = replay_generator.refresh_golden_sets_post_merge(
                merged_adapter_id=merged_id,
                seen_regimes=seen_regimes
            )

        val_summary["verdict"] = "ACCEPTED_MERGE"
        val_summary["merged_adapter"] = candidate_merged
        val_summary["drift_logs"] = drift_logs
        val_summary["adapter_path"] = str(save_path)

        return True, val_summary
