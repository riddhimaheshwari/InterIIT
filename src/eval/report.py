import json
import os
from pathlib import Path
from typing import Dict, List, Any

class EvalReportGenerator:
    def __init__(self, report_dir: str = "artifacts/reports"):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.report_dir / "progression_curves.json"

    def load_history(self) -> Dict[str, Any]:
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "quarters": [],
            "avg_accuracy": [],
            "bwt": [],
            "fwt": [],
            "adapter_footprint_mb": [],
            "basis_rank": [],
            "hallucination_rate": [],
            "confusion_score": [],
            "drift_diff": [],
            "regime_accuracy_matrix": {}
        }

    def record_quarter_update(
        self,
        regime: str,
        avg_acc: float,
        bwt: float,
        fwt: float,
        adapter_footprint_mb: float,
        basis_rank: int,
        hallucination_rate: float,
        confusion_score: float,
        drift_diff: float,
        accuracy_row: Dict[str, float]
    ):
        history = self.load_history()
        
        # Check if already present, update or append
        if regime in history["quarters"]:
            idx = history["quarters"].index(regime)
            history["avg_accuracy"][idx] = avg_acc
            history["bwt"][idx] = bwt
            history["fwt"][idx] = fwt
            history["adapter_footprint_mb"][idx] = adapter_footprint_mb
            history["basis_rank"][idx] = basis_rank
            history["hallucination_rate"][idx] = hallucination_rate
            history["confusion_score"][idx] = confusion_score
            history["drift_diff"][idx] = drift_diff
        else:
            history["quarters"].append(regime)
            history["avg_accuracy"].append(avg_acc)
            history["bwt"].append(bwt)
            history["fwt"].append(fwt)
            history["adapter_footprint_mb"].append(adapter_footprint_mb)
            history["basis_rank"].append(basis_rank)
            history["hallucination_rate"].append(hallucination_rate)
            history["confusion_score"].append(confusion_score)
            history["drift_diff"].append(drift_diff)

        history["regime_accuracy_matrix"][regime] = accuracy_row

        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        return history

    def get_summary_report(self) -> Dict[str, Any]:
        return self.load_history()
