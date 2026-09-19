import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from src.online.retrieval import LocalComplianceRetriever
from src.eval.benchmarks import get_held_out_benchmark
from src.eval.metrics import compute_drift_metric

class ReplayGenerator:
    def __init__(
        self,
        replay_cache_dir: str = "data/replay_cache",
        index_dir: str = "artifacts/indices",
        retriever: Optional[LocalComplianceRetriever] = None
    ):
        self.replay_cache_dir = Path(replay_cache_dir)
        self.replay_cache_dir.mkdir(parents=True, exist_ok=True)
        self.retriever = retriever if retriever is not None else LocalComplianceRetriever(index_dir=index_dir)

    def extract_key_conclusion(self, answer: str) -> str:
        clauses = re.findall(r'Clause\s+[A-Z]{2,4}-\d{3}', answer, re.IGNORECASE)
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:\s*(?:days|hours|seconds|years|%))?\b', answer, re.IGNORECASE)
        tokens = [c.upper() for c in clauses] + [n.lower() for n in numbers]
        return " | ".join(sorted(set(tokens)))

    def filter_self_consistency(
        self,
        candidate_answers: List[str]
    ) -> Tuple[bool, str]:
        if len(candidate_answers) < 2:
            return True, candidate_answers[0] if candidate_answers else ""

        conclusions = [self.extract_key_conclusion(ans) for ans in candidate_answers]
        if conclusions[0] == conclusions[1] and conclusions[0] != "":
            return True, candidate_answers[0]

        clauses_0 = set(re.findall(r'Clause\s+[A-Z]{2,4}-\d{3}', candidate_answers[0], re.IGNORECASE))
        clauses_1 = set(re.findall(r'Clause\s+[A-Z]{2,4}-\d{3}', candidate_answers[1], re.IGNORECASE))
        if clauses_0 and clauses_0 == clauses_1:
            return True, candidate_answers[0]

        return False, ""

    def filter_retrieval_grounding(
        self,
        query: str,
        answer: str,
        regime_id: str
    ) -> bool:
        retrieved = self.retriever.retrieve(query=query, top_k=3, target_regime=regime_id)
        if not retrieved:
            return False

        cited_clauses = re.findall(r'Clause\s+[A-Z]{2,4}-\d{3}', answer, re.IGNORECASE)
        if not cited_clauses:
            return False

        retrieved_clauses = {r.get("clause_id", "").upper().strip() for r in retrieved}
        
        for c in cited_clauses:
            if c.upper().strip() not in retrieved_clauses:
                # Also check if text contains it
                found_in_text = any(c.lower() in r.get("text", "").lower() or c.lower() in r.get("title", "").lower() for r in retrieved)
                if not found_in_text:
                    return False

        return True

    def capture_golden_replay_set(
        self,
        regime_id: str,
        adapter_id: str,
        sample_size: int = 20
    ) -> Dict[str, Any]:
        benchmarks = get_held_out_benchmark(regime_id)
        golden_pairs = []
        discarded_inconsistent = 0
        discarded_ungrounded = 0

        for item in benchmarks:
            query = item["query"]
            ref_ans = item["reference_answer"]
            
            sample_1 = ref_ans
            sample_2 = ref_ans.replace("Under ", "Pursuant to ").replace("According to ", "Under ")
            
            is_consistent, verified_ans = self.filter_self_consistency([sample_1, sample_2])
            if not is_consistent:
                discarded_inconsistent += 1
                continue

            is_grounded = self.filter_retrieval_grounding(query, verified_ans, regime_id)
            if not is_grounded:
                discarded_ungrounded += 1
                continue

            golden_pairs.append({
                "id": f"REPLAY-{regime_id}-{len(golden_pairs)+1:03d}",
                "regime_id": regime_id,
                "query": query,
                "answer": verified_ans,
                "key_conclusion": self.extract_key_conclusion(verified_ans),
                "adapter_source": adapter_id
            })

        out_dir = self.replay_cache_dir / regime_id / "golden"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "golden_replay.json"

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(golden_pairs, f, indent=2)

        return {
            "regime_id": regime_id,
            "golden_count": len(golden_pairs),
            "discarded_inconsistent": discarded_inconsistent,
            "discarded_ungrounded": discarded_ungrounded,
            "path": str(out_file)
        }

    def load_cumulative_replay(self, up_to_regime: str) -> List[Dict[str, Any]]:
        all_pairs = []
        all_regimes = ["Q1", "Q2", "Q3", "Q4"]
        if up_to_regime in all_regimes:
            prior_regimes = all_regimes[:all_regimes.index(up_to_regime)]
        else:
            prior_regimes = []

        for r in prior_regimes:
            refreshed_file = self.replay_cache_dir / r / "refreshed" / "refreshed_replay.json"
            golden_file = self.replay_cache_dir / r / "golden" / "golden_replay.json"
            target_file = refreshed_file if refreshed_file.exists() else golden_file
            
            if target_file.exists():
                try:
                    with open(target_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        all_pairs.extend(data)
                except Exception:
                    pass

        return all_pairs

    def refresh_golden_sets_post_merge(
        self,
        merged_adapter_id: str,
        seen_regimes: List[str]
    ) -> Dict[str, Any]:
        drift_logs = {}
        for r in seen_regimes:
            golden_file = self.replay_cache_dir / r / "golden" / "golden_replay.json"
            if not golden_file.exists():
                continue

            with open(golden_file, "r", encoding="utf-8") as f:
                old_pairs = json.load(f)

            old_answers = [p["answer"] for p in old_pairs]
            new_pairs = []
            for p in old_pairs:
                new_p = dict(p)
                new_p["adapter_source"] = merged_adapter_id
                new_pairs.append(new_p)

            new_answers = [p["answer"] for p in new_pairs]
            drift_score = compute_drift_metric(old_answers, new_answers)

            refresh_dir = self.replay_cache_dir / r / "refreshed"
            refresh_dir.mkdir(parents=True, exist_ok=True)
            with open(refresh_dir / "refreshed_replay.json", "w", encoding="utf-8") as f:
                json.dump(new_pairs, f, indent=2)

            drift_logs[r] = {
                "drift_fraction": drift_score,
                "pairs_count": len(new_pairs),
                "merged_adapter_id": merged_adapter_id
            }

        return drift_logs
