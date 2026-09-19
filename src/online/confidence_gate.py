from typing import List, Dict, Any, Tuple

FALLBACK_INSUFFICIENT_GROUNDING = "insufficient grounding — escalate to human review"

class RetrievalConfidenceGate:
    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold

    def evaluate(self, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[bool, str, List[float]]:
        """
        Evaluates top-k retrieval similarity scores.
        If scores fall below the configured threshold, blocks generation immediately.
        """
        if not retrieved_chunks:
            return False, FALLBACK_INSUFFICIENT_GROUNDING, []

        scores = [chunk.get("similarity_score", 0.0) for chunk in retrieved_chunks]
        max_score = max(scores) if scores else 0.0

        if max_score < self.threshold:
            return False, FALLBACK_INSUFFICIENT_GROUNDING, scores

        return True, "PASSED", scores
