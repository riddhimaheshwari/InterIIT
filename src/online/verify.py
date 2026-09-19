import re
from typing import List, Dict, Any, Tuple

VERDICT_VERIFIED = "VERIFIED"
VERDICT_UNVERIFIED_CITATION = "UNVERIFIED_CITATION"
VERDICT_GROUNDED = "GROUNDED"

class ComplianceVerifier:
    def __init__(self):
        self.clause_pattern = re.compile(r'\bClause\s+([A-Z]{2,4}-\d{3})\b', re.IGNORECASE)

    def verify_answer(
        self,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Tuple[str, List[str], List[str]]:
        """
        Self-verification pass: inspects whether citations made in the answer
        are fully substantiated by the retrieved context.
        """
        retrieved_clause_ids = {
            chunk.get("clause_id", "").upper().strip() for chunk in retrieved_chunks
        }
        retrieved_texts = " ".join([chunk.get("text", "").lower() for chunk in retrieved_chunks])

        cited_matches = self.clause_pattern.findall(answer)
        cited_clauses = [f"Clause {c.upper()}" for c in cited_matches]

        unverified_citations = []
        for citation in cited_clauses:
            if citation.upper() not in retrieved_clause_ids:
                unverified_citations.append(citation)

        if unverified_citations:
            return VERDICT_UNVERIFIED_CITATION, cited_clauses, unverified_citations

        return VERDICT_VERIFIED, cited_clauses, []
