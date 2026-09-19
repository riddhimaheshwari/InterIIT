import pytest
from src.offline.replay_gen import ReplayGenerator

def test_self_consistency_filter_agreement():
    gen = ReplayGenerator()
    ans1 = "Under Clause AI-101, high-risk systems require risk mitigation within 15 seconds."
    ans2 = "Pursuant to Clause AI-101, high risk classification applies and mandates 15 seconds oversight."

    passed, final_ans = gen.filter_self_consistency([ans1, ans2])
    assert passed is True
    assert final_ans == ans1

def test_self_consistency_filter_disagreement():
    gen = ReplayGenerator()
    ans1 = "Under Clause AI-101, the response window is 15 seconds."
    ans2 = "Under Clause AUD-201, third party audits take 12 months."

    passed, final_ans = gen.filter_self_consistency([ans1, ans2])
    assert passed is False

def test_retrieval_grounding_filter():
    gen = ReplayGenerator()
    # Mock retriever chunks
    gen.retriever.chunks = [
        {
            "chunk_id": "REG-Q1-DOC-01",
            "clause_id": "Clause AI-101",
            "regime_id": "Q1",
            "title": "High-Risk AI",
            "text": "High risk AI systems classified under AI-101.",
            "effective_date": "2025-01-01"
        }
    ]
    gen.retriever.embeddings = gen.retriever._compute_chunk_embedding("Clause AI-101 High-Risk AI")[None, :]

    # Grounded answer citing existing clause
    grounded = gen.filter_retrieval_grounding(
        query="What is high risk under AI-101?",
        answer="Under Clause AI-101, systems are high risk.",
        regime_id="Q1"
    )
    assert grounded is True

    # Hallucinated answer citing non-existent clause
    hallucinated = gen.filter_retrieval_grounding(
        query="What is high risk?",
        answer="Under Clause FAKE-999, all models are forbidden.",
        regime_id="Q1"
    )
    assert hallucinated is False
