import pytest
from src.online.confidence_gate import RetrievalConfidenceGate, FALLBACK_INSUFFICIENT_GROUNDING

def test_confidence_gate_passed():
    gate = RetrievalConfidenceGate(threshold=0.65)
    chunks = [
        {"chunk_id": "C1", "similarity_score": 0.82},
        {"chunk_id": "C2", "similarity_score": 0.55}
    ]
    passed, msg, scores = gate.evaluate(chunks)
    assert passed is True
    assert msg == "PASSED"
    assert scores == [0.82, 0.55]

def test_confidence_gate_blocks_low_similarity():
    gate = RetrievalConfidenceGate(threshold=0.65)
    chunks = [
        {"chunk_id": "C1", "similarity_score": 0.42},
        {"chunk_id": "C2", "similarity_score": 0.38}
    ]
    passed, msg, scores = gate.evaluate(chunks)
    assert passed is False
    assert msg == FALLBACK_INSUFFICIENT_GROUNDING
    assert scores == [0.42, 0.38]

def test_confidence_gate_empty_chunks():
    gate = RetrievalConfidenceGate(threshold=0.65)
    passed, msg, scores = gate.evaluate([])
    assert passed is False
    assert msg == FALLBACK_INSUFFICIENT_GROUNDING
    assert scores == []
