import sqlite3
import pytest
from src.audit.log import AuditLogger

def test_audit_log_hash_chain_creation_and_integrity(tmp_path):
    db_file = str(tmp_path / "test_audit.db")
    logger = AuditLogger(db_path=db_file)

    rec1 = logger.log_query(
        query_id="Q-001",
        query_text="What is Clause AI-101?",
        retrieved_chunk_ids=["REG-Q1-DOC-01"],
        regime_tags=["Q1"],
        retrieval_index_hash="abc123hash",
        adapter_hash="adapterhash1",
        confidence_scores=[0.95],
        verification_verdict="VERIFIED",
        final_answer="High risk classification."
    )

    rec2 = logger.log_query(
        query_id="Q-002",
        query_text="What is Clause AI-102?",
        retrieved_chunk_ids=["REG-Q1-DOC-02"],
        regime_tags=["Q1"],
        retrieval_index_hash="abc123hash",
        adapter_hash="adapterhash1",
        confidence_scores=[0.88],
        verification_verdict="VERIFIED",
        final_answer="15 second human oversight."
    )

    # Verify previous_hash linking
    assert rec2["previous_hash"] == rec1["record_hash"]

    # Verify ledger integrity
    is_valid, err = logger.verify_integrity()
    assert is_valid is True
    assert err is None

def test_audit_log_tamper_detection(tmp_path):
    db_file = str(tmp_path / "test_audit_tamper.db")
    logger = AuditLogger(db_path=db_file)

    logger.log_query(
        query_id="Q-001",
        query_text="Original query",
        retrieved_chunk_ids=["C1"],
        regime_tags=["Q1"],
        retrieval_index_hash="hash1",
        adapter_hash="ahash1",
        confidence_scores=[0.9],
        verification_verdict="VERIFIED",
        final_answer="Original Answer"
    )

    logger.log_query(
        query_id="Q-002",
        query_text="Second query",
        retrieved_chunk_ids=["C2"],
        regime_tags=["Q1"],
        retrieval_index_hash="hash1",
        adapter_hash="ahash1",
        confidence_scores=[0.9],
        verification_verdict="VERIFIED",
        final_answer="Second Answer"
    )

    # Tamper with the first record in SQLite directly
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("UPDATE audit_log SET final_answer = 'Tampered Answer' WHERE query_id = 'Q-001'")
    conn.commit()
    conn.close()

    # Verify ledger catches the tampering
    is_valid, err = logger.verify_integrity()
    assert is_valid is False
    assert "Tampered record" in err
