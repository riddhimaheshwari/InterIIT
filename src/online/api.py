from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from src.online.infer import ComplianceInferenceEngine
from src.audit.registry import AdapterRegistry
from src.audit.log import AuditLogger

app = FastAPI(
    title="Continual-Counsel Edge Inference API",
    description="Edge-deployable compliance assistant with verifiable audit logging and zero network egress.",
    version="1.0.0"
)

# Initialize engine and storage
engine = ComplianceInferenceEngine()
registry = AdapterRegistry()
audit_logger = AuditLogger()

class QueryRequest(BaseModel):
    query: str = Field(..., description="Compliance query or statutory question")
    target_regime: Optional[str] = Field(None, description="Optional target regime (e.g. Q1, Q2, Q3, Q4)")

class QueryResponse(BaseModel):
    query_id: str
    status: str
    answer: str
    confidence_passed: bool
    confidence_scores: List[float]
    verification_verdict: str
    adapter_id: Optional[str] = None
    adapter_hash: Optional[str] = None
    retrieval_index_hash: Optional[str] = None
    audit_record: Dict[str, Any]

@app.get("/health")
def health_check():
    latest_adapter = registry.get_latest_adapter()
    return {
        "status": "healthy",
        "mode": "edge_offline",
        "active_adapter_id": latest_adapter["adapter_id"] if latest_adapter else "genesis",
        "active_adapter_hash": latest_adapter["adapter_hash"] if latest_adapter else "0" * 64,
        "retrieval_index_hash": engine.retriever.get_index_hash(),
        "network_egress": "disabled"
    }

@app.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    result = engine.process_query(
        query=request.query,
        target_regime=request.target_regime
    )
    return result

@app.get("/audit/{query_id}")
def get_audit_record(query_id: str):
    record = audit_logger.get_record(query_id)
    if not record:
        raise HTTPException(status_code=404, detail="Audit record not found for the specified query_id")

    # Fetch corresponding adapter lineage
    adapter_info = None
    if record.get("adapter_hash"):
        all_adapters = registry.list_all_adapters()
        for a in all_adapters:
            if a["adapter_hash"] == record["adapter_hash"]:
                adapter_info = {
                    "adapter_id": a["adapter_id"],
                    "regime_id": a["regime_id"],
                    "merge_lineage": a["merge_lineage"],
                    "is_merged": a["is_merged"],
                    "validation_metrics": a["validation_metrics"],
                    "created_at": a["created_at"]
                }
                break

    return {
        "audit_record": record,
        "adapter_lineage": adapter_info
    }

@app.get("/audit/ledger/verify")
def verify_audit_ledger():
    is_valid, error_msg = audit_logger.verify_integrity()
    return {
        "integrity_verified": is_valid,
        "error": error_msg,
        "status": "TAMPER_FREE" if is_valid else "CORRUPTED_OR_TAMPERED"
    }

@app.get("/adapters")
def list_adapters():
    return {
        "adapters": registry.list_all_adapters()
    }
