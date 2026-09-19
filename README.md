# Continual-Counsel: Edge-Deployable Legal Compliance Assistant with Orthogonal LoRA & Verifiable Auditability

[![CI Test & Offline Isolation Gate](https://github.com/continual-counsel/continual-counsel/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Continual-Counsel** is an edge-deployable legal and regulatory compliance assistant designed to ingest quarterly regulatory updates without full retraining, without catastrophic forgetting of earlier regimes, and with strict verifiable auditability.

---

## 1. System Architecture: Strict Physical Loop Separation

The codebase is physically partitioned into two non-overlapping loops to satisfy sovereign compliance and edge hardware deployment constraints:

```
continual-counsel/
  configs/
    base.yaml              # Base model fallback cascade, data paths, FAISS index directory
    training.yaml          # LoRA rank/alpha, lambda_ortho, replay ratio, merge cadence, gates
  data/
    regimes/               # Quarterly regulatory texts (Q1..Q4: AI Act, Transparency, Liability, Sovereignty)
    replay_cache/          # Golden + refreshed synthetic replay sets per regime
  src/
    offline/               # [TRAINING MACHINE ONLY]
      replay_gen.py        # Step A: Golden replay capture + self-consistency & grounding filters
      train_adapter.py     # Step C: O-LoRA training loop with incremental QR orthonormal basis
      validate.py          # Step D: Automated BWT degradation gate (< 3% threshold)
      merge.py             # Step E: TIES-merge parameter consolidation (Trim, Elect, Average)
      distill.py           # Feature-level hidden-state matching distillation loss
      dpo_polish.py        # Optional DPO alignment pass with Option (a) re-validation gate
      pipeline.py          # Orchestrates A -> B -> C -> D -> E for quarterly update cycles
    online/                # [EDGE INFERENCE ONLY - ZERO NETWORK CALLS]
      retrieval.py         # Local FAISS index + temporal metadata filter + SHA-256 index hashing
      confidence_gate.py   # Similarity gate blocking generation on ungrounded queries (< 0.65)
      verify.py            # Self-verification pass cross-checking cited clauses against retrieved context
      infer.py             # Glues frozen base + merged adapter + retrieval + verify + audit log
      api.py               # FastAPI edge entrypoint (/query, /audit/{query_id}, /health)
    audit/
      registry.py          # SQLite adapter registry (hyperparameters, training data hash, merge lineage)
      log.py               # Tamper-evident cryptographic hash-chain audit ledger
    eval/
      benchmarks.py        # Per-quarter held-out benchmarks (statutory QA + multi-clause reasoning)
      confusion_set.py     # Cross-regime adversarial eval set (reconciling conflicting/superseding rules)
      metrics.py           # Backward Transfer (BWT), Forward Transfer (FWT), citation hallucinations
      report.py            # Generates progression curves for the dashboard
  dashboard/
    app.py                 # Streamlit dashboard (BWT/FWT curves, basis rank sawtooth, audit explorer)
  scripts/
    run_quarter_update.sh  # CLI runner for quarterly pipeline updates
    run_full_eval.sh       # Runs full Q1-Q4 progression and tests
    export_gguf.sh         # Quantizes consolidated adapter to 4-bit GGUF for edge deployment
    smoke_test_no_network.sh # AST inspector and socket blocker proving zero network egress
  .github/workflows/ci.yml # Automated CI pipeline running pytest and network isolation gate
  Dockerfile               # Edge inference container with network isolation
  tests/                   # Pytest suite
```

> **Zero-Network Invariant**: `src/online/` contains **0 HTTP client imports** (`requests`, `httpx`, `aiohttp`, `urllib.request`). The embedding model, vector index, and model weights are bundled locally. Enforced via AST inspection in `tests/test_no_network_isolation.py` and `scripts/smoke_test_no_network.sh`.

---

## 2. Mathematical Design & 5-Step Offline Pipeline

Every quarterly regulatory tranche progresses through 5 sequential steps:

### Step A — Golden Replay Capture
Runs at the **end of regime $t$'s own update cycle** using the just-accepted adapter. Generates synthetic QA pairs from held-out templates, filtered through:
1. **Self-Consistency Filter**: Multi-sample evaluation under temperature perturbation; keeps pairs where generations agree on key statutory conclusions.
2. **Retrieval-Grounding Filter**: Validates answer claims against indexed source clauses in FAISS; discards ungrounded pairs.
- **Golden Refresh (Config-Gated)**: On Step E merge, regenerates a consolidated golden set for the merged state, calculates drift diff fraction, and logs drift without deleting historical pre-refresh golden sets.

### Step B — Compose Training Batch
$$\mathcal{D}_{\text{train}} = \mathcal{D}_{\text{new}} \cup \mathcal{D}_{\text{replay}}$$
Mixed at 70:30 by default (sweepable 10–40%).

### Step C — O-LoRA Training with Incremental QR Basis
Trains low-rank LoRA matrices $A_t, B_t$ with the composite loss:
$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda || A_t^T Q ||_F^2 + \lambda_{\text{distill}} \mathcal{L}_{\text{distill}}$$
- $Q \in \mathbb{R}^{d \times k}$ is an orthonormal projection basis maintained via incremental QR decomposition across historical adapters.
- **Sawtooth Basis Reset**: When Step E merges the adapter stack, $Q$ resets and rebuilds from $Q = \text{qr}(A_{\text{merged}})[0]$, preventing the orthogonal complement from shrinking over time.

### Step D — Validation Gate
Evaluates candidate adapter against all seen regimes $1..t$. Computes Backward Transfer (BWT):
$$\text{BWT} = \frac{1}{t-1}\sum_{i=1}^{t-1} (\text{Acc}_{t,i} - \text{Acc}_{i,i})$$
If degradation on any prior quarter exceeds **3% absolute (0.03)**, the update is rejected.

### Step E — Consolidation (TIES-Merge)
Every $N=2$ quarters, consolidates the accumulated adapter stack via TIES:
1. **Trim**: Retain top 20% magnitude parameters per adapter.
2. **Elect**: Determine majority sign across adapters per parameter.
3. **Disjoint Average**: Average values agreeing with the majority sign.
On successful validation of the merged adapter, resets the orthogonal basis and triggers golden refresh.

---

## 3. DPO Re-Validation Mode: Option (a)

**Decision**: We implement **Option (a)**: Re-running the Step D benchmark gate on the post-DPO adapter.

**Rationale**:
Direct Preference Optimization updates adapter weights $A_t, B_t$ directly based on contrastive preference pairs (grounded vs. hallucinated). Because DPO does not natively incorporate the QR projection penalty, it can pull adapter weights out of the orthogonal subspace validated in Step D. Under Option (a), if the post-DPO adapter degrades prior-regime BWT beyond the 3% threshold, the pipeline rejects the DPO pass and automatically falls back to the pre-DPO validated adapter, preserving catastrophic forgetting guarantees.

---

## 4. Tamper-Evident Cryptographic Audit Ledger

Every inference query logs an immutable block chained cryptographically:
$$H_n = \text{SHA256}(H_{n-1} \parallel \text{canonical\_payload})$$

Logged fields: `query_id`, `timestamp`, `query_text`, `retrieved_chunk_ids`, `regime_tags`, `retrieval_index_hash`, `adapter_hash`, `confidence_scores`, `verification_verdict`, `final_answer`, `previous_hash`, `record_hash`.

Endpoint `GET /audit/{query_id}` returns the complete answer provenance and merges lineage back to the original per-quarter training data.

---

## 5. End-to-End Quickstart (Reproduce in 2 Minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Quarterly Updates (Q1 → Q4)
```bash
python -m src.offline.pipeline --regime Q1
python -m src.offline.pipeline --regime Q2
python -m src.offline.pipeline --regime Q3
python -m src.offline.pipeline --regime Q4
```

### Step 3: Run the Edge Inference API
```bash
python -m uvicorn src.online.api:app --host 0.0.0.0 --port 8000
```
Test with curl or Python:
```bash
# Valid grounded query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the reporting timeframe for critical AI malfunctions under INC-301?"}'

# Ungrounded query (triggers confidence gate block)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do you make chocolate cake in Paris?"}'
```

### Step 4: Open the Streamlit Dashboard
```bash
python -m streamlit run dashboard/app.py
```
Explore:
- Continual learning curves (BWT, FWT, Average Accuracy across Q1-Q4)
- Orthogonal basis rank sawtooth curve & bounded adapter footprint
- Live edge query console with real-time provenance tracing
- Cryptographic audit ledger verification
- Cross-regime adversarial confusion benchmark

---

## 6. Running Tests & Zero-Network Verification

Run the full pytest suite:
```bash
python -m pytest tests/ -v
```

Run the zero-network offline smoke test gate:
```bash
python -c "
from src.online.infer import ComplianceInferenceEngine
import socket
socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(ConnectionRefusedError('No network allowed!'))
engine = ComplianceInferenceEngine()
res = engine.process_query('What is the human oversight intervention threshold under Clause AI-102?')
assert res['status'] == 'success'
print('Verified: Zero network calls during edge inference.')
"
```
