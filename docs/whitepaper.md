# Continual-Counsel (The Magistrate): Continuous Regulatory Adaptation for Air-Gapped Compliance Copilots

**Technical Whitepaper - Inter IIT Tech Meet 2025 Submission**
*Track: The Amnesiac Magistrate (Problem Statement 2)*

---

## Abstract
Edge-deployed compliance assistants for air-gapped sovereign wealth funds and tier-1 banking institutions face a fundamental trilemma: (1) absorbing sequential quarterly statutory overhauls without catastrophic forgetting of foundational contract law, (2) operating under strict edge hardware memory constraints without unbounded adapter proliferation, and (3) complying with stringent non-disclosure agreements that strictly prohibit retaining historical confidential text for experience replay.

We present The Magistrate, a dual-loop continual adaptation architecture that resolves this trilemma through four synergistic innovations:
1. O-LoRA Subspace Orthogonality: Parameter updates are constrained to the orthogonal complement of historical parameter subspaces via incremental QR decomposition, physically eliminating destructive interference.
2. Sawtooth TIES-Merge Consolidation: Periodic parameter consolidation via Trim, Elect Sign, and Disjoint Average bounds edge adapter memory footprint to ~19MB and resets orthogonal capacity.
3. Privacy-Preserving Synthetic Golden Replay: Experience replay without raw text retention, verified by dual-seed self-consistency and retrieval-grounding filters.
4. Air-Gapped Gated Edge Verification: Zero-network local FAISS retrieval, confidence-gating thresholding, and SHA-256 tamper-evident hash-chained audit logging.

Empirical evaluation across a 4-quarter regulatory trajectory (Q1 High-Risk AI Act, Q2 Algorithmic Transparency, Q3 Incident Liability, Q4 Sovereign Compute) demonstrates Backward Transfer (BWT) of -0.0125, average accuracy of 94.8%, citation hallucination rates under 2.5%, and zero network leakage verified by AST inspection and socket blocking.

---

## 1. Problem Formulation & System Constraints

### 1.1 The Operational Scenario
Lexis Sovereign operates an 8-billion parameter local model deployed in air-gapped financial vaults. Every quarter t in {1, 2, ..., T}, a new regulatory tranche D_t is promulgated. The primary operational hazards are:
- Catastrophic Amnesia: Adapting to tranche D_t overwrites weights essential for interpreting tranche D_1, causing statutory deduction errors.
- Precedent Hallucination: Generating fictitious clauses or misattributing liabilities across regimes.
- Hardware & Privacy Boundaries:
  - The inference engine must make ZERO outbound network calls.
  - Historical confidential client filings cannot be retained (|D_<t^raw| = 0).
  - Total adapter VRAM allocation must remain bounded (O(1) scaling, not O(T)).

---

## 2. Mathematical Architecture

### 2.1 Low-Rank Parameter Subspaces & Orthogonalization
Standard LoRA parametrizes the weight update as Delta W = B * A, where A in R^(r x d) and B in R^(d x r) with r << d. Under sequential fine-tuning on tasks 1, ..., t, subsequent adapter updates Delta W_t project onto non-orthogonal subspaces, causing catastrophic interference:
Delta W_t * Delta W_<t != 0

To guarantee interference-free parameter allocation, we maintain an orthonormal basis matrix Q_(t-1) in R^(d x k) spanning all previous adapter subspaces using incremental QR decomposition:
[Q_(t-1) | A_t^T] = Q_t * R_t

We introduce an Orthogonality Regularization Penalty to the training objective:
L_ortho = || A_t^T * Q_(t-1) ||_F^2 = Tr(A_t^T * Q_(t-1) * Q_(t-1)^T * A_t)

This penalty penalizes any projection of the new adapter A_t onto the historical subspace Q_(t-1), mathematically confining new statutory learning to the orthogonal null-space Null(Q_(t-1)^T).

### 2.2 Feature-Level Distillation on Replay Buffers
To stabilize intermediate representations without storing client documents, we apply feature-level distillation on replay tokens:
L_distill = (1 / N) * sum(|| h_s(x_i) - h_t(x_i) ||_2^2)
where h_s and h_t are the student and frozen teacher representations from the penultimate transformer layer.

### 2.3 Composite Optimization Objective
The total loss minimized during offline quarterly updates is:
L_total = L_CE + lambda_ortho * L_ortho + lambda_distill * L_distill
where lambda_ortho = 0.10 and lambda_distill = 0.05.

---

## 3. Parameter Consolidation: The Sawtooth TIES-Merge

### 3.1 The Orthogonal Capacity Dilemma
Because the ambient dimension d is finite, incrementally accumulating rank-r subspaces will eventually consume the available orthogonal complement as t -> infinity. Furthermore, storing individual adapters {A_1, ..., A_T} causes linear memory growth O(T * r * d) on edge hardware.

### 3.2 TIES Consolidation Algorithm
Every C=2 quarters, the accumulated adapter stack is consolidated into a single merged adapter W_merged via TIES-Merging:
1. Trim: For each adapter parameter matrix, retain only the top k = 20% parameters by absolute magnitude:
   tau_i = quantile_(1-k)(|W_i|),  W_hat_i = W_i * I(|W_i| >= tau_i)
2. Elect Sign: Compute the majority directional sign across the trimmed parameter stack:
   gamma = sign(sum(sign(W_hat_i)))
3. Disjoint Average: Average only the parameters that agree with the majority sign gamma:
   W_merged = sum(W_hat_i * I(sign(W_hat_i) == gamma)) / sum(I(sign(W_hat_i) == gamma))

### 3.3 The Sawtooth Basis Reset
Upon successful consolidation, the orthonormal basis Q is reset to the single merged adapter subspace:
Q <- orth(A_merged)
This resets the basis rank from M * r back to r, releasing (M-1) * r dimensions back to the orthogonal complement and producing a stable sawtooth basis curve.

---

## 4. Privacy-Preserving Golden Replay

To satisfy customer privacy agreements forbidding raw text retention:
1. Synthetic Generation: At the close of quarter t, the newly tuned model synthesizes candidate question-answer pairs spanning core clauses.
2. Self-Consistency Gate: Each candidate is sampled with two distinct random seeds. Candidates with divergent conclusions (>15% token disagreement) are rejected.
3. Retrieval Grounding Filter: Answers are checked against the local FAISS index. Any claim unsupported by a retrieved chunk is discarded.
4. Result: A compact golden cache of synthetic reasoning pairs that contains zero raw client disclosures.

---

## 5. Air-Gapped Edge Inference Loop

The inference engine runs in an isolated edge container with the following zero-trust pipeline:
1. Local FAISS Vector Retrieval: Exact inner-product cosine similarity over local MiniLM embeddings (d=384).
2. Confidence Gate: If top retrieved chunk similarity sigma < 0.65, inference halts immediately with an audit flag (FALLBACK_INSUFFICIENT_GROUNDING), preventing hallucinations on out-of-distribution queries.
3. Self-Verification Pass: Regex parser verifies all cited Clause citations against retrieved chunks before emission.
4. Cryptographic Audit Log: Every query, retrieved context hash, adapter version, confidence score, and answer is committed to a SHA-256 hash-chained SQLite ledger:
   H_n = SHA-256(H_(n-1) || timestamp || query || adapter_hash || answer)
   Any retroactive modification invalidates the entire chain, guaranteeing judicial defensibility.

---

## 6. Empirical Evaluation & Ablation Analysis

| Regime Update | Retained Accuracy (Avg) | Backward Transfer (BWT) | Citation Hallucination Rate | Basis Rank | Adapter Footprint |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Q1 (Initial)** | 96.0% | 0.000 | 2.0% | 16 | 18.5 MB |
| **Q2 (+ TIES Merge)** | 94.5% | -0.015 | 2.5% | 16 (Reset) | 18.5 MB |
| **Q3 (Incremental)** | 95.2% | -0.010 | 2.2% | 32 | 37.0 MB |
| **Q4 (+ TIES Merge)** | 94.8% | -0.012 | 2.1% | 16 (Reset) | 18.5 MB |

### Comparative Ablation
- Naive LoRA (Sequential): BWT degraded to -0.218 after Q4; severe catastrophic forgetting of Q1 contract rules.
- Pure RAG (No Fine-Tuning): Citation hallucination spiked on cross-regime reasoning (failed 75% of confusion set questions).
- The Magistrate (O-LoRA + TIES): BWT maintained within -0.012, hallucination < 2.5%, memory strictly bounded.

---

## 7. Conclusion
The Magistrate establishes that air-gapped regulatory compliance does not require choosing between catastrophic amnesia and privacy violations. By mathematically enforcing subspace orthogonality, periodically consolidating parameter drift, and cryptographically chaining inference provenance, it delivers a mathematically defensible, edge-deployable compliance copilot.
