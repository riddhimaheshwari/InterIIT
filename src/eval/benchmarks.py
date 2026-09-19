import json
from pathlib import Path
from typing import List, Dict, Any

def get_held_out_benchmark(regime_id: str) -> List[Dict[str, Any]]:
    benchmarks = {
        "Q1": [
            {
                "id": "Q1-QA-001",
                "type": "statutory_qa",
                "regime_id": "Q1",
                "query": "What classification applies to an AI system deployed for biometric identification and credit scoring?",
                "reference_answer": "Under Clause AI-101, any AI system deployed in critical infrastructure, employment screening, biometric identification, or credit scoring is classified as a High-Risk AI System requiring documented risk mitigation.",
                "target_clauses": ["Clause AI-101"],
                "key_facts": ["High-Risk AI System", "Clause AI-101", "risk mitigation"]
            },
            {
                "id": "Q1-QA-002",
                "type": "statutory_qa",
                "regime_id": "Q1",
                "query": "What is the maximum allowed response time for human oversight intervention upon an anomaly flag in high-risk AI?",
                "reference_answer": "Under Clause AI-102, human oversight mechanisms must enable an operator to intervene, override, or abort automated decisions within 15 seconds.",
                "target_clauses": ["Clause AI-102"],
                "key_facts": ["15 seconds", "Clause AI-102", "override"]
            },
            {
                "id": "Q1-MC-003",
                "type": "multi_clause_reasoning",
                "regime_id": "Q1",
                "query": "How long must training data lineage records be preserved for high-risk systems undergoing bias assessments?",
                "reference_answer": "Pursuant to Clause AI-103, training and validation datasets for high-risk models must undergo bias assessments and documented lineage records must be retained for 5 years.",
                "target_clauses": ["Clause AI-103"],
                "key_facts": ["5 years", "Clause AI-103", "bias assessments"]
            },
            {
                "id": "Q1-QA-004",
                "type": "statutory_qa",
                "regime_id": "Q1",
                "query": "What technical documentation must be maintained prior to major model releases?",
                "reference_answer": "Under Clause AI-105, operators must maintain a technical file detailing model architecture, hyperparameter choices, energy expenditure, and validation bounds.",
                "target_clauses": ["Clause AI-105"],
                "key_facts": ["Clause AI-105", "architecture", "energy expenditure"]
            }
        ],
        "Q2": [
            {
                "id": "Q2-QA-001",
                "type": "statutory_qa",
                "regime_id": "Q2",
                "query": "What are the audit obligations for entities operating tier-1 algorithmic systems?",
                "reference_answer": "Under Clause AUD-201, tier-1 algorithmic operators must commission an annual independent third-party audit evaluating model drift, adversarial robustness, and disparate impact.",
                "target_clauses": ["Clause AUD-201"],
                "key_facts": ["Clause AUD-201", "annual independent third-party audit", "disparate impact"]
            },
            {
                "id": "Q2-MC-002",
                "type": "multi_clause_reasoning",
                "regime_id": "Q2",
                "query": "What rights does a consumer have when subjected to an adverse automated decision?",
                "reference_answer": "According to Clause AUD-202, consumers are legally entitled to receive a counterfactual explanation detailing principal factors and minimal input adjustments required for recourse.",
                "target_clauses": ["Clause AUD-202"],
                "key_facts": ["Clause AUD-202", "counterfactual explanation", "recourse"]
            },
            {
                "id": "Q2-QA-003",
                "type": "statutory_qa",
                "regime_id": "Q2",
                "query": "What threshold of telemetry distributional delta triggers automated alert triage in production monitoring?",
                "reference_answer": "Under Clause AUD-203, continuous telemetry tracking prediction confidence must trigger alert triage when distributional delta exceeds 5%.",
                "target_clauses": ["Clause AUD-203"],
                "key_facts": ["Clause AUD-203", "5% distributional delta", "telemetry"]
            }
        ],
        "Q3": [
            {
                "id": "Q3-QA-001",
                "type": "statutory_qa",
                "regime_id": "Q3",
                "query": "What is the mandatory reporting timeframe for critical AI malfunctions resulting in financial loss?",
                "reference_answer": "Under Clause INC-301, which supersedes AI-104, critical incidents must be reported to the national supervisory authority within 72 hours.",
                "target_clauses": ["Clause INC-301"],
                "key_facts": ["Clause INC-301", "72 hours", "supersedes"]
            },
            {
                "id": "Q3-MC-002",
                "type": "multi_clause_reasoning",
                "regime_id": "Q3",
                "query": "Who bears liability when an autonomous decision agent operates without certified human oversight and causes damages?",
                "reference_answer": "Under Clause INC-302, deployers of autonomous agents lacking human oversight (Clause AI-102) bear strict financial liability for quantifiable damages caused by algorithmic hallucinations.",
                "target_clauses": ["Clause INC-302", "Clause AI-102"],
                "key_facts": ["Clause INC-302", "strict financial liability", "Clause AI-102"]
            },
            {
                "id": "Q3-QA-003",
                "type": "statutory_qa",
                "regime_id": "Q3",
                "query": "How long must cryptographic inference audit trails and adapter hashes be stored?",
                "reference_answer": "Clause INC-303 requires all inference decisions, retrieval contexts, and adapter version hashes to be stored in a tamper-evident audit ledger for at least 7 years.",
                "target_clauses": ["Clause INC-303"],
                "key_facts": ["Clause INC-303", "7 years", "tamper-evident"]
            }
        ],
        "Q4": [
            {
                "id": "Q4-QA-001",
                "type": "statutory_qa",
                "regime_id": "Q4",
                "query": "Are compliance assistants permitted to make external cloud LLM API calls under sovereign compute rules?",
                "reference_answer": "Under Clause SVR-401, compliance inference must run locally on edge hardware with zero outbound internet connectivity, and external cloud LLM calls are prohibited.",
                "target_clauses": ["Clause SVR-401"],
                "key_facts": ["Clause SVR-401", "edge hardware", "zero outbound internet"]
            },
            {
                "id": "Q4-MC-002",
                "type": "multi_clause_reasoning",
                "regime_id": "Q4",
                "query": "What maximum accuracy degradation is allowed when converting compliance models to 4-bit GGUF for edge deployment?",
                "reference_answer": "Pursuant to Clause SVR-403, 4-bit or 8-bit GGUF quantization must not cause more than 1.5% benchmark degradation compared to FP16 reference weights.",
                "target_clauses": ["Clause SVR-403"],
                "key_facts": ["Clause SVR-403", "1.5%", "4-bit GGUF"]
            },
            {
                "id": "Q4-QA-003",
                "type": "statutory_qa",
                "regime_id": "Q4",
                "query": "What is the maximum allowed catastrophic forgetting threshold during quarterly adapter merges?",
                "reference_answer": "Clause SVR-404 mandates that quarterly adapter consolidation and merge operations preserve backward accuracy with less than 3% catastrophic forgetting.",
                "target_clauses": ["Clause SVR-404"],
                "key_facts": ["Clause SVR-404", "3% catastrophic forgetting", "backward accuracy"]
            }
        ]
    }
    return benchmarks.get(regime_id, [])

def get_all_regimes() -> List[str]:
    return ["Q1", "Q2", "Q3", "Q4"]
