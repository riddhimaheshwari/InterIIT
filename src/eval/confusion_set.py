from typing import List, Dict, Any

ADVERSARIAL_CONFUSION_SET: List[Dict[str, Any]] = [
    {
        "id": "CONFUSION-01",
        "title": "Incident Reporting Timeline Conflict (Q1 vs Q3)",
        "query": "A high-risk biometric screening system experiences a safety malfunction causing minor financial loss. Is the operator required to report within 14 days under AI-104 or 72 hours under INC-301?",
        "cross_regimes": ["Q1", "Q3"],
        "governing_regime": "Q3",
        "expected_synthesis": "Clause INC-301 (Q3) explicitly supersedes Clause AI-104 (Q1). Therefore, the operator must report within 72 hours, not 14 days.",
        "key_requirements": ["supersedes AI-104", "72 hours", "INC-301", "not 14 days"],
        "confusing_elements": ["AI-104 was 14 days in Q1, but amended in Q3"]
    },
    {
        "id": "CONFUSION-02",
        "title": "Autonomous Liability and Human Oversight Synergy (Q1 vs Q3)",
        "query": "If an autonomous credit scoring system makes a discriminatory determination, how does the 15-second oversight rule in AI-102 interact with the strict liability provisions in INC-302?",
        "cross_regimes": ["Q1", "Q3"],
        "governing_regime": "Both (Q1 + Q3)",
        "expected_synthesis": "Under AI-102 (Q1), human oversight must enable override within 15 seconds. If deployers fail to maintain this certified oversight, INC-302 (Q3) imposes strict financial liability for quantifiable damages caused by algorithmic hallucinations or errors.",
        "key_requirements": ["AI-102", "15 seconds", "INC-302", "strict financial liability"],
        "confusing_elements": ["Oversight definition from Q1 triggers liability shield or exposure in Q3"]
    },
    {
        "id": "CONFUSION-03",
        "title": "Edge Compute Mandate with Continuous Telemetry (Q2 vs Q4)",
        "query": "Can real-time telemetry tracking prediction confidence under AUD-203 be streamed to a centralized cloud analytics server under SVR-401?",
        "cross_regimes": ["Q2", "Q4"],
        "governing_regime": "Q4",
        "expected_synthesis": "While AUD-203 (Q2) mandates telemetry and confidence logging with 5% drift triage, SVR-401 (Q4) strictly prohibits outbound internet transmission from compliance edge hardware. Telemetry and drift monitoring must be executed locally on the edge.",
        "key_requirements": ["AUD-203", "SVR-401", "local telemetry", "no outbound internet"],
        "confusing_elements": ["Telemetry obligation from Q2 must conform to sovereign edge constraint in Q4"]
    },
    {
        "id": "CONFUSION-04",
        "title": "Model Card Provenance and Adapter Transfer (Q2 vs Q4)",
        "query": "When exporting a fine-tuned LoRA adapter across borders, what dual documentation is required under model card standards and weight transfer regulations?",
        "cross_regimes": ["Q2", "Q4"],
        "governing_regime": "Both (Q2 + Q4)",
        "expected_synthesis": "The exporter must attach a standardized public model card (AUD-204 from Q2) and a cryptographically signed provenance certificate identifying training lineage (SVR-402 from Q4).",
        "key_requirements": ["AUD-204", "model card", "SVR-402", "cryptographically signed provenance certificate"],
        "confusing_elements": ["Model card ethical/benchmark disclosures combined with cryptographic weight certificates"]
    }
]

def get_confusion_benchmark() -> List[Dict[str, Any]]:
    return ADVERSARIAL_CONFUSION_SET
