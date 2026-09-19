import numpy as np
import pytest
from src.offline.dpo_polish import DPOPolisher

def test_dpo_revalidation_option_a_acceptance():
    polisher = DPOPolisher(revalidation_mode="option_a", bwt_threshold=0.03)

    candidate = {
        "adapter_id": "adapter_Q2",
        "regime_id": "Q2",
        "A_matrix": np.random.randn(64, 8),
        "B_matrix": np.random.randn(8, 64)
    }
    seen = ["Q1", "Q2"]
    baselines = {"Q1": 0.95, "Q2": 0.96}
    held_out = [{"query": "test query", "reference_answer": "Under Clause AUD-201, audit is annual."}]

    final_adapter, passed, summary = polisher.apply_dpo_polish(
        candidate_adapter=candidate,
        seen_regimes=seen,
        historical_baselines=baselines,
        held_out_slice=held_out
    )

    assert passed is True
    assert summary["passed"] is True

def test_dpo_revalidation_option_a_rejection_and_fallback():
    # Set tight threshold (0.001) that triggers rejection
    polisher = DPOPolisher(revalidation_mode="option_a", bwt_threshold=0.001)

    candidate = {
        "adapter_id": "adapter_Q2_original",
        "regime_id": "Q2",
        "A_matrix": np.random.randn(64, 8),
        "B_matrix": np.random.randn(8, 64)
    }
    seen = ["Q1", "Q2"]
    baselines = {"Q1": 0.98, "Q2": 0.96}
    held_out = [{"query": "test", "reference_answer": "test"}]

    final_adapter, passed, summary = polisher.apply_dpo_polish(
        candidate_adapter=candidate,
        seen_regimes=seen,
        historical_baselines=baselines,
        held_out_slice=held_out
    )

    # DPO should be rejected and fall back to pre-DPO candidate adapter
    assert passed is False
    assert final_adapter["adapter_id"] == "adapter_Q2_original"
    assert summary.get("dpo_fallback") is True
