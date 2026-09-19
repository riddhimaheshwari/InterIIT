import numpy as np
import pytest
from src.offline.merge import TIESConsolidator

def test_ties_merge_trimming_and_sign_election():
    consolidator = TIESConsolidator(trim_threshold=0.50)

    # 3 matrices with clear dominant signs
    # Matrix 1: Positive values
    m1 = np.array([[10.0, 20.0], [0.1, -0.2]])
    # Matrix 2: Positive values agreeing with m1
    m2 = np.array([[8.0, 15.0], [-0.1, 0.2]])
    # Matrix 3: Conflicting negative values
    m3 = np.array([[-1.0, 12.0], [0.0, 0.0]])

    merged = consolidator.ties_merge_matrices([m1, m2, m3])

    # Position (0, 0): top magnitude keeps positive, majority sign is +1
    assert merged[0, 0] > 0
    # Position (0, 1): all agree positive
    assert merged[0, 1] > 10.0
    # Position (1, 0): trimmed small values
    assert np.abs(merged[1, 0]) < 1.0

def test_ties_merge_disjoint_averaging():
    consolidator = TIESConsolidator(trim_threshold=1.0) # Keep all

    # Both agree on sign (+1) with values 4.0 and 6.0 => average = 5.0
    m1 = np.array([[4.0]])
    m2 = np.array([[6.0]])

    merged = consolidator.ties_merge_matrices([m1, m2])
    np.testing.assert_allclose(merged[0, 0], 5.0)
