import numpy as np
import pytest
from src.offline.train_adapter import OrthogonalBasisTracker

def test_qr_basis_initialization_and_orthonormality():
    tracker = OrthogonalBasisTracker(hidden_dim=64, rank=8)
    assert tracker.get_basis_rank() == 0

    A1 = np.random.randn(64, 8)
    tracker.update_basis(A1)
    
    assert tracker.get_basis_rank() == 8
    Q = tracker.basis_matrix
    # Check Q^T Q == I_8
    identity = np.dot(Q.T, Q)
    np.testing.assert_allclose(identity, np.eye(8), atol=1e-5)

def test_incremental_basis_update_and_orthogonality_loss():
    tracker = OrthogonalBasisTracker(hidden_dim=64, rank=8)
    A1 = np.random.randn(64, 8)
    tracker.update_basis(A1)

    # A2 in the same subspace as Q should yield high orthogonality penalty
    A2_parallel = tracker.basis_matrix @ np.random.randn(8, 8)
    loss_parallel = tracker.compute_orthogonality_loss(A2_parallel)
    assert loss_parallel > 0.5

    # A2 orthogonal to Q should yield zero orthogonality penalty
    Q = tracker.basis_matrix
    # Nullspace projection
    P_perp = np.eye(64) - Q @ Q.T
    A2_ortho = P_perp @ np.random.randn(64, 8)
    loss_ortho = tracker.compute_orthogonality_loss(A2_ortho)
    assert loss_ortho < 1e-6

def test_basis_reset_at_ties_merge():
    tracker = OrthogonalBasisTracker(hidden_dim=64, rank=8)
    A1 = np.random.randn(64, 8)
    A2 = np.random.randn(64, 8)
    tracker.update_basis(A1)
    tracker.update_basis(A2)
    assert tracker.get_basis_rank() == 16

    # Merged adapter consolidation
    A_merged = (A1 + A2) / 2.0
    tracker.reset_basis_to_merged(A_merged)
    
    # Basis rank resets to 8 (LoRA rank)
    assert tracker.get_basis_rank() == 8
    Q_reset = tracker.basis_matrix
    np.testing.assert_allclose(np.dot(Q_reset.T, Q_reset), np.eye(8), atol=1e-5)
