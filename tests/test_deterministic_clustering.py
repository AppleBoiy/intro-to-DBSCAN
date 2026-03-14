"""
Property-Based Test for Deterministic Clustering

Feature: comprehensive-dbscan-learning-repository
Property 2: Deterministic Clustering

**Validates: Requirements 14.2, 14.6**

This test validates that DBSCAN produces deterministic results - running
the algorithm multiple times on the same input with the same parameters
always produces identical cluster assignments. This is a fundamental
property that distinguishes DBSCAN from algorithms like K-Means that
use random initialization.
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dbscan_from_scratch import DBSCAN
from hypothesis import given, strategies as st, settings


@settings(max_examples=100, deadline=None)
@given(
    n_samples=st.integers(min_value=10, max_value=200),
    n_features=st.integers(min_value=2, max_value=5),
    eps=st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False),
    min_pts=st.integers(min_value=2, max_value=10),
    random_seed=st.integers(min_value=0, max_value=10000)
)
def test_deterministic_clustering(n_samples, n_features, eps, min_pts, random_seed):
    """
    Feature: comprehensive-dbscan-learning-repository
    Property 2: Deterministic Clustering
    
    For any dataset and parameters, running DBSCAN multiple times
    should produce identical cluster assignments.
    
    This property test generates random datasets with varying:
    - Sample counts (10-200 points)
    - Feature dimensions (2-5 dimensions)
    - Epsilon values (0.1-3.0)
    - MinPts values (2-10)
    - Random seeds for data generation
    
    The test runs DBSCAN twice on the same data and verifies that
    the cluster labels are identical, confirming deterministic behavior.
    """
    # Generate random dataset with fixed seed for this test iteration
    np.random.seed(random_seed)
    X = np.random.randn(n_samples, n_features)
    
    # First run
    dbscan1 = DBSCAN(eps=eps, min_pts=min_pts)
    labels1 = dbscan1.fit_predict(X)
    
    # Second run on the same data
    dbscan2 = DBSCAN(eps=eps, min_pts=min_pts)
    labels2 = dbscan2.fit_predict(X)
    
    # Labels should be identical
    assert np.array_equal(labels1, labels2), (
        f"Deterministic clustering failed!\n"
        f"  Dataset: {n_samples} samples, {n_features} features\n"
        f"  Parameters: eps={eps:.3f}, min_pts={min_pts}\n"
        f"  Random seed: {random_seed}\n"
        f"  First run labels: {labels1}\n"
        f"  Second run labels: {labels2}\n"
        f"  Difference found at indices: {np.where(labels1 != labels2)[0]}"
    )
    
    # Also verify that core sample indices are identical
    assert np.array_equal(dbscan1.core_sample_indices_, dbscan2.core_sample_indices_), (
        f"Core sample indices differ between runs!\n"
        f"  First run: {dbscan1.core_sample_indices_}\n"
        f"  Second run: {dbscan2.core_sample_indices_}"
    )
    
    # Verify that components are identical
    assert np.array_equal(dbscan1.components_, dbscan2.components_), (
        f"Components differ between runs!\n"
        f"  First run shape: {dbscan1.components_.shape}\n"
        f"  Second run shape: {dbscan2.components_.shape}"
    )


if __name__ == "__main__":
    # Run the property test
    print("Running deterministic clustering property test...")
    print("Testing with 100+ random examples...")
    test_deterministic_clustering()
    print("✓ All tests passed! DBSCAN is deterministic.")
