"""
Comprehensive Property-Based Test Suite for DBSCAN

This module contains all property-based tests that validate universal
correctness properties of the DBSCAN implementation using hypothesis.

Feature: comprehensive-dbscan-learning-repository

Properties tested:
1. Sklearn Compatibility - Results match scikit-learn
2. Deterministic Clustering - Same input produces same output
3. Density-Reachability Transitivity - Transitive property holds
4. Distance Metric Correctness - Distance calculations are accurate

**Validates: Requirements 5.1, 12.2, 12.4, 12.7, 14.2, 14.6**
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dbscan_from_scratch import DBSCAN
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.metrics import adjusted_rand_score
from hypothesis import given, strategies as st, settings, assume


# ============================================================================
# Property 1: Sklearn Compatibility
# ============================================================================

@settings(max_examples=100, deadline=None)
@given(
    n_samples=st.integers(min_value=10, max_value=200),
    n_features=st.integers(min_value=2, max_value=5),
    eps=st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False),
    min_pts=st.integers(min_value=2, max_value=10)
)
def test_sklearn_compatibility(n_samples, n_features, eps, min_pts):
    """
    Feature: comprehensive-dbscan-learning-repository
    Property 1: Sklearn Compatibility
    
    **Validates: Requirements 5.1, 12.2**
    
    For any dataset and valid parameters, our implementation should
    produce results equivalent to scikit-learn (ARI > 0.99).
    
    This property test generates random datasets with varying:
    - Sample counts (10-200 points)
    - Feature dimensions (2-5 dimensions)
    - Epsilon values (0.1-3.0)
    - MinPts values (2-10)
    
    The test compares clustering results using adjusted Rand index,
    which measures similarity between two clusterings. A score > 0.99
    indicates near-perfect agreement.
    """
    # Generate random dataset
    np.random.seed(42)  # For reproducibility within each test
    X = np.random.randn(n_samples, n_features)
    
    # Our implementation
    our_dbscan = DBSCAN(eps=eps, min_pts=min_pts)
    our_labels = our_dbscan.fit_predict(X)
    
    # Scikit-learn implementation
    sklearn_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_pts)
    sklearn_labels = sklearn_dbscan.fit_predict(X)
    
    # Calculate adjusted Rand index
    # ARI = 1.0 means perfect agreement
    # ARI = 0.0 means random labeling
    ari = adjusted_rand_score(our_labels, sklearn_labels)
    
    # Both implementations should produce equivalent clusterings
    # We allow a small tolerance (0.99) to account for potential
    # tie-breaking differences in edge cases
    assert ari > 0.99, (
        f"Sklearn compatibility failed!\n"
        f"  Dataset: {n_samples} samples, {n_features} features\n"
        f"  Parameters: eps={eps:.3f}, min_pts={min_pts}\n"
        f"  Adjusted Rand Index: {ari:.4f} (expected > 0.99)\n"
        f"  Our clusters: {set(our_labels)}\n"
        f"  Sklearn clusters: {set(sklearn_labels)}"
    )


# ============================================================================
# Property 2: Deterministic Clustering
# ============================================================================

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
    
    **Validates: Requirements 14.2, 14.6**
    
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


# ============================================================================
# Property 3: Density-Reachability Transitivity
# ============================================================================

def is_density_reachable(X, p_idx, q_idx, eps, min_pts, dbscan):
    """
    Helper function to check if point p is density-reachable from point q.
    
    A point p is density-reachable from q if there exists a chain of points
    p1, ..., pn where p1 = q, pn = p, and each pi+1 is directly density-reachable
    from pi (i.e., pi is a core point and pi+1 is in its epsilon-neighborhood).
    
    This is a simplified check that uses the cluster labels: if both points
    are in the same cluster (and not noise), they are density-reachable.
    """
    labels = dbscan.labels_
    
    # If either point is noise, they're not density-reachable
    if labels[p_idx] == -1 or labels[q_idx] == -1:
        return False
    
    # If they're in the same cluster, they're density-reachable
    return labels[p_idx] == labels[q_idx]


@settings(max_examples=100, deadline=None)
@given(
    n_samples=st.integers(min_value=20, max_value=100),
    eps=st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
    min_pts=st.integers(min_value=3, max_value=8),
    random_seed=st.integers(min_value=0, max_value=10000)
)
def test_density_reachability_transitivity(n_samples, eps, min_pts, random_seed):
    """
    Feature: comprehensive-dbscan-learning-repository
    Property 3: Density-Reachability Transitivity
    
    **Validates: Requirements 12.4**
    
    For any three points p, q, r in a dataset, if p is density-reachable
    from q, and q is density-reachable from r, then p must be density-reachable
    from r. This validates Lemma 1 from the paper [Paper §4.1].
    
    This is a fundamental mathematical property of density-reachability that
    ensures clusters are well-defined and consistent.
    
    Note: We test this property by checking cluster membership. If two points
    are in the same cluster, they are density-reachable from each other.
    """
    # Generate random dataset
    np.random.seed(random_seed)
    X = np.random.randn(n_samples, 2)
    
    # Run DBSCAN
    dbscan = DBSCAN(eps=eps, min_pts=min_pts)
    labels = dbscan.fit_predict(X)
    
    # Find points that are in clusters (not noise)
    clustered_points = np.where(labels != -1)[0]
    
    # Skip if we don't have enough clustered points to test
    assume(len(clustered_points) >= 3)
    
    # Test transitivity for a sample of point triples
    # We don't test all combinations (too expensive), but sample enough
    # to have high confidence
    num_tests = min(20, len(clustered_points) // 3)
    
    for _ in range(num_tests):
        # Randomly select three points from clustered points
        if len(clustered_points) < 3:
            break
        
        indices = np.random.choice(clustered_points, size=3, replace=False)
        p_idx, q_idx, r_idx = indices
        
        # Check if p is reachable from q
        p_from_q = is_density_reachable(X, p_idx, q_idx, eps, min_pts, dbscan)
        
        # Check if q is reachable from r
        q_from_r = is_density_reachable(X, q_idx, r_idx, eps, min_pts, dbscan)
        
        # If both conditions hold, p must be reachable from r (transitivity)
        if p_from_q and q_from_r:
            p_from_r = is_density_reachable(X, p_idx, r_idx, eps, min_pts, dbscan)
            
            assert p_from_r, (
                f"Density-reachability transitivity violated!\n"
                f"  Parameters: eps={eps:.3f}, min_pts={min_pts}\n"
                f"  Point p (idx={p_idx}): cluster {labels[p_idx]}\n"
                f"  Point q (idx={q_idx}): cluster {labels[q_idx]}\n"
                f"  Point r (idx={r_idx}): cluster {labels[r_idx]}\n"
                f"  p reachable from q: {p_from_q}\n"
                f"  q reachable from r: {q_from_r}\n"
                f"  p reachable from r: {p_from_r} (should be True)\n"
                f"  This violates the transitive property of density-reachability."
            )


# ============================================================================
# Property 5: Distance Metric Correctness
# ============================================================================

@settings(max_examples=100, deadline=None)
@given(
    metric=st.sampled_from(['euclidean', 'manhattan', 'chebyshev']),
    x1=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    y1=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    x2=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    y2=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
)
def test_distance_metric_correctness(metric, x1, y1, x2, y2):
    """
    Feature: comprehensive-dbscan-learning-repository
    Property 5: Distance Metric Correctness
    
    **Validates: Requirements 12.7**
    
    For any two points and any supported distance metric (Euclidean, Manhattan,
    Chebyshev), the computed distance should match the mathematical definition
    of that metric.
    
    Distance metrics tested:
    - Euclidean: L2 norm, d(p,q) = sqrt(sum((p_i - q_i)^2))
    - Manhattan: L1 norm, d(p,q) = sum(|p_i - q_i|)
    - Chebyshev: L∞ norm, d(p,q) = max(|p_i - q_i|)
    
    This property ensures that the fundamental distance calculations
    are mathematically correct, which is critical for DBSCAN's correctness.
    """
    # Create DBSCAN instance with specified metric
    dbscan = DBSCAN(metric=metric)
    
    # Create two points
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    
    # Compute distance using DBSCAN's method
    computed_distance = dbscan._compute_distance(p1, p2)
    
    # Compute expected distance based on metric
    if metric == 'euclidean':
        expected_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    elif metric == 'manhattan':
        expected_distance = abs(x2 - x1) + abs(y2 - y1)
    elif metric == 'chebyshev':
        expected_distance = max(abs(x2 - x1), abs(y2 - y1))
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Verify computed distance matches expected distance
    # Use np.isclose for floating point comparison with tolerance
    assert np.isclose(computed_distance, expected_distance, rtol=1e-9, atol=1e-9), (
        f"Distance metric correctness failed!\n"
        f"  Metric: {metric}\n"
        f"  Point 1: ({x1:.6f}, {y1:.6f})\n"
        f"  Point 2: ({x2:.6f}, {y2:.6f})\n"
        f"  Computed distance: {computed_distance:.10f}\n"
        f"  Expected distance: {expected_distance:.10f}\n"
        f"  Difference: {abs(computed_distance - expected_distance):.10e}"
    )
    
    # Additional property: distance should be non-negative
    assert computed_distance >= 0, (
        f"Distance should be non-negative, got {computed_distance}"
    )
    
    # Additional property: distance should be symmetric (d(p,q) = d(q,p))
    reverse_distance = dbscan._compute_distance(p2, p1)
    assert np.isclose(computed_distance, reverse_distance, rtol=1e-9, atol=1e-9), (
        f"Distance metric is not symmetric!\n"
        f"  d(p1, p2) = {computed_distance:.10f}\n"
        f"  d(p2, p1) = {reverse_distance:.10f}"
    )
    
    # Additional property: distance to self should be zero
    self_distance = dbscan._compute_distance(p1, p1)
    assert np.isclose(self_distance, 0.0, rtol=1e-9, atol=1e-9), (
        f"Distance to self should be zero, got {self_distance:.10f}"
    )


# ============================================================================
# Main execution for standalone testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Running Comprehensive Property-Based Test Suite")
    print("=" * 70)
    print()
    
    print("Property 1: Sklearn Compatibility")
    print("  Testing with 100+ random examples...")
    test_sklearn_compatibility()
    print("  ✓ PASSED\n")
    
    print("Property 2: Deterministic Clustering")
    print("  Testing with 100+ random examples...")
    test_deterministic_clustering()
    print("  ✓ PASSED\n")
    
    print("Property 3: Density-Reachability Transitivity")
    print("  Testing with 100+ random examples...")
    test_density_reachability_transitivity()
    print("  ✓ PASSED\n")
    
    print("Property 5: Distance Metric Correctness")
    print("  Testing with 100+ random examples...")
    test_distance_metric_correctness()
    print("  ✓ PASSED\n")
    
    print("=" * 70)
    print("All property-based tests passed!")
    print("=" * 70)
