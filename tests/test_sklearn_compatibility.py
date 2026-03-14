"""
Comprehensive sklearn Compatibility Tests

Feature: comprehensive-dbscan-learning-repository
Task 11.4: Create sklearn compatibility tests

**Validates: Requirements 12.2**

This module contains comprehensive tests that validate equivalence
between our DBSCAN implementation and scikit-learn's DBSCAN across:
- Multiple dataset types (moons, blobs, circles, varying density)
- Various parameter combinations
- Core sample indices matching
- Edge cases (empty, single point, all noise)
"""
import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dbscan_from_scratch import DBSCAN
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.metrics import adjusted_rand_score
from hypothesis import given, strategies as st, settings, assume


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
# Comprehensive Unit Tests for sklearn Compatibility
# ============================================================================

class TestSklearnCompatibilityDatasets:
    """Test equivalence on multiple dataset types."""
    
    def test_moons_dataset(self):
        """
        Test equivalence on moon-shaped dataset.
        
        Moons dataset has two interleaving half circles - a classic
        test case for density-based clustering.
        """
        # Generate moons dataset
        X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
        
        # Test with appropriate parameters for moons
        eps = 0.3
        min_pts = 5
        
        # Our implementation
        our_dbscan = DBSCAN(eps=eps, min_pts=min_pts)
        our_labels = our_dbscan.fit_predict(X)
        
        # Scikit-learn implementation
        sklearn_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_pts)
        sklearn_labels = sklearn_dbscan.fit_predict(X)
        
        # Calculate adjusted Rand index
        ari = adjusted_rand_score(our_labels, sklearn_labels)
        
        assert ari > 0.99, (
            f"Moons dataset compatibility failed!\n"
            f"  ARI: {ari:.4f} (expected > 0.99)\n"
            f"  Our clusters: {set(our_labels)}\n"
            f"  Sklearn clusters: {set(sklearn_labels)}"
        )
    
    def test_blobs_dataset(self):
        """
        Test equivalence on blob-shaped dataset.
        
        Blobs dataset has well-separated spherical clusters.
        """
        # Generate blobs dataset
        X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.5, random_state=42)
        
        # Test with appropriate parameters for blobs
        eps = 0.8
        min_pts = 5
        
        # Our implementation
        our_dbscan = DBSCAN(eps=eps, min_pts=min_pts)
        our_labels = our_dbscan.fit_predict(X)
        
        # Scikit-learn implementation
        sklearn_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_pts)
        sklearn_labels = sklearn_dbscan.fit_predict(X)
        
        # Calculate adjusted Rand index
        ari = adjusted_rand_score(our_labels, sklearn_labels)
        
        assert ari > 0.99, (
            f"Blobs dataset compatibility failed!\n"
            f"  ARI: {ari:.4f} (expected > 0.99)\n"
            f"  Our clusters: {set(our_labels)}\n"
            f"  Sklearn clusters: {set(sklearn_labels)}"
        )
    
    def test_circles_dataset(self):
        """
        Test equivalence on concentric circles dataset.
        
        Circles dataset has nested circular clusters.
        """
        # Generate circles dataset
        X, _ = make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42)
        
        # Test with appropriate parameters for circles
        eps = 0.2
        min_pts = 5
        
        # Our implementation
        our_dbscan = DBSCAN(eps=eps, min_pts=min_pts)
        our_labels = our_dbscan.fit_predict(X)
        
        # Scikit-learn implementation
        sklearn_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_pts)
        sklearn_labels = sklearn_dbscan.fit_predict(X)
        
        # Calculate adjusted Rand index
        ari = adjusted_rand_score(our_labels, sklearn_labels)
        
        assert ari > 0.99, (
            f"Circles dataset compatibility failed!\n"
            f"  ARI: {ari:.4f} (expected > 0.99)\n"
            f"  Our clusters: {set(our_labels)}\n"
            f"  Sklearn clusters: {set(sklearn_labels)}"
        )
    
    def test_varying_density_dataset(self):
        """
        Test equivalence on dataset with varying density clusters.
        
        This dataset has clusters with different densities to test
        DBSCAN's ability to handle density variations.
        """
        # Generate varying density dataset
        # Dense cluster
        dense = np.random.randn(100, 2) * 0.3 + [0, 0]
        # Sparse cluster
        sparse = np.random.randn(100, 2) * 1.0 + [5, 5]
        X = np.vstack([dense, sparse])
        
        # Test with parameters that work for both densities
        eps = 0.5
        min_pts = 5
        
        # Our implementation
        our_dbscan = DBSCAN(eps=eps, min_pts=min_pts)
        our_labels = our_dbscan.fit_predict(X)
        
        # Scikit-learn implementation
        sklearn_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_pts)
        sklearn_labels = sklearn_dbscan.fit_predict(X)
        
        # Calculate adjusted Rand index
        ari = adjusted_rand_score(our_labels, sklearn_labels)
        
        assert ari > 0.99, (
            f"Varying density dataset compatibility failed!\n"
            f"  ARI: {ari:.4f} (expected > 0.99)\n"
            f"  Our clusters: {set(our_labels)}\n"
            f"  Sklearn clusters: {set(sklearn_labels)}"
        )


class TestSklearnCompatibilityParameters:
    """Test equivalence with various parameter combinations."""
    
    @pytest.mark.parametrize("eps,min_pts", [
        (0.3, 3),
        (0.3, 5),
        (0.3, 10),
        (0.5, 3),
        (0.5, 5),
        (0.5, 10),
        (1.0, 3),
        (1.0, 5),
        (1.0, 10),
    ])
    def test_parameter_combinations(self, eps, min_pts):
        """
        Test equivalence with various eps and min_pts combinations.
        
        This tests that our implementation matches sklearn across
        a range of parameter values.
        """
        # Generate test dataset
        X, _ = make_blobs(n_samples=150, centers=3, random_state=42)
        
        # Our implementation
        our_dbscan = DBSCAN(eps=eps, min_pts=min_pts)
        our_labels = our_dbscan.fit_predict(X)
        
        # Scikit-learn implementation
        sklearn_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_pts)
        sklearn_labels = sklearn_dbscan.fit_predict(X)
        
        # Calculate adjusted Rand index
        ari = adjusted_rand_score(our_labels, sklearn_labels)
        
        assert ari > 0.99, (
            f"Parameter combination failed!\n"
            f"  eps={eps}, min_pts={min_pts}\n"
            f"  ARI: {ari:.4f} (expected > 0.99)\n"
            f"  Our clusters: {set(our_labels)}\n"
            f"  Sklearn clusters: {set(sklearn_labels)}"
        )
    
    def test_small_eps(self):
        """Test with very small epsilon (tight clustering)."""
        X, _ = make_blobs(n_samples=100, centers=2, random_state=42)
        
        eps = 0.1
        min_pts = 3
        
        our_dbscan = DBSCAN(eps=eps, min_pts=min_pts)
        our_labels = our_dbscan.fit_predict(X)
        
        sklearn_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_pts)
        sklearn_labels = sklearn_dbscan.fit_predict(X)
        
        ari = adjusted_rand_score(our_labels, sklearn_labels)
        assert ari > 0.99
    
    def test_large_eps(self):
        """Test with very large epsilon (loose clustering)."""
        X, _ = make_blobs(n_samples=100, centers=2, random_state=42)
        
        eps = 5.0
        min_pts = 3
        
        our_dbscan = DBSCAN(eps=eps, min_pts=min_pts)
        our_labels = our_dbscan.fit_predict(X)
        
        sklearn_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_pts)
        sklearn_labels = sklearn_dbscan.fit_predict(X)
        
        ari = adjusted_rand_score(our_labels, sklearn_labels)
        assert ari > 0.99
    
    def test_small_min_pts(self):
        """Test with minimum min_pts value."""
        X, _ = make_blobs(n_samples=100, centers=2, random_state=42)
        
        eps = 0.5
        min_pts = 2
        
        our_dbscan = DBSCAN(eps=eps, min_pts=min_pts)
        our_labels = our_dbscan.fit_predict(X)
        
        sklearn_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_pts)
        sklearn_labels = sklearn_dbscan.fit_predict(X)
        
        ari = adjusted_rand_score(our_labels, sklearn_labels)
        assert ari > 0.99
    
    def test_large_min_pts(self):
        """Test with large min_pts value."""
        X, _ = make_blobs(n_samples=200, centers=2, random_state=42)
        
        eps = 0.8
        min_pts = 20
        
        our_dbscan = DBSCAN(eps=eps, min_pts=min_pts)
        our_labels = our_dbscan.fit_predict(X)
        
        sklearn_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_pts)
        sklearn_labels = sklearn_dbscan.fit_predict(X)
        
        ari = adjusted_rand_score(our_labels, sklearn_labels)
        assert ari > 0.99


class TestSklearnCompatibilityCoreIndices:
    """Test that core_sample_indices_ match between implementations."""
    
    def test_core_indices_match(self):
        """
        Test that core sample indices match sklearn.
        
        Both implementations should identify the same points as core points.
        """
        X, _ = make_blobs(n_samples=150, centers=3, random_state=42)
        
        eps = 0.8
        min_pts = 5
        
        # Our implementation
        our_dbscan = DBSCAN(eps=eps, min_pts=min_pts)
        our_dbscan.fit_predict(X)
        our_core_indices = set(our_dbscan.core_sample_indices_)
        
        # Scikit-learn implementation
        sklearn_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_pts)
        sklearn_dbscan.fit_predict(X)
        sklearn_core_indices = set(sklearn_dbscan.core_sample_indices_)
        
        # Core indices should match exactly
        assert our_core_indices == sklearn_core_indices, (
            f"Core sample indices don't match!\n"
            f"  Our core indices: {len(our_core_indices)} points\n"
            f"  Sklearn core indices: {len(sklearn_core_indices)} points\n"
            f"  Difference: {our_core_indices.symmetric_difference(sklearn_core_indices)}"
        )
    
    def test_core_indices_moons(self):
        """Test core indices match on moons dataset."""
        X, _ = make_moons(n_samples=200, noise=0.05, random_state=42)
        
        eps = 0.3
        min_pts = 5
        
        our_dbscan = DBSCAN(eps=eps, min_pts=min_pts)
        our_dbscan.fit_predict(X)
        our_core_indices = set(our_dbscan.core_sample_indices_)
        
        sklearn_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_pts)
        sklearn_dbscan.fit_predict(X)
        sklearn_core_indices = set(sklearn_dbscan.core_sample_indices_)
        
        assert our_core_indices == sklearn_core_indices
    
    def test_core_indices_circles(self):
        """Test core indices match on circles dataset."""
        X, _ = make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42)
        
        eps = 0.2
        min_pts = 5
        
        our_dbscan = DBSCAN(eps=eps, min_pts=min_pts)
        our_dbscan.fit_predict(X)
        our_core_indices = set(our_dbscan.core_sample_indices_)
        
        sklearn_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_pts)
        sklearn_dbscan.fit_predict(X)
        sklearn_core_indices = set(sklearn_dbscan.core_sample_indices_)
        
        assert our_core_indices == sklearn_core_indices


class TestSklearnCompatibilityEdgeCases:
    """Test edge cases match sklearn behavior."""
    
    def test_empty_dataset(self):
        """
        Test that empty dataset is handled identically.
        
        Note: sklearn raises an error on empty datasets, so we only
        test that our implementation handles it gracefully.
        """
        X = np.array([]).reshape(0, 2)
        
        eps = 0.5
        min_pts = 5
        
        # Our implementation should handle empty datasets gracefully
        our_dbscan = DBSCAN(eps=eps, min_pts=min_pts)
        our_labels = our_dbscan.fit_predict(X)
        
        # Should return empty arrays
        assert len(our_labels) == 0
        assert len(our_dbscan.core_sample_indices_) == 0
        assert our_dbscan.components_.shape == (0, 2)
    
    def test_single_point(self):
        """Test that single point is classified as noise."""
        X = np.array([[0, 0]])
        
        eps = 0.5
        min_pts = 2
        
        # Our implementation
        our_dbscan = DBSCAN(eps=eps, min_pts=min_pts)
        our_labels = our_dbscan.fit_predict(X)
        
        # Scikit-learn implementation
        sklearn_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_pts)
        sklearn_labels = sklearn_dbscan.fit_predict(X)
        
        # Both should classify as noise
        assert our_labels[0] == -1
        assert sklearn_labels[0] == -1
        assert len(our_dbscan.core_sample_indices_) == 0
        assert len(sklearn_dbscan.core_sample_indices_) == 0
    
    def test_all_noise(self):
        """
        Test dataset where all points are noise.
        
        With very small epsilon, all points should be classified as noise.
        """
        X = np.random.randn(50, 2) * 10  # Widely scattered points
        
        eps = 0.1  # Very small epsilon
        min_pts = 5
        
        # Our implementation
        our_dbscan = DBSCAN(eps=eps, min_pts=min_pts)
        our_labels = our_dbscan.fit_predict(X)
        
        # Scikit-learn implementation
        sklearn_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_pts)
        sklearn_labels = sklearn_dbscan.fit_predict(X)
        
        # Both should classify all as noise
        assert np.all(our_labels == -1)
        assert np.all(sklearn_labels == -1)
        assert len(our_dbscan.core_sample_indices_) == 0
        assert len(sklearn_dbscan.core_sample_indices_) == 0
    
    def test_single_cluster(self):
        """
        Test dataset that forms a single cluster.
        
        With large epsilon, all points should form one cluster.
        """
        X = np.random.randn(50, 2) * 0.5  # Tightly packed points
        
        eps = 5.0  # Very large epsilon
        min_pts = 3
        
        # Our implementation
        our_dbscan = DBSCAN(eps=eps, min_pts=min_pts)
        our_labels = our_dbscan.fit_predict(X)
        
        # Scikit-learn implementation
        sklearn_dbscan = SklearnDBSCAN(eps=eps, min_samples=min_pts)
        sklearn_labels = sklearn_dbscan.fit_predict(X)
        
        # Calculate adjusted Rand index
        ari = adjusted_rand_score(our_labels, sklearn_labels)
        assert ari > 0.99
        
        # Both should find one cluster (no noise)
        our_n_clusters = len(set(our_labels)) - (1 if -1 in our_labels else 0)
        sklearn_n_clusters = len(set(sklearn_labels)) - (1 if -1 in sklearn_labels else 0)
        
        assert our_n_clusters == sklearn_n_clusters
        assert our_n_clusters == 1


if __name__ == "__main__":
    # Run the property test
    print("Running sklearn compatibility property test...")
    print("Testing with 100+ random examples...")
    test_sklearn_compatibility()
    print("✓ All tests passed! Our DBSCAN is compatible with scikit-learn.")
