"""
Unit Tests for Parameter Tuning Module
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.parameter_tuning import ParameterSelector


def test_compute_k_distances_basic():
    """Test basic k-distance computation"""
    selector = ParameterSelector()
    
    # Simple 2D dataset
    X = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1],
        [5, 5], [6, 5], [5, 6], [6, 6]
    ])
    
    k_distances = selector.compute_k_distances(X, k=2)
    
    # Check output shape
    assert len(k_distances) == len(X), "Should return one distance per point"
    
    # Check that distances are sorted in descending order
    assert np.all(k_distances[:-1] >= k_distances[1:]), "Distances should be sorted descending"
    
    # Check that all distances are non-negative
    assert np.all(k_distances >= 0), "Distances should be non-negative"
    
    print("✓ compute_k_distances basic test passed")


def test_compute_k_distances_single_point():
    """Test k-distance with single point"""
    selector = ParameterSelector()
    X = np.array([[0, 0]])
    
    k_distances = selector.compute_k_distances(X, k=1)
    
    assert len(k_distances) == 1, "Should return one distance"
    assert k_distances[0] == 0.0, "Single point should have 0 distance"
    
    print("✓ compute_k_distances single point test passed")


def test_compute_k_distances_validation():
    """Test input validation for compute_k_distances"""
    selector = ParameterSelector()
    X = np.array([[0, 0], [1, 1], [2, 2]])
    
    # Test invalid k
    try:
        selector.compute_k_distances(X, k=0)
        assert False, "Should raise ValueError for k < 1"
    except ValueError:
        pass
    
    # Test k > n_samples
    try:
        selector.compute_k_distances(X, k=10)
        assert False, "Should raise ValueError for k > n_samples"
    except ValueError:
        pass
    
    # Test non-numpy array
    try:
        selector.compute_k_distances([[0, 0], [1, 1]], k=2)
        assert False, "Should raise TypeError for non-numpy array"
    except TypeError:
        pass
    
    print("✓ compute_k_distances validation test passed")



def test_find_elbow_point_basic():
    """Test basic elbow point detection"""
    selector = ParameterSelector()
    
    # Create a simple curve with clear elbow
    # High values, then sharp drop, then low values
    distances = np.array([10, 9.5, 9, 8.5, 8, 5, 2, 1.5, 1, 0.5, 0.3, 0.2, 0.1])
    
    elbow_idx, suggested_eps = selector.find_elbow_point(distances)
    
    # Check that we get valid output
    assert 0 <= elbow_idx < len(distances), "Elbow index should be within array bounds"
    assert suggested_eps == distances[elbow_idx], "Suggested eps should match distance at elbow"
    
    # The elbow should be somewhere in the middle (not at extremes)
    assert elbow_idx > 0, "Elbow should not be at first point"
    assert elbow_idx < len(distances) - 1, "Elbow should not be at last point"
    
    print("✓ find_elbow_point basic test passed")


def test_find_elbow_point_validation():
    """Test input validation for find_elbow_point"""
    selector = ParameterSelector()
    
    # Test empty array
    try:
        selector.find_elbow_point(np.array([]))
        assert False, "Should raise ValueError for empty array"
    except ValueError:
        pass
    
    # Test array with less than 3 points
    try:
        selector.find_elbow_point(np.array([1.0, 0.5]))
        assert False, "Should raise ValueError for < 3 points"
    except ValueError:
        pass
    
    print("✓ find_elbow_point validation test passed")


def test_grid_search_basic():
    """Test basic grid search functionality"""
    selector = ParameterSelector()
    
    # Create simple clustered data
    np.random.seed(42)
    cluster1 = np.random.randn(20, 2) * 0.3
    cluster2 = np.random.randn(20, 2) * 0.3 + [5, 5]
    X = np.vstack([cluster1, cluster2])
    
    eps_range = [0.3, 0.5, 0.7]
    minpts_range = [3, 4, 5]
    
    results = selector.grid_search(X, eps_range, minpts_range)
    
    # Check output structure
    assert 'best_eps' in results, "Should return best_eps"
    assert 'best_minpts' in results, "Should return best_minpts"
    assert 'best_score' in results, "Should return best_score"
    assert 'results' in results, "Should return results matrix"
    
    # Check that best parameters are from the search space
    assert results['best_eps'] in eps_range, "best_eps should be from eps_range"
    assert results['best_minpts'] in minpts_range, "best_minpts should be from minpts_range"
    
    # Check results matrix shape
    assert results['results'].shape == (len(eps_range), len(minpts_range))
    
    print("✓ grid_search basic test passed")


def test_grid_search_validation():
    """Test input validation for grid_search"""
    selector = ParameterSelector()
    X = np.array([[0, 0], [1, 1], [2, 2]])
    
    # Test empty eps_range
    try:
        selector.grid_search(X, [], [3, 4, 5])
        assert False, "Should raise ValueError for empty eps_range"
    except ValueError:
        pass
    
    # Test empty minpts_range
    try:
        selector.grid_search(X, [0.5, 1.0], [])
        assert False, "Should raise ValueError for empty minpts_range"
    except ValueError:
        pass
    
    # Test invalid metric
    try:
        selector.grid_search(X, [0.5], [3], metric='invalid')
        assert False, "Should raise ValueError for invalid metric"
    except ValueError:
        pass
    
    print("✓ grid_search validation test passed")


def test_suggest_parameters_basic():
    """Test basic parameter suggestion"""
    selector = ParameterSelector()
    
    # Create simple 2D dataset
    np.random.seed(42)
    X = np.random.randn(50, 2)
    
    suggested_eps, suggested_minpts = selector.suggest_parameters(X)
    
    # Check that we get valid parameters
    assert suggested_eps > 0, "Suggested eps should be positive"
    assert suggested_minpts >= 4, "Suggested minpts should be at least 4"
    assert isinstance(suggested_eps, (float, np.floating)), "eps should be float"
    assert isinstance(suggested_minpts, (int, np.integer)), "minpts should be int"
    
    print("✓ suggest_parameters basic test passed")


def test_suggest_parameters_dimensionality():
    """Test that minpts scales with dimensionality"""
    selector = ParameterSelector()
    
    np.random.seed(42)
    
    # 2D data
    X_2d = np.random.randn(50, 2)
    _, minpts_2d = selector.suggest_parameters(X_2d)
    
    # 5D data
    X_5d = np.random.randn(50, 5)
    _, minpts_5d = selector.suggest_parameters(X_5d)
    
    # Higher dimensional data should suggest higher minpts
    assert minpts_5d >= minpts_2d, "Higher dimensions should suggest higher minpts"
    
    print("✓ suggest_parameters dimensionality test passed")


def test_suggest_parameters_validation():
    """Test input validation for suggest_parameters"""
    selector = ParameterSelector()
    
    # Test empty array
    try:
        selector.suggest_parameters(np.array([]).reshape(0, 2))
        assert False, "Should raise ValueError for empty array"
    except ValueError:
        pass
    
    # Test non-numpy array
    try:
        selector.suggest_parameters([[0, 0], [1, 1]])
        assert False, "Should raise TypeError for non-numpy array"
    except TypeError:
        pass
    
    # Test 1D array
    try:
        selector.suggest_parameters(np.array([1, 2, 3]))
        assert False, "Should raise ValueError for 1D array"
    except ValueError:
        pass
    
    print("✓ suggest_parameters validation test passed")


def test_k_distances_ordering():
    """Test that k-distances are properly ordered"""
    selector = ParameterSelector()
    
    # Create data with clear distance structure
    X = np.array([
        [0, 0], [0.1, 0], [0.2, 0],  # tight cluster
        [10, 10], [10.1, 10], [10.2, 10],  # another tight cluster
        [5, 5]  # isolated point
    ])
    
    k_distances = selector.compute_k_distances(X, k=2)
    
    # The isolated point should have the largest k-distance
    # It should be at the beginning of the sorted array
    assert k_distances[0] > k_distances[-1], "Largest distance should be first"
    
    print("✓ k_distances ordering test passed")


if __name__ == "__main__":
    test_compute_k_distances_basic()
    test_compute_k_distances_single_point()
    test_compute_k_distances_validation()
    test_find_elbow_point_basic()
    test_find_elbow_point_validation()
    test_grid_search_basic()
    test_grid_search_validation()
    test_suggest_parameters_basic()
    test_suggest_parameters_dimensionality()
    test_suggest_parameters_validation()
    test_k_distances_ordering()
    test_compute_k_distances_edge_cases()
    test_compute_k_distances_input_validation()
    test_grid_search_input_validation()
    test_suggest_parameters_input_validation()
    test_compute_metric_edge_cases()
    test_suggest_parameters_small_dataset()
    print("\n✓ All parameter tuning tests passed!")


def test_compute_k_distances_edge_cases():
    """Test compute_k_distances with edge cases"""
    selector = ParameterSelector()
    
    # Test with k equal to n_samples (boundary case)
    X = np.array([[0, 0], [1, 1], [2, 2]])  # 3 points
    k_distances = selector.compute_k_distances(X, k=3)  # k = n_samples
    
    # Should handle gracefully
    assert len(k_distances) == 3
    assert np.all(k_distances >= 0)
    
    print("✓ compute_k_distances edge cases test passed")


def test_compute_k_distances_input_validation():
    """Test input validation for compute_k_distances"""
    selector = ParameterSelector()
    
    # Test 1D array
    try:
        selector.compute_k_distances(np.array([1, 2, 3]), k=2)
        assert False, "Should raise ValueError for 1D array"
    except ValueError as e:
        assert "2-dimensional" in str(e)
    
    # Test 3D array
    try:
        selector.compute_k_distances(np.array([[[1, 2]]]), k=2)
        assert False, "Should raise ValueError for 3D array"
    except ValueError as e:
        assert "2-dimensional" in str(e)
    
    print("✓ compute_k_distances input validation test passed")


def test_grid_search_input_validation():
    """Test comprehensive input validation for grid_search"""
    selector = ParameterSelector()
    
    # Test 1D array
    try:
        selector.grid_search(np.array([1, 2, 3]), [0.5], [3])
        assert False, "Should raise ValueError for 1D array"
    except ValueError as e:
        assert "2-dimensional" in str(e)
    
    # Test 3D array
    try:
        selector.grid_search(np.array([[[1, 2]]]), [0.5], [3])
        assert False, "Should raise ValueError for 3D array"
    except ValueError as e:
        assert "2-dimensional" in str(e)
    
    print("✓ grid_search input validation test passed")


def test_suggest_parameters_input_validation():
    """Test comprehensive input validation for suggest_parameters"""
    selector = ParameterSelector()
    
    # Test empty array
    try:
        selector.suggest_parameters(np.array([]).reshape(0, 2))
        assert False, "Should raise ValueError for empty array"
    except ValueError as e:
        assert "empty" in str(e) or "at least" in str(e)
    
    # Test 1D array
    try:
        selector.suggest_parameters(np.array([1, 2, 3]))
        assert False, "Should raise ValueError for 1D array"
    except ValueError as e:
        assert "2-dimensional" in str(e)
    
    # Test 3D array
    try:
        selector.suggest_parameters(np.array([[[1, 2]]]))
        assert False, "Should raise ValueError for 3D array"
    except ValueError as e:
        assert "2-dimensional" in str(e)
    
    print("✓ suggest_parameters input validation test passed")


def test_compute_metric_edge_cases():
    """Test _compute_metric with edge cases"""
    selector = ParameterSelector()
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    
    # Test with all noise labels (-1)
    labels_all_noise = np.array([-1, -1, -1, -1])
    score = selector._compute_metric(X, labels_all_noise, 'silhouette')
    # Should handle gracefully (silhouette undefined for all noise)
    assert score is not None
    
    # Test with single cluster
    labels_single = np.array([0, 0, 0, 0])
    score = selector._compute_metric(X, labels_single, 'silhouette')
    # Should handle gracefully (silhouette undefined for single cluster)
    assert score is not None
    
    print("✓ _compute_metric edge cases test passed")


def test_suggest_parameters_small_dataset():
    """Test suggest_parameters with very small dataset"""
    selector = ParameterSelector()
    
    # Very small dataset
    X = np.array([[0, 0], [1, 1]])
    eps, minpts = selector.suggest_parameters(X)
    
    assert eps > 0
    # For very small datasets, minpts should adapt to dataset size
    assert minpts >= 2  # Should be at least 2 for small datasets
    assert minpts <= len(X)  # Should not exceed dataset size
    
    print("✓ suggest_parameters small dataset test passed")