"""
Unit Tests for DBSCAN Implementation

**Validates: Requirements 12.1, 12.5**
"""
import numpy as np
import pytest
from src.dbscan_from_scratch import DBSCAN, PointType


def test_dbscan_basic(sample_data_blobs):
    """Test basic DBSCAN functionality"""
    dbscan = DBSCAN(eps=0.8, min_pts=5)
    labels = dbscan.fit_predict(sample_data_blobs)
    
    # Check that labels are generated
    assert labels is not None
    assert len(labels) == len(sample_data_blobs)
    
    # Check that at least 1 cluster exists
    unique_labels = set(labels)
    assert len(unique_labels) > 1


def test_euclidean_distance():
    """
    Test Euclidean distance calculation accuracy.
    
    **Validates: Requirements 12.1, 12.5**
    
    Tests that the distance metric correctly computes Euclidean distance
    using the classic 3-4-5 right triangle.
    """
    dbscan = DBSCAN(metric='euclidean')
    p1 = np.array([0, 0])
    p2 = np.array([3, 4])
    
    distance = dbscan._compute_distance(p1, p2)
    assert np.isclose(distance, 5.0), f"Expected 5.0, got {distance}"
    
    # Test with different points
    p3 = np.array([1, 1])
    p4 = np.array([4, 5])
    distance2 = dbscan._compute_distance(p3, p4)
    expected = np.sqrt((4-1)**2 + (5-1)**2)
    assert np.isclose(distance2, expected)


def test_region_query():
    """
    Test epsilon-neighborhood computation (region query).
    
    **Validates: Requirements 12.1, 12.5**
    
    Tests that _get_neighbors correctly identifies all points within
    epsilon distance of a query point [Paper §4.1].
    """
    X = np.array([
        [0, 0],   # Point 0
        [0, 1],   # Point 1 - within eps of 0
        [0, 2],   # Point 2 - within eps of 1
        [10, 10]  # Point 3 - far away (noise)
    ])
    
    dbscan = DBSCAN(eps=1.5, min_pts=2)
    
    # Test neighborhood of point 0
    neighbors_0 = dbscan._get_neighbors(X, 0)
    assert 0 in neighbors_0, "Point should be in its own neighborhood"
    assert 1 in neighbors_0, "Point 1 should be neighbor of point 0"
    assert 2 not in neighbors_0, "Point 2 should not be neighbor of point 0"
    assert 3 not in neighbors_0, "Point 3 should not be neighbor of point 0"
    
    # Test neighborhood of point 1
    neighbors_1 = dbscan._get_neighbors(X, 1)
    assert 0 in neighbors_1, "Point 0 should be neighbor of point 1"
    assert 1 in neighbors_1, "Point should be in its own neighborhood"
    assert 2 in neighbors_1, "Point 2 should be neighbor of point 1"
    assert 3 not in neighbors_1, "Point 3 should not be neighbor of point 1"


def test_core_point_identification():
    """
    Test core point identification.
    
    **Validates: Requirements 12.1, 12.5**
    
    Tests that points with at least min_pts neighbors are correctly
    identified as core points [Paper §4.1].
    """
    # Create a dense cluster and an isolated point
    X = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1],  # Dense cluster (4 points)
        [0.5, 0.5],                       # Center point (5th point in cluster)
        [10, 10]                          # Isolated noise point
    ])
    
    dbscan = DBSCAN(eps=1.5, min_pts=3)
    labels = dbscan.fit_predict(X)
    core_points = dbscan.get_core_points()
    
    # Should have core points in the cluster
    assert len(core_points) > 0, "Should identify core points in dense cluster"
    
    # Noise point should not be core
    assert 5 not in core_points, "Isolated point should not be core"
    
    # Verify core points have enough neighbors
    for core_idx in core_points:
        neighbors = dbscan._get_neighbors(X, core_idx)
        assert len(neighbors) >= dbscan.min_pts, \
            f"Core point {core_idx} should have at least {dbscan.min_pts} neighbors"


def test_cluster_expansion():
    """
    Test cluster expansion logic.
    
    **Validates: Requirements 12.1, 12.5**
    
    Tests that clusters correctly expand to include all density-reachable
    points [Paper Algorithm, p.228].
    """
    # Create two separate clusters
    cluster1 = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]
    ])
    cluster2 = np.array([
        [10, 10], [10, 11], [11, 10], [11, 11], [10.5, 10.5]
    ])
    X = np.vstack([cluster1, cluster2])
    
    dbscan = DBSCAN(eps=1.5, min_pts=3)
    labels = dbscan.fit_predict(X)
    
    # Should find exactly 2 clusters
    unique_clusters = set(labels) - {-1}  # Exclude noise label
    assert len(unique_clusters) == 2, f"Expected 2 clusters, found {len(unique_clusters)}"
    
    # All points in cluster1 should have same label
    cluster1_labels = labels[:5]
    assert len(set(cluster1_labels)) == 1, "All points in cluster1 should have same label"
    assert cluster1_labels[0] != -1, "Cluster1 points should not be noise"
    
    # All points in cluster2 should have same label
    cluster2_labels = labels[5:]
    assert len(set(cluster2_labels)) == 1, "All points in cluster2 should have same label"
    assert cluster2_labels[0] != -1, "Cluster2 points should not be noise"
    
    # Clusters should have different labels
    assert cluster1_labels[0] != cluster2_labels[0], "Clusters should have different labels"


def test_noise_detection():
    """
    Test noise point detection.
    
    **Validates: Requirements 12.1, 12.5**
    
    Tests that isolated points with insufficient neighbors are correctly
    classified as noise [Paper §4.1].
    """
    # Create data with clear noise points
    X = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1],  # Dense cluster
        [10, 10],                         # Isolated noise point
        [15, 15],                         # Another isolated noise point
        [-10, -10]                        # Third isolated noise point
    ])
    
    dbscan = DBSCAN(eps=1.5, min_pts=3)
    labels = dbscan.fit_predict(X)
    
    # Last three points should be noise (-1)
    assert labels[4] == -1, "Point at [10, 10] should be noise"
    assert labels[5] == -1, "Point at [15, 15] should be noise"
    assert labels[6] == -1, "Point at [-10, -10] should be noise"
    
    # First four points should form a cluster (not noise)
    assert labels[0] != -1, "Point at [0, 0] should not be noise"
    assert labels[1] != -1, "Point at [0, 1] should not be noise"
    assert labels[2] != -1, "Point at [1, 0] should not be noise"
    assert labels[3] != -1, "Point at [1, 1] should not be noise"
    
    # Verify noise points have correct type
    assert dbscan.get_point_type(4) == PointType.NOISE
    assert dbscan.get_point_type(5) == PointType.NOISE
    assert dbscan.get_point_type(6) == PointType.NOISE


def test_point_type_enum():
    """Test PointType enum"""
    assert PointType.UNVISITED.value == 0
    assert PointType.NOISE.value == -1
    assert PointType.CORE.value == 1
    assert PointType.BORDER.value == 2


def test_get_core_points():
    """Test get_core_points method"""
    X = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1],  # cluster
        [10, 10]  # noise point
    ])
    
    dbscan = DBSCAN(eps=1.5, min_pts=3)
    labels = dbscan.fit_predict(X)
    core_points = dbscan.get_core_points()
    
    # Should have at least one core point
    assert len(core_points) > 0, "Should have at least one core point"
    # Core points should not include the noise point
    assert 4 not in core_points, "Noise point should not be a core point"


def test_get_point_type():
    """Test get_point_type method"""
    X = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1],  # cluster
        [10, 10]  # noise point
    ])
    
    dbscan = DBSCAN(eps=1.5, min_pts=3)
    labels = dbscan.fit_predict(X)
    
    # Last point should be noise
    assert dbscan.get_point_type(4) == PointType.NOISE, "Last point should be noise"
    
    # At least one point should be core
    core_found = False
    for i in range(4):
        if dbscan.get_point_type(i) == PointType.CORE:
            core_found = True
            break
    assert core_found, "Should have at least one core point"


def test_manhattan_distance():
    """Test Manhattan distance metric"""
    dbscan = DBSCAN(metric='manhattan')
    p1 = np.array([0, 0])
    p2 = np.array([3, 4])
    
    distance = dbscan._compute_distance(p1, p2)
    assert np.isclose(distance, 7.0), f"Expected 7.0, got {distance}"


def test_chebyshev_distance():
    """Test Chebyshev distance metric"""
    dbscan = DBSCAN(metric='chebyshev')
    p1 = np.array([0, 0])
    p2 = np.array([3, 4])
    
    distance = dbscan._compute_distance(p1, p2)
    assert np.isclose(distance, 4.0), f"Expected 4.0, got {distance}"


def test_empty_dataset(empty_dataset):
    """Test handling of empty dataset"""
    dbscan = DBSCAN(eps=0.5, min_pts=5)
    labels = dbscan.fit_predict(empty_dataset)
    
    assert len(labels) == 0, "Empty dataset should return empty labels"
    assert len(dbscan.get_core_points()) == 0, "Empty dataset should have no core points"


def test_single_point(single_point_dataset):
    """Test handling of single point"""
    dbscan = DBSCAN(eps=0.5, min_pts=2)
    labels = dbscan.fit_predict(single_point_dataset)
    
    assert labels[0] == -1, "Single point should be noise"
    assert len(dbscan.get_core_points()) == 0, "Single point should not be core"


def test_all_noise():
    """Test dataset where all points are noise"""
    X = np.array([[0, 0], [10, 10], [20, 20], [30, 30]])
    dbscan = DBSCAN(eps=1.0, min_pts=3)
    labels = dbscan.fit_predict(X)
    
    assert np.all(labels == -1), "All points should be noise"
    assert len(dbscan.get_core_points()) == 0, "Should have no core points"


def test_parameter_validation():
    """Test parameter validation"""
    # Test invalid eps
    try:
        dbscan = DBSCAN(eps=-1.0)
        assert False, "Should raise ValueError for negative eps"
    except ValueError:
        pass
    
    # Test invalid min_pts
    try:
        dbscan = DBSCAN(min_pts=0)
        assert False, "Should raise ValueError for min_pts < 1"
    except ValueError:
        pass
    
    # Test invalid metric
    try:
        dbscan = DBSCAN(metric='invalid')
        assert False, "Should raise ValueError for invalid metric"
    except ValueError:
        pass


def test_input_validation():
    """Test input validation in fit_predict"""
    dbscan = DBSCAN(eps=0.5, min_pts=5)
    
    # Test non-numpy array
    try:
        dbscan.fit_predict([[1, 2], [3, 4]])
        assert False, "Should raise TypeError for non-numpy array"
    except TypeError:
        pass
    
    # Test wrong dimensionality
    try:
        dbscan.fit_predict(np.array([1, 2, 3]))
        assert False, "Should raise ValueError for 1D array"
    except ValueError:
        pass


def test_core_sample_indices_attribute():
    """Test core_sample_indices_ attribute"""
    X = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1],  # cluster
        [10, 10]  # noise
    ])
    
    dbscan = DBSCAN(eps=1.5, min_pts=3)
    labels = dbscan.fit_predict(X)
    
    assert dbscan.core_sample_indices_ is not None, "core_sample_indices_ should be set"
    assert isinstance(dbscan.core_sample_indices_, np.ndarray), "core_sample_indices_ should be numpy array"


def test_components_attribute():
    """Test components_ attribute"""
    X = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1],  # cluster
        [10, 10]  # noise
    ])
    
    dbscan = DBSCAN(eps=1.5, min_pts=3)
    labels = dbscan.fit_predict(X)
    
    assert dbscan.components_ is not None, "components_ should be set"
    assert isinstance(dbscan.components_, np.ndarray), "components_ should be numpy array"
    assert dbscan.components_.shape[1] == X.shape[1], "components_ should have same features as X"
