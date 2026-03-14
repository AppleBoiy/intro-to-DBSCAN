"""
Pytest Configuration and Fixtures

This module provides reusable pytest fixtures for DBSCAN testing.
Fixtures reduce code duplication and ensure consistent test data across
all test files.

**Validates: Requirements 12.1**
"""
import pytest
import numpy as np
from sklearn.datasets import make_moons, make_blobs, make_circles
from src.dbscan_from_scratch import DBSCAN
from src.visualization import DBSCANVisualizer, VisualizationConfig
from src.data_loader import DatasetGenerator


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_data():
    """
    Generate sample moon-shaped dataset for testing.
    
    Returns
    -------
    np.ndarray
        Dataset of shape (100, 2) with two interleaving half circles
    
    Notes
    -----
    This is the most commonly used test dataset. It has:
    - 100 samples
    - 2 features (2D for easy visualization)
    - Two clear clusters with non-convex shapes
    - Low noise level (0.05) for reproducibility
    - Fixed random_state for deterministic results
    
    Examples
    --------
    >>> def test_example(sample_data):
    ...     assert sample_data.shape == (100, 2)
    ...     dbscan = DBSCAN(eps=0.3, min_pts=5)
    ...     labels = dbscan.fit_predict(sample_data)
    """
    X, _ = make_moons(n_samples=100, noise=0.05, random_state=42)
    return X


@pytest.fixture
def sample_data_blobs():
    """
    Generate sample blob-shaped dataset for testing.
    
    Returns
    -------
    np.ndarray
        Dataset of shape (150, 2) with three spherical clusters
    
    Notes
    -----
    This dataset is useful for testing:
    - Convex cluster shapes
    - Well-separated clusters
    - Comparison with K-Means
    
    Examples
    --------
    >>> def test_example(sample_data_blobs):
    ...     assert sample_data_blobs.shape == (150, 2)
    """
    X, _ = make_blobs(n_samples=150, centers=3, random_state=42)
    return X


@pytest.fixture
def sample_data_circles():
    """
    Generate sample concentric circles dataset for testing.
    
    Returns
    -------
    np.ndarray
        Dataset of shape (100, 2) with two concentric circles
    
    Notes
    -----
    This dataset is useful for testing:
    - Non-convex cluster shapes
    - Nested clusters
    - DBSCAN's advantage over K-Means
    
    Examples
    --------
    >>> def test_example(sample_data_circles):
    ...     assert sample_data_circles.shape == (100, 2)
    """
    X, _ = make_circles(n_samples=100, noise=0.05, factor=0.5, random_state=42)
    return X


@pytest.fixture
def sample_data_small():
    """
    Generate small dataset for quick tests.
    
    Returns
    -------
    np.ndarray
        Dataset of shape (20, 2) with two small clusters
    
    Notes
    -----
    Use this for tests that need to run quickly or when you need
    to manually verify results. Small enough to inspect visually.
    
    Examples
    --------
    >>> def test_example(sample_data_small):
    ...     assert sample_data_small.shape == (20, 2)
    """
    X, _ = make_blobs(n_samples=20, centers=2, random_state=42)
    return X


@pytest.fixture
def sample_data_with_noise():
    """
    Generate dataset with clear noise points.
    
    Returns
    -------
    np.ndarray
        Dataset with clusters and isolated noise points
    
    Notes
    -----
    This dataset is specifically designed to test noise detection.
    It contains:
    - A dense cluster of 10 points
    - 3 isolated noise points far from the cluster
    
    Examples
    --------
    >>> def test_noise_detection(sample_data_with_noise):
    ...     dbscan = DBSCAN(eps=1.5, min_pts=3)
    ...     labels = dbscan.fit_predict(sample_data_with_noise)
    ...     # Last 3 points should be noise
    ...     assert labels[-1] == -1
    """
    # Create a dense cluster
    cluster = np.array([
        [0, 0], [0, 1], [1, 0], [1, 1],
        [0.5, 0.5], [0.2, 0.8], [0.8, 0.2],
        [0.3, 0.3], [0.7, 0.7], [0.5, 1.0]
    ])
    
    # Add isolated noise points
    noise = np.array([
        [10, 10],
        [15, 15],
        [-10, -10]
    ])
    
    return np.vstack([cluster, noise])


@pytest.fixture
def empty_dataset():
    """
    Generate empty dataset for edge case testing.
    
    Returns
    -------
    np.ndarray
        Empty array of shape (0, 2)
    
    Notes
    -----
    Use this to test edge case handling for empty inputs.
    
    Examples
    --------
    >>> def test_empty(empty_dataset):
    ...     dbscan = DBSCAN(eps=0.5, min_pts=5)
    ...     labels = dbscan.fit_predict(empty_dataset)
    ...     assert len(labels) == 0
    """
    return np.array([]).reshape(0, 2)


@pytest.fixture
def single_point_dataset():
    """
    Generate single-point dataset for edge case testing.
    
    Returns
    -------
    np.ndarray
        Dataset with single point at origin
    
    Notes
    -----
    Use this to test edge case handling for single-point inputs.
    A single point should always be classified as noise.
    
    Examples
    --------
    >>> def test_single_point(single_point_dataset):
    ...     dbscan = DBSCAN(eps=0.5, min_pts=2)
    ...     labels = dbscan.fit_predict(single_point_dataset)
    ...     assert labels[0] == -1  # Should be noise
    """
    return np.array([[0, 0]])


# ============================================================================
# DBSCAN Instance Fixtures
# ============================================================================

@pytest.fixture
def dbscan_instance():
    """
    Create DBSCAN instance with default parameters.
    
    Returns
    -------
    DBSCAN
        DBSCAN instance with eps=0.5, min_pts=5
    
    Notes
    -----
    This provides a standard DBSCAN instance for testing.
    Parameters are chosen to work well with most test datasets.
    
    Examples
    --------
    >>> def test_example(dbscan_instance, sample_data):
    ...     labels = dbscan_instance.fit_predict(sample_data)
    ...     assert labels is not None
    """
    return DBSCAN(eps=0.5, min_pts=5)


@pytest.fixture
def dbscan_instance_tight():
    """
    Create DBSCAN instance with tight clustering parameters.
    
    Returns
    -------
    DBSCAN
        DBSCAN instance with eps=0.3, min_pts=5
    
    Notes
    -----
    Use this for testing with smaller epsilon values.
    Will find tighter, more conservative clusters.
    
    Examples
    --------
    >>> def test_example(dbscan_instance_tight, sample_data):
    ...     labels = dbscan_instance_tight.fit_predict(sample_data)
    """
    return DBSCAN(eps=0.3, min_pts=5)


@pytest.fixture
def dbscan_instance_loose():
    """
    Create DBSCAN instance with loose clustering parameters.
    
    Returns
    -------
    DBSCAN
        DBSCAN instance with eps=1.0, min_pts=3
    
    Notes
    -----
    Use this for testing with larger epsilon values.
    Will find larger, more inclusive clusters.
    
    Examples
    --------
    >>> def test_example(dbscan_instance_loose, sample_data):
    ...     labels = dbscan_instance_loose.fit_predict(sample_data)
    """
    return DBSCAN(eps=1.0, min_pts=3)


@pytest.fixture
def dbscan_manhattan():
    """
    Create DBSCAN instance with Manhattan distance metric.
    
    Returns
    -------
    DBSCAN
        DBSCAN instance with Manhattan (L1) distance
    
    Notes
    -----
    Use this for testing alternative distance metrics.
    
    Examples
    --------
    >>> def test_manhattan(dbscan_manhattan, sample_data):
    ...     labels = dbscan_manhattan.fit_predict(sample_data)
    """
    return DBSCAN(eps=0.5, min_pts=5, metric='manhattan')


@pytest.fixture
def dbscan_chebyshev():
    """
    Create DBSCAN instance with Chebyshev distance metric.
    
    Returns
    -------
    DBSCAN
        DBSCAN instance with Chebyshev (L∞) distance
    
    Notes
    -----
    Use this for testing alternative distance metrics.
    
    Examples
    --------
    >>> def test_chebyshev(dbscan_chebyshev, sample_data):
    ...     labels = dbscan_chebyshev.fit_predict(sample_data)
    """
    return DBSCAN(eps=0.5, min_pts=5, metric='chebyshev')


# ============================================================================
# Visualization Fixtures
# ============================================================================

@pytest.fixture
def visualizer():
    """
    Create DBSCANVisualizer instance with default configuration.
    
    Returns
    -------
    DBSCANVisualizer
        Visualizer with default styling
    
    Notes
    -----
    This provides a standard visualizer for testing visualization functions.
    Uses default configuration suitable for most tests.
    
    Examples
    --------
    >>> def test_plot(visualizer, sample_data):
    ...     labels = np.zeros(len(sample_data))
    ...     visualizer.plot_clusters(sample_data, labels, "Test")
    """
    return DBSCANVisualizer()


@pytest.fixture
def visualizer_custom():
    """
    Create DBSCANVisualizer instance with custom configuration.
    
    Returns
    -------
    DBSCANVisualizer
        Visualizer with custom styling
    
    Notes
    -----
    Use this for testing custom visualization configurations.
    
    Examples
    --------
    >>> def test_custom_plot(visualizer_custom, sample_data):
    ...     labels = np.zeros(len(sample_data))
    ...     visualizer_custom.plot_clusters(sample_data, labels, "Test")
    """
    custom_config = VisualizationConfig(
        figsize=(8, 6),
        core_color='red',
        noise_marker='+'
    )
    return DBSCANVisualizer(config=custom_config)


# ============================================================================
# Data Generator Fixtures
# ============================================================================

@pytest.fixture
def data_generator():
    """
    Create DatasetGenerator instance.
    
    Returns
    -------
    DatasetGenerator
        Generator for creating various test datasets
    
    Notes
    -----
    Use this for generating custom datasets in tests.
    
    Examples
    --------
    >>> def test_generator(data_generator):
    ...     X = data_generator.generate_basic_shapes('moons', 100, 0.05)
    ...     assert X.shape == (100, 2)
    """
    return DatasetGenerator()


# ============================================================================
# Fitted DBSCAN Fixtures (for testing post-fit operations)
# ============================================================================

@pytest.fixture
def fitted_dbscan(sample_data):
    """
    Create fitted DBSCAN instance with sample data.
    
    Parameters
    ----------
    sample_data : np.ndarray
        Sample dataset fixture
    
    Returns
    -------
    tuple
        (DBSCAN instance, labels, sample_data)
    
    Notes
    -----
    Use this when you need a DBSCAN instance that has already been fitted.
    Useful for testing post-fit operations like get_core_points().
    
    Examples
    --------
    >>> def test_core_points(fitted_dbscan):
    ...     dbscan, labels, X = fitted_dbscan
    ...     core_points = dbscan.get_core_points()
    ...     assert len(core_points) > 0
    """
    dbscan = DBSCAN(eps=0.3, min_pts=5)
    labels = dbscan.fit_predict(sample_data)
    return dbscan, labels, sample_data


@pytest.fixture
def fitted_dbscan_blobs(sample_data_blobs):
    """
    Create fitted DBSCAN instance with blob data.
    
    Parameters
    ----------
    sample_data_blobs : np.ndarray
        Blob dataset fixture
    
    Returns
    -------
    tuple
        (DBSCAN instance, labels, sample_data_blobs)
    
    Notes
    -----
    Use this for testing with well-separated spherical clusters.
    
    Examples
    --------
    >>> def test_blobs(fitted_dbscan_blobs):
    ...     dbscan, labels, X = fitted_dbscan_blobs
    ...     n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    ...     assert n_clusters == 3
    """
    dbscan = DBSCAN(eps=0.8, min_pts=5)
    labels = dbscan.fit_predict(sample_data_blobs)
    return dbscan, labels, sample_data_blobs


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """
    Configure pytest with custom markers and settings.
    
    This function is called by pytest during initialization.
    It registers custom markers for organizing tests.
    """
    config.addinivalue_line(
        "markers", "property: mark test as a property-based test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "visualization: mark test as visualization test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


# ============================================================================
# Pytest Collection Hooks
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers automatically.
    
    This function is called after test collection and can modify
    the collected test items. We use it to automatically add markers
    based on test names and locations.
    """
    for item in items:
        # Add 'property' marker to property-based tests
        if "test_properties" in item.nodeid:
            item.add_marker(pytest.mark.property)
        
        # Add 'visualization' marker to visualization tests
        if "test_visualization" in item.nodeid:
            item.add_marker(pytest.mark.visualization)
        
        # Add 'slow' marker to performance tests
        if "test_performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
