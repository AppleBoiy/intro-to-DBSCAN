# Test Fixtures Guide

This document explains the pytest fixtures available in `conftest.py` for testing the DBSCAN implementation.

## Overview

Fixtures are reusable test components that provide consistent test data and instances across all test files. They reduce code duplication and ensure reproducibility.

## Available Fixtures

### Sample Data Fixtures

#### `sample_data`
Standard moon-shaped dataset (100 samples, 2 features).
```python
def test_example(sample_data):
    assert sample_data.shape == (100, 2)
    dbscan = DBSCAN(eps=0.3, min_pts=5)
    labels = dbscan.fit_predict(sample_data)
```

#### `sample_data_blobs`
Blob-shaped dataset with 3 spherical clusters (150 samples, 2 features).
```python
def test_example(sample_data_blobs):
    dbscan = DBSCAN(eps=0.8, min_pts=5)
    labels = dbscan.fit_predict(sample_data_blobs)
```

#### `sample_data_circles`
Concentric circles dataset (100 samples, 2 features).
```python
def test_example(sample_data_circles):
    dbscan = DBSCAN(eps=0.3, min_pts=5)
    labels = dbscan.fit_predict(sample_data_circles)
```

#### `sample_data_small`
Small dataset for quick tests (20 samples, 2 features).
```python
def test_example(sample_data_small):
    # Fast test with small dataset
    dbscan = DBSCAN(eps=0.8, min_pts=3)
    labels = dbscan.fit_predict(sample_data_small)
```

#### `sample_data_with_noise`
Dataset with clear noise points (13 samples: 10 cluster + 3 noise).
```python
def test_noise(sample_data_with_noise):
    dbscan = DBSCAN(eps=1.5, min_pts=3)
    labels = dbscan.fit_predict(sample_data_with_noise)
    # Last 3 points should be noise
    assert labels[-1] == -1
```

#### `empty_dataset`
Empty dataset for edge case testing (0 samples).
```python
def test_empty(empty_dataset):
    dbscan = DBSCAN(eps=0.5, min_pts=5)
    labels = dbscan.fit_predict(empty_dataset)
    assert len(labels) == 0
```

#### `single_point_dataset`
Single point dataset for edge case testing (1 sample).
```python
def test_single(single_point_dataset):
    dbscan = DBSCAN(eps=0.5, min_pts=2)
    labels = dbscan.fit_predict(single_point_dataset)
    assert labels[0] == -1  # Should be noise
```

### DBSCAN Instance Fixtures

#### `dbscan_instance`
Standard DBSCAN instance (eps=0.5, min_pts=5, metric='euclidean').
```python
def test_example(dbscan_instance, sample_data):
    labels = dbscan_instance.fit_predict(sample_data)
```

#### `dbscan_instance_tight`
Tight clustering parameters (eps=0.3, min_pts=5).
```python
def test_tight(dbscan_instance_tight, sample_data):
    labels = dbscan_instance_tight.fit_predict(sample_data)
```

#### `dbscan_instance_loose`
Loose clustering parameters (eps=1.0, min_pts=3).
```python
def test_loose(dbscan_instance_loose, sample_data):
    labels = dbscan_instance_loose.fit_predict(sample_data)
```

#### `dbscan_manhattan`
DBSCAN with Manhattan distance metric.
```python
def test_manhattan(dbscan_manhattan, sample_data):
    labels = dbscan_manhattan.fit_predict(sample_data)
```

#### `dbscan_chebyshev`
DBSCAN with Chebyshev distance metric.
```python
def test_chebyshev(dbscan_chebyshev, sample_data):
    labels = dbscan_chebyshev.fit_predict(sample_data)
```

### Visualization Fixtures

#### `visualizer`
Standard DBSCANVisualizer instance with default configuration.
```python
def test_plot(visualizer, sample_data):
    labels = np.zeros(len(sample_data))
    visualizer.plot_clusters(sample_data, labels, "Test")
```

#### `visualizer_custom`
DBSCANVisualizer with custom configuration (red core points, '+' noise marker).
```python
def test_custom(visualizer_custom, sample_data):
    labels = np.zeros(len(sample_data))
    visualizer_custom.plot_clusters(sample_data, labels, "Test")
```

### Data Generator Fixture

#### `data_generator`
DatasetGenerator instance for creating custom datasets.
```python
def test_generator(data_generator):
    X = data_generator.generate_basic_shapes('moons', 100, 0.05)
    assert X.shape == (100, 2)
```

### Fitted DBSCAN Fixtures

#### `fitted_dbscan`
Pre-fitted DBSCAN instance with sample_data.
Returns tuple: (dbscan, labels, X)
```python
def test_core_points(fitted_dbscan):
    dbscan, labels, X = fitted_dbscan
    core_points = dbscan.get_core_points()
    assert len(core_points) > 0
```

#### `fitted_dbscan_blobs`
Pre-fitted DBSCAN instance with blob data.
Returns tuple: (dbscan, labels, X)
```python
def test_blobs(fitted_dbscan_blobs):
    dbscan, labels, X = fitted_dbscan_blobs
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    assert n_clusters == 3
```

## Pytest Markers

The conftest.py also configures custom markers for organizing tests:

- `@pytest.mark.property` - Property-based tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.visualization` - Visualization tests
- `@pytest.mark.integration` - Integration tests

### Using Markers

```python
@pytest.mark.property
def test_determinism(sample_data):
    """Property-based test for determinism"""
    pass

@pytest.mark.slow
def test_large_dataset():
    """Test with large dataset (slow)"""
    pass

@pytest.mark.visualization
def test_plot(visualizer, sample_data):
    """Test visualization function"""
    pass
```

### Running Tests by Marker

```bash
# Run only property-based tests
pytest -m property

# Run all except slow tests
pytest -m "not slow"

# Run visualization tests
pytest -m visualization
```

## Best Practices

1. **Use fixtures instead of creating data in tests**
   ```python
   # Good
   def test_example(sample_data):
       dbscan = DBSCAN(eps=0.5, min_pts=5)
       labels = dbscan.fit_predict(sample_data)
   
   # Avoid
   def test_example():
       X, _ = make_moons(n_samples=100, noise=0.05, random_state=42)
       dbscan = DBSCAN(eps=0.5, min_pts=5)
       labels = dbscan.fit_predict(X)
   ```

2. **Combine fixtures for complex scenarios**
   ```python
   def test_multiple_metrics(sample_data, dbscan_instance, 
                            dbscan_manhattan, dbscan_chebyshev):
       labels_euclidean = dbscan_instance.fit_predict(sample_data)
       labels_manhattan = dbscan_manhattan.fit_predict(sample_data)
       labels_chebyshev = dbscan_chebyshev.fit_predict(sample_data)
   ```

3. **Use fitted fixtures for post-fit operations**
   ```python
   def test_point_types(fitted_dbscan):
       dbscan, labels, X = fitted_dbscan
       # No need to call fit_predict again
       core_points = dbscan.get_core_points()
   ```

4. **Choose appropriate dataset size**
   - Use `sample_data_small` for quick unit tests
   - Use `sample_data` for standard tests
   - Use `sample_data_blobs` for larger tests

5. **Test edge cases with dedicated fixtures**
   ```python
   def test_edge_cases(empty_dataset, single_point_dataset):
       dbscan = DBSCAN(eps=0.5, min_pts=5)
       
       # Empty dataset
       labels_empty = dbscan.fit_predict(empty_dataset)
       assert len(labels_empty) == 0
       
       # Single point
       labels_single = dbscan.fit_predict(single_point_dataset)
       assert labels_single[0] == -1
   ```

## Adding New Fixtures

To add new fixtures, edit `tests/conftest.py`:

```python
@pytest.fixture
def my_custom_fixture():
    """
    Description of what this fixture provides.
    
    Returns
    -------
    type
        Description of return value
    """
    # Setup code
    data = create_data()
    
    # Return fixture value
    return data
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_dbscan.py

# Run specific test
pytest tests/test_dbscan.py::test_euclidean_distance

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## References

- [Pytest Fixtures Documentation](https://docs.pytest.org/en/stable/fixture.html)
- [Pytest Markers Documentation](https://docs.pytest.org/en/stable/mark.html)
- Design Document: `.kiro/specs/comprehensive-dbscan-learning-repository/design.md`
- Requirements: `.kiro/specs/comprehensive-dbscan-learning-repository/requirements.md`
