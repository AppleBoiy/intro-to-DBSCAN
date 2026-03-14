# Performance Testing Guide

This document explains how to run and interpret the performance tests for the DBSCAN implementation.

## Overview

The performance test suite (`test_performance.py`) validates the computational complexity and scalability characteristics of the DBSCAN implementation. It includes:

1. **Scalability tests** - Measure runtime with varying dataset sizes
2. **Complexity verification** - Empirically verify O(n²) time complexity
3. **Memory usage tests** - Estimate and document memory requirements
4. **Parameter sensitivity** - Test performance with different parameters
5. **Edge case performance** - Test worst-case and best-case scenarios

## Running Performance Tests

### Run All Performance Tests

```bash
pytest tests/test_performance.py -v -s -m slow
```

**Note**: This will take approximately 10-15 minutes to complete all tests.

### Run Specific Tests

```bash
# Test scalability only
pytest tests/test_performance.py::test_scalability_with_dataset_size -v -s

# Test complexity verification
pytest tests/test_performance.py::test_quadratic_complexity_empirically -v -s

# Generate performance report
pytest tests/test_performance.py::test_generate_performance_report -v -s
```

### Quick Performance Check

```bash
# Run the test file directly for a quick check
python tests/test_performance.py
```

## Test Categories

### 1. Scalability Tests

Tests DBSCAN performance with dataset sizes: 100, 500, 1000, 5000 points.

**Expected Results**:
- 100 points: < 0.1s
- 500 points: < 2s
- 1000 points: < 10s
- 5000 points: < 300s (5 minutes)

### 2. Complexity Verification

Verifies O(n²) complexity by measuring runtime ratios when dataset size doubles.

**Expected Results**:
- When size doubles (e.g., 100 → 200), runtime should increase by ~4x (2²)
- Observed ratios typically range from 3.5x to 4.5x

### 3. Memory Usage Tests

Estimates memory requirements for different dataset sizes.

**Expected Memory Usage**:
- Data storage: O(n × d) where d is dimensionality
- Labels array: O(n)
- Neighbor lists: O(n × k) where k is average neighbors
- Worst case: O(n²) for very dense datasets

### 4. Dimensionality Impact

Tests how increasing dimensionality affects runtime.

**Expected Results**:
- Distance computation is O(d), so higher dimensions may take slightly longer
- However, the O(n²) term dominates, so the effect is often minimal

### 5. Parameter Sensitivity

Tests performance with different eps and min_pts values.

**Observations**:
- Larger eps values may increase runtime (more neighbors to process)
- Different parameters can significantly affect the number of clusters found

### 6. Distance Metrics

Compares performance of different distance metrics (Euclidean, Manhattan, Chebyshev).

**Expected Results**:
- All metrics should have similar performance
- Euclidean may be slightly slower due to square root computation

## Performance Report

The `test_generate_performance_report` test generates a comprehensive report including:

- Scalability metrics for all dataset sizes
- Complexity verification results
- Dimensionality impact analysis
- Summary of algorithm characteristics

Example output:

```
======================================================================
                    DBSCAN PERFORMANCE REPORT
======================================================================

1. SCALABILITY TEST
----------------------------------------------------------------------
  Size:   100 | Runtime:   0.0686s | Clusters:   2 | Noise:   13
  Size:   500 | Runtime:   1.7371s | Clusters:   1 | Noise:   14
  Size:  1000 | Runtime:   6.8199s | Clusters:   1 | Noise:   14
  Size:  5000 | Runtime: 171.0392s | Clusters:   1 | Noise:   11

2. COMPLEXITY VERIFICATION (O(n²))
----------------------------------------------------------------------
  Size 100: 0.0690s
  Size 200: 0.2755s (ratio: 3.99x)
  Size 400: 1.0856s (ratio: 3.94x)
  Expected ratio for O(n²): ~4.0x
  Observed average: 3.97x

3. DIMENSIONALITY IMPACT
----------------------------------------------------------------------
  Dimensions:  2 | Runtime:   1.7025s
  Dimensions:  5 | Runtime:   1.6787s
  Dimensions: 10 | Runtime:   1.7005s

======================================================================
SUMMARY
======================================================================
  Algorithm: DBSCAN (naive implementation)
  Time Complexity: O(n²)
  Space Complexity: O(n)
  Parameters: eps=0.5, min_pts=5
  Test completed successfully
======================================================================
```

## Memory Profiling

For detailed memory profiling, install `memory_profiler`:

```bash
pip install memory-profiler
```

Then create a test script:

```python
from memory_profiler import profile
from src.dbscan_from_scratch import DBSCAN
import numpy as np

@profile
def test_memory():
    X = np.random.randn(1000, 2)
    dbscan = DBSCAN(eps=0.5, min_pts=5)
    labels = dbscan.fit_predict(X)

if __name__ == "__main__":
    test_memory()
```

Run with:

```bash
python -m memory_profiler test_script.py
```

## Performance Optimization

The current implementation uses a naive O(n²) approach. For better performance on large datasets:

1. **Use spatial indexing** (KD-tree, R-tree) - reduces complexity to O(n log n)
2. **Use scikit-learn's implementation** - highly optimized with spatial indexing
3. **Reduce dataset size** - sample or filter data before clustering
4. **Adjust parameters** - smaller eps values reduce neighbor counts

See `docs/08_performance_optimization.md` for detailed optimization strategies.

## Interpreting Results

### Good Performance Indicators

- Runtime ratios close to 4.0x when size doubles (confirms O(n²))
- Consistent runtime across multiple runs (low variance)
- Memory usage scales linearly with dataset size

### Performance Issues

- Runtime ratios significantly different from 4.0x (may indicate implementation issues)
- High runtime variance (system load or non-deterministic behavior)
- Excessive memory usage (memory leaks or inefficient data structures)

## Continuous Integration

Performance tests are marked with `@pytest.mark.slow` to allow selective execution:

```bash
# Run all tests except slow ones (for CI)
pytest tests/ -v -m "not slow"

# Run only slow tests (for nightly builds)
pytest tests/ -v -m slow
```

## Troubleshooting

### Tests Taking Too Long

- Run individual tests instead of the full suite
- Use smaller dataset sizes for quick checks
- Skip the 5000-point tests if time is limited

### Memory Errors

- Reduce dataset sizes in the test configuration
- Close other applications to free memory
- Use a machine with more RAM for large-scale tests

### Inconsistent Results

- Ensure no other processes are consuming CPU
- Run tests multiple times and average results
- Check for thermal throttling on laptops

## Related Documentation

- `docs/05_complexity_analysis.md` - Theoretical complexity analysis
- `docs/08_performance_optimization.md` - Optimization techniques
- `notebooks/10_performance_analysis.ipynb` - Interactive performance analysis

## Requirements Validation

These tests validate:

- **Requirement 12.6**: Performance tests documenting runtime complexity
- **Requirement 15.1**: Explanation of naive O(n²) complexity

All tests include proper documentation and validate expected complexity characteristics.
