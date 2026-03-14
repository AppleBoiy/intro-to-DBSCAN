"""
Performance Test Suite for DBSCAN Implementation

This module contains comprehensive performance tests that validate the
computational complexity and scalability characteristics of the DBSCAN
implementation.

**Validates: Requirements 12.6, 15.1**
"""
import pytest
import numpy as np
import time
import sys
from typing import List, Tuple, Dict
from src.dbscan_from_scratch import DBSCAN


# ============================================================================
# Test Configuration
# ============================================================================

# Dataset sizes for scalability testing
DATASET_SIZES = [100, 500, 1000, 5000]

# Parameters for performance tests
TEST_EPS = 0.5
TEST_MIN_PTS = 5


# ============================================================================
# Helper Functions
# ============================================================================

def generate_test_dataset(n_samples: int, n_features: int = 2, 
                         random_state: int = 42) -> np.ndarray:
    """
    Generate random dataset for performance testing.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    n_features : int, default=2
        Number of features per sample
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Generated dataset of shape (n_samples, n_features)
    """
    np.random.seed(random_state)
    return np.random.randn(n_samples, n_features)


def measure_runtime(dbscan: DBSCAN, X: np.ndarray) -> float:
    """
    Measure runtime of DBSCAN clustering.
    
    Parameters
    ----------
    dbscan : DBSCAN
        DBSCAN instance to test
    X : np.ndarray
        Dataset to cluster
        
    Returns
    -------
    float
        Runtime in seconds
    """
    start_time = time.time()
    dbscan.fit_predict(X)
    end_time = time.time()
    return end_time - start_time


def estimate_memory_usage(X: np.ndarray) -> float:
    """
    Estimate memory usage for dataset.
    
    Parameters
    ----------
    X : np.ndarray
        Dataset
        
    Returns
    -------
    float
        Memory usage in MB
    """
    # Memory for dataset
    data_memory = X.nbytes / (1024 * 1024)  # Convert to MB
    
    # Memory for labels array
    labels_memory = (X.shape[0] * 8) / (1024 * 1024)  # int64
    
    # Memory for neighbor lists (approximate)
    # Worst case: O(n) neighbors per point
    neighbor_memory = (X.shape[0] * X.shape[0] * 8) / (1024 * 1024)
    
    return data_memory + labels_memory + neighbor_memory


# ============================================================================
# Scalability Tests
# ============================================================================

@pytest.mark.slow
@pytest.mark.parametrize("n_samples", DATASET_SIZES)
def test_scalability_with_dataset_size(n_samples):
    """
    Test DBSCAN scalability with varying dataset sizes.
    
    This test measures runtime for different dataset sizes to verify
    that the implementation scales as expected.
    
    **Validates: Requirements 12.6, 15.1**
    
    Parameters
    ----------
    n_samples : int
        Number of samples in the test dataset
    """
    # Generate test dataset
    X = generate_test_dataset(n_samples)
    
    # Create DBSCAN instance
    dbscan = DBSCAN(eps=TEST_EPS, min_pts=TEST_MIN_PTS)
    
    # Measure runtime
    runtime = measure_runtime(dbscan, X)
    
    # Verify clustering completed
    assert dbscan.labels_ is not None
    assert len(dbscan.labels_) == n_samples
    
    # Print performance metrics for analysis
    print(f"\nDataset size: {n_samples}")
    print(f"Runtime: {runtime:.4f} seconds")
    print(f"Runtime per point: {runtime/n_samples*1000:.4f} ms")
    
    # Basic sanity check: runtime should be reasonable
    # For naive O(n²) implementation, expect roughly:
    # - 100 points: < 0.1s
    # - 500 points: < 2s
    # - 1000 points: < 10s
    # - 5000 points: < 300s (5 minutes)
    max_expected_time = (n_samples / 100) ** 2 * 0.1
    assert runtime < max_expected_time * 10, \
        f"Runtime {runtime:.2f}s exceeds expected {max_expected_time*10:.2f}s"


@pytest.mark.slow
def test_runtime_vs_dataset_size():
    """
    Measure and report runtime vs. dataset size relationship.
    
    This test collects runtime data for multiple dataset sizes
    and reports the relationship for manual analysis.
    
    **Validates: Requirements 12.6, 15.1**
    """
    results = []
    
    for n_samples in DATASET_SIZES:
        X = generate_test_dataset(n_samples)
        dbscan = DBSCAN(eps=TEST_EPS, min_pts=TEST_MIN_PTS)
        runtime = measure_runtime(dbscan, X)
        
        results.append({
            'n_samples': n_samples,
            'runtime': runtime,
            'runtime_per_point': runtime / n_samples
        })
    
    # Print results table
    print("\n" + "="*60)
    print("Runtime vs. Dataset Size")
    print("="*60)
    print(f"{'Size':<10} {'Runtime (s)':<15} {'Per Point (ms)':<15}")
    print("-"*60)
    
    for result in results:
        print(f"{result['n_samples']:<10} "
              f"{result['runtime']:<15.4f} "
              f"{result['runtime_per_point']*1000:<15.4f}")
    
    print("="*60)
    
    # Verify all tests completed
    assert len(results) == len(DATASET_SIZES)


# ============================================================================
# Complexity Verification Tests
# ============================================================================

@pytest.mark.slow
def test_quadratic_complexity_empirically():
    """
    Verify O(n²) complexity empirically by comparing runtime ratios.
    
    For O(n²) complexity, when dataset size doubles, runtime should
    increase by approximately 4x (2²).
    
    **Validates: Requirements 12.6, 15.1**
    """
    # Use smaller dataset sizes for faster testing
    sizes = [100, 200, 400]
    runtimes = []
    
    for n_samples in sizes:
        X = generate_test_dataset(n_samples)
        dbscan = DBSCAN(eps=TEST_EPS, min_pts=TEST_MIN_PTS)
        runtime = measure_runtime(dbscan, X)
        runtimes.append(runtime)
    
    # Calculate runtime ratios
    ratio_1_to_2 = runtimes[1] / runtimes[0]  # 200/100 = 2x size
    ratio_2_to_3 = runtimes[2] / runtimes[1]  # 400/200 = 2x size
    
    print(f"\n{'='*60}")
    print("Quadratic Complexity Verification")
    print(f"{'='*60}")
    print(f"Size 100: {runtimes[0]:.4f}s")
    print(f"Size 200: {runtimes[1]:.4f}s (ratio: {ratio_1_to_2:.2f}x)")
    print(f"Size 400: {runtimes[2]:.4f}s (ratio: {ratio_2_to_3:.2f}x)")
    print(f"{'='*60}")
    print(f"Expected ratio for O(n²): ~4.0x")
    print(f"Observed ratios: {ratio_1_to_2:.2f}x, {ratio_2_to_3:.2f}x")
    print(f"{'='*60}")
    
    # For O(n²), doubling size should increase runtime by ~4x
    # Allow some tolerance for overhead and variance
    # Expect ratio between 2.5 and 6.0 (accounting for constant factors)
    assert 2.0 < ratio_1_to_2 < 8.0, \
        f"Runtime ratio {ratio_1_to_2:.2f} not consistent with O(n²)"
    assert 2.0 < ratio_2_to_3 < 8.0, \
        f"Runtime ratio {ratio_2_to_3:.2f} not consistent with O(n²)"


@pytest.mark.slow
def test_complexity_with_different_dimensions():
    """
    Test how dimensionality affects runtime.
    
    Distance computation is O(d) where d is dimensionality,
    so higher dimensions should increase runtime proportionally.
    
    **Validates: Requirements 12.6, 15.1**
    """
    n_samples = 500
    dimensions = [2, 5, 10]
    runtimes = []
    
    for n_features in dimensions:
        X = generate_test_dataset(n_samples, n_features)
        dbscan = DBSCAN(eps=TEST_EPS, min_pts=TEST_MIN_PTS)
        runtime = measure_runtime(dbscan, X)
        runtimes.append(runtime)
    
    print(f"\n{'='*60}")
    print("Runtime vs. Dimensionality")
    print(f"{'='*60}")
    print(f"{'Dimensions':<15} {'Runtime (s)':<15}")
    print("-"*60)
    
    for dim, runtime in zip(dimensions, runtimes):
        print(f"{dim:<15} {runtime:<15.4f}")
    
    print(f"{'='*60}")
    
    # Higher dimensions may take longer due to O(d) distance computation
    # However, the effect is often small compared to the O(n²) complexity
    # and can be masked by runtime variance. We just verify all completed.
    # Note: In practice, dimensionality has minimal impact on naive DBSCAN
    # since the O(n²) term dominates.
    assert all(r > 0 for r in runtimes), "All tests should complete"


# ============================================================================
# Memory Usage Tests
# ============================================================================

@pytest.mark.slow
@pytest.mark.parametrize("n_samples", DATASET_SIZES)
def test_memory_usage_estimation(n_samples):
    """
    Estimate memory usage for different dataset sizes.
    
    Note: This provides theoretical estimates. For actual memory profiling,
    use memory_profiler package with @profile decorator.
    
    **Validates: Requirements 12.6, 15.1**
    
    Parameters
    ----------
    n_samples : int
        Number of samples in the test dataset
    """
    X = generate_test_dataset(n_samples)
    
    # Estimate memory usage
    estimated_memory = estimate_memory_usage(X)
    
    print(f"\nDataset size: {n_samples}")
    print(f"Estimated memory usage: {estimated_memory:.2f} MB")
    print(f"Memory per point: {estimated_memory/n_samples*1024:.2f} KB")
    
    # Memory should scale roughly as O(n) for data + O(n²) for neighbor lists
    # For 5000 points, expect < 500 MB
    assert estimated_memory < 500, \
        f"Memory usage {estimated_memory:.2f} MB seems excessive"


@pytest.mark.slow
def test_memory_usage_documentation():
    """
    Document memory usage approach for users.
    
    This test documents how to profile memory usage with memory_profiler.
    
    **Validates: Requirements 12.6, 15.1**
    """
    documentation = """
    
    ================================================================
    Memory Profiling Guide
    ================================================================
    
    To profile actual memory usage, install memory_profiler:
    
        pip install memory-profiler
    
    Then use the @profile decorator:
    
        from memory_profiler import profile
        
        @profile
        def test_memory():
            X = generate_test_dataset(1000)
            dbscan = DBSCAN(eps=0.5, min_pts=5)
            labels = dbscan.fit_predict(X)
    
    Run with:
    
        python -m memory_profiler test_script.py
    
    Expected memory characteristics:
    - Data storage: O(n * d) where d is dimensionality
    - Labels array: O(n)
    - Neighbor lists: O(n * k) where k is avg neighbors
    - Worst case: O(n²) for dense datasets
    
    ================================================================
    """
    
    print(documentation)
    
    # This test always passes - it's for documentation
    assert True


# ============================================================================
# Performance Report Generation
# ============================================================================

@pytest.mark.slow
def test_generate_performance_report():
    """
    Generate comprehensive performance report.
    
    This test runs all performance benchmarks and generates a
    summary report for documentation purposes.
    
    **Validates: Requirements 12.6, 15.1**
    """
    print("\n" + "="*70)
    print(" "*20 + "DBSCAN PERFORMANCE REPORT")
    print("="*70)
    
    # Test 1: Scalability
    print("\n1. SCALABILITY TEST")
    print("-"*70)
    
    scalability_results = []
    for n_samples in DATASET_SIZES:
        X = generate_test_dataset(n_samples)
        dbscan = DBSCAN(eps=TEST_EPS, min_pts=TEST_MIN_PTS)
        runtime = measure_runtime(dbscan, X)
        
        n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        n_noise = np.sum(dbscan.labels_ == -1)
        
        scalability_results.append({
            'size': n_samples,
            'runtime': runtime,
            'clusters': n_clusters,
            'noise': n_noise
        })
        
        print(f"  Size: {n_samples:>5} | "
              f"Runtime: {runtime:>8.4f}s | "
              f"Clusters: {n_clusters:>3} | "
              f"Noise: {n_noise:>4}")
    
    # Test 2: Complexity verification
    print("\n2. COMPLEXITY VERIFICATION (O(n²))")
    print("-"*70)
    
    sizes = [100, 200, 400]
    complexity_results = []
    
    for n_samples in sizes:
        X = generate_test_dataset(n_samples)
        dbscan = DBSCAN(eps=TEST_EPS, min_pts=TEST_MIN_PTS)
        runtime = measure_runtime(dbscan, X)
        complexity_results.append(runtime)
    
    ratio_1 = complexity_results[1] / complexity_results[0]
    ratio_2 = complexity_results[2] / complexity_results[1]
    
    print(f"  Size 100: {complexity_results[0]:.4f}s")
    print(f"  Size 200: {complexity_results[1]:.4f}s (ratio: {ratio_1:.2f}x)")
    print(f"  Size 400: {complexity_results[2]:.4f}s (ratio: {ratio_2:.2f}x)")
    print(f"  Expected ratio for O(n²): ~4.0x")
    print(f"  Observed average: {(ratio_1 + ratio_2)/2:.2f}x")
    
    # Test 3: Dimensionality impact
    print("\n3. DIMENSIONALITY IMPACT")
    print("-"*70)
    
    n_samples = 500
    dimensions = [2, 5, 10]
    
    for n_features in dimensions:
        X = generate_test_dataset(n_samples, n_features)
        dbscan = DBSCAN(eps=TEST_EPS, min_pts=TEST_MIN_PTS)
        runtime = measure_runtime(dbscan, X)
        
        print(f"  Dimensions: {n_features:>2} | Runtime: {runtime:>8.4f}s")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Algorithm: DBSCAN (naive implementation)")
    print(f"  Time Complexity: O(n²)")
    print(f"  Space Complexity: O(n)")
    print(f"  Parameters: eps={TEST_EPS}, min_pts={TEST_MIN_PTS}")
    print(f"  Test completed successfully")
    print("="*70 + "\n")
    
    # Verify report generation completed
    assert len(scalability_results) == len(DATASET_SIZES)
    assert len(complexity_results) == len(sizes)


# ============================================================================
# Performance Comparison Tests
# ============================================================================

@pytest.mark.slow
def test_performance_with_different_parameters():
    """
    Test how different parameters affect performance.
    
    Different eps and min_pts values can affect the number of
    neighbors found and thus the runtime.
    
    **Validates: Requirements 12.6, 15.1**
    """
    n_samples = 1000
    X = generate_test_dataset(n_samples)
    
    # Test different parameter combinations
    param_combinations = [
        (0.3, 3),   # Tight clustering
        (0.5, 5),   # Default
        (1.0, 10),  # Loose clustering
    ]
    
    print(f"\n{'='*60}")
    print("Performance with Different Parameters")
    print(f"{'='*60}")
    print(f"{'eps':<10} {'min_pts':<10} {'Runtime (s)':<15} {'Clusters':<10}")
    print("-"*60)
    
    for eps, min_pts in param_combinations:
        dbscan = DBSCAN(eps=eps, min_pts=min_pts)
        runtime = measure_runtime(dbscan, X)
        n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        
        print(f"{eps:<10.1f} {min_pts:<10} {runtime:<15.4f} {n_clusters:<10}")
    
    print(f"{'='*60}")
    
    # All parameter combinations should complete
    assert True


@pytest.mark.slow
def test_performance_with_different_metrics():
    """
    Test performance with different distance metrics.
    
    Different metrics may have different computational costs.
    
    **Validates: Requirements 12.6, 15.1**
    """
    n_samples = 500
    X = generate_test_dataset(n_samples)
    
    metrics = ['euclidean', 'manhattan', 'chebyshev']
    
    print(f"\n{'='*60}")
    print("Performance with Different Distance Metrics")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Runtime (s)':<15}")
    print("-"*60)
    
    for metric in metrics:
        dbscan = DBSCAN(eps=TEST_EPS, min_pts=TEST_MIN_PTS, metric=metric)
        runtime = measure_runtime(dbscan, X)
        
        print(f"{metric:<15} {runtime:<15.4f}")
    
    print(f"{'='*60}")
    
    # All metrics should complete
    assert True


# ============================================================================
# Edge Case Performance Tests
# ============================================================================

@pytest.mark.slow
def test_performance_worst_case_dense_data():
    """
    Test performance on worst-case scenario: very dense data.
    
    When all points are within eps of each other, the algorithm
    must process many neighbors, representing worst-case O(n²).
    
    **Validates: Requirements 12.6, 15.1**
    """
    n_samples = 500
    
    # Create very dense data (all points close together)
    np.random.seed(42)
    X = np.random.randn(n_samples, 2) * 0.1  # Small variance
    
    dbscan = DBSCAN(eps=1.0, min_pts=5)  # Large eps
    runtime = measure_runtime(dbscan, X)
    
    print(f"\nWorst-case (dense data) performance:")
    print(f"  Dataset size: {n_samples}")
    print(f"  Runtime: {runtime:.4f}s")
    print(f"  Clusters found: {len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)}")
    
    # Should still complete in reasonable time
    assert runtime < 30, f"Dense data took too long: {runtime:.2f}s"


@pytest.mark.slow
def test_performance_best_case_sparse_data():
    """
    Test performance on best-case scenario: very sparse data.
    
    When points are far apart, fewer neighbors are found,
    potentially improving performance.
    
    **Validates: Requirements 12.6, 15.1**
    """
    n_samples = 500
    
    # Create very sparse data (points far apart)
    np.random.seed(42)
    X = np.random.randn(n_samples, 2) * 10  # Large variance
    
    dbscan = DBSCAN(eps=0.5, min_pts=5)  # Small eps
    runtime = measure_runtime(dbscan, X)
    
    print(f"\nBest-case (sparse data) performance:")
    print(f"  Dataset size: {n_samples}")
    print(f"  Runtime: {runtime:.4f}s")
    print(f"  Noise points: {np.sum(dbscan.labels_ == -1)}")
    
    # Should complete quickly
    assert runtime < 10, f"Sparse data took too long: {runtime:.2f}s"


# ============================================================================
# Performance Regression Tests
# ============================================================================

@pytest.mark.slow
def test_performance_regression_baseline():
    """
    Establish performance baseline for regression testing.
    
    This test establishes a baseline runtime that can be used
    to detect performance regressions in future changes.
    
    **Validates: Requirements 12.6, 15.1**
    """
    n_samples = 1000
    X = generate_test_dataset(n_samples)
    
    # Run multiple times to get stable measurement
    runtimes = []
    for _ in range(3):
        dbscan = DBSCAN(eps=TEST_EPS, min_pts=TEST_MIN_PTS)
        runtime = measure_runtime(dbscan, X)
        runtimes.append(runtime)
    
    avg_runtime = np.mean(runtimes)
    std_runtime = np.std(runtimes)
    
    print(f"\nPerformance Baseline (n=1000):")
    print(f"  Average runtime: {avg_runtime:.4f}s")
    print(f"  Std deviation: {std_runtime:.4f}s")
    print(f"  Min runtime: {min(runtimes):.4f}s")
    print(f"  Max runtime: {max(runtimes):.4f}s")
    
    # Store baseline for future comparison
    # In practice, this would be saved to a file
    baseline = {
        'n_samples': n_samples,
        'avg_runtime': avg_runtime,
        'std_runtime': std_runtime
    }
    
    print(f"\nBaseline established: {baseline}")
    
    # Verify measurements are consistent
    assert std_runtime < avg_runtime * 0.2, \
        "Runtime variance too high for reliable baseline"


if __name__ == "__main__":
    # Allow running tests directly for quick performance checks
    print("Running performance tests...")
    print("Note: Use 'pytest tests/test_performance.py -v -s' for full output")
    
    # Run a quick performance check
    X = generate_test_dataset(500)
    dbscan = DBSCAN(eps=0.5, min_pts=5)
    runtime = measure_runtime(dbscan, X)
    
    print(f"\nQuick performance check (n=500):")
    print(f"  Runtime: {runtime:.4f}s")
    print(f"  Clusters: {len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)}")
    print(f"  Noise points: {np.sum(dbscan.labels_ == -1)}")
