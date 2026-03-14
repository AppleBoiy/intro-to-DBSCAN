"""
Unit tests for mathematical and performance visualizations (Task 3.5)

Tests the three new visualization methods:
1. plot_distance_metrics() showing geometric distances
2. plot_complexity_analysis() with theoretical curves
3. plot_scalability_benchmark() from results dataframe
"""
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from src.visualization import DBSCANVisualizer


class TestDistanceMetrics:
    """Tests for plot_distance_metrics method"""
    
    def test_plot_distance_metrics_all_metrics(self):
        """Test distance metrics visualization with all three metrics"""
        visualizer = DBSCANVisualizer()
        
        point_a = np.array([0, 0])
        point_b = np.array([3, 4])
        
        # Should not raise any errors
        result = visualizer.plot_distance_metrics(point_a, point_b)
        assert result is not None
        plt.close()
    
    def test_plot_distance_metrics_single_metric(self):
        """Test distance metrics with single metric"""
        visualizer = DBSCANVisualizer()
        
        point_a = np.array([1, 1])
        point_b = np.array([4, 5])
        
        # Test each metric individually
        for metric in ['euclidean', 'manhattan', 'chebyshev']:
            result = visualizer.plot_distance_metrics(point_a, point_b, metrics=[metric])
            assert result is not None
            plt.close()
    
    def test_plot_distance_metrics_two_metrics(self):
        """Test distance metrics with two metrics"""
        visualizer = DBSCANVisualizer()
        
        point_a = np.array([0, 0])
        point_b = np.array([2, 3])
        
        result = visualizer.plot_distance_metrics(
            point_a, point_b, 
            metrics=['euclidean', 'manhattan']
        )
        assert result is not None
        plt.close()
    
    def test_plot_distance_metrics_negative_coords(self):
        """Test distance metrics with negative coordinates"""
        visualizer = DBSCANVisualizer()
        
        point_a = np.array([-2, -3])
        point_b = np.array([1, 2])
        
        result = visualizer.plot_distance_metrics(point_a, point_b)
        assert result is not None
        plt.close()
    
    def test_plot_distance_metrics_same_point(self):
        """Test distance metrics when points are the same"""
        visualizer = DBSCANVisualizer()
        
        point_a = np.array([1, 1])
        point_b = np.array([1, 1])
        
        result = visualizer.plot_distance_metrics(point_a, point_b)
        assert result is not None
        plt.close()
    
    def test_plot_distance_metrics_custom_title(self):
        """Test distance metrics with custom title"""
        visualizer = DBSCANVisualizer()
        
        point_a = np.array([0, 0])
        point_b = np.array([5, 5])
        
        result = visualizer.plot_distance_metrics(
            point_a, point_b,
            title="Custom Distance Comparison"
        )
        assert result is not None
        plt.close()


class TestComplexityAnalysis:
    """Tests for plot_complexity_analysis method"""
    
    def test_plot_complexity_analysis_quadratic(self):
        """Test complexity analysis with O(n²) complexity"""
        visualizer = DBSCANVisualizer()
        
        sizes = np.array([100, 500, 1000, 2000, 5000])
        times = np.array([0.01, 0.25, 1.0, 4.0, 25.0])
        
        result = visualizer.plot_complexity_analysis(sizes, times)
        assert result is not None
        plt.close()
    
    def test_plot_complexity_analysis_nlogn(self):
        """Test complexity analysis with O(n log n) complexity"""
        visualizer = DBSCANVisualizer()
        
        sizes = np.array([100, 500, 1000, 5000, 10000])
        # Generate times that follow O(n log n)
        times = sizes * np.log(sizes) * 0.0001
        
        result = visualizer.plot_complexity_analysis(
            sizes, times,
            theoretical_complexity="O(n log n)"
        )
        assert result is not None
        plt.close()
    
    def test_plot_complexity_analysis_linear(self):
        """Test complexity analysis with O(n) complexity"""
        visualizer = DBSCANVisualizer()
        
        sizes = np.array([100, 500, 1000, 5000])
        times = sizes * 0.001
        
        result = visualizer.plot_complexity_analysis(
            sizes, times,
            theoretical_complexity="O(n)"
        )
        assert result is not None
        plt.close()
    
    def test_plot_complexity_analysis_custom_algorithm(self):
        """Test complexity analysis with custom algorithm name"""
        visualizer = DBSCANVisualizer()
        
        sizes = np.array([100, 500, 1000])
        times = np.array([0.01, 0.25, 1.0])
        
        result = visualizer.plot_complexity_analysis(
            sizes, times,
            algorithm_name="Custom Algorithm"
        )
        assert result is not None
        plt.close()
    
    def test_plot_complexity_analysis_custom_title(self):
        """Test complexity analysis with custom title"""
        visualizer = DBSCANVisualizer()
        
        sizes = np.array([100, 500, 1000])
        times = np.array([0.01, 0.25, 1.0])
        
        result = visualizer.plot_complexity_analysis(
            sizes, times,
            title="Custom Complexity Analysis"
        )
        assert result is not None
        plt.close()
    
    def test_plot_complexity_analysis_small_dataset(self):
        """Test complexity analysis with small dataset"""
        visualizer = DBSCANVisualizer()
        
        sizes = np.array([10, 20, 30])
        times = np.array([0.001, 0.004, 0.009])
        
        result = visualizer.plot_complexity_analysis(sizes, times)
        assert result is not None
        plt.close()


class TestScalabilityBenchmark:
    """Tests for plot_scalability_benchmark method"""
    
    def test_plot_scalability_benchmark_runtime_only(self):
        """Test scalability benchmark with runtime data only"""
        visualizer = DBSCANVisualizer()
        
        results = pd.DataFrame({
            'n_samples': [100, 500, 1000, 5000],
            'runtime': [0.01, 0.25, 1.0, 25.0]
        })
        
        result = visualizer.plot_scalability_benchmark(results)
        assert result is not None
        plt.close()
    
    def test_plot_scalability_benchmark_with_memory(self):
        """Test scalability benchmark with memory data"""
        visualizer = DBSCANVisualizer()
        
        results = pd.DataFrame({
            'n_samples': [100, 500, 1000, 5000],
            'runtime': [0.01, 0.25, 1.0, 25.0],
            'memory_mb': [10, 15, 25, 80]
        })
        
        result = visualizer.plot_scalability_benchmark(results)
        assert result is not None
        plt.close()
    
    def test_plot_scalability_benchmark_with_clusters(self):
        """Test scalability benchmark with cluster statistics"""
        visualizer = DBSCANVisualizer()
        
        results = pd.DataFrame({
            'n_samples': [100, 500, 1000, 5000],
            'runtime': [0.01, 0.25, 1.0, 25.0],
            'n_clusters': [3, 3, 3, 3],
            'n_noise': [5, 20, 40, 200]
        })
        
        result = visualizer.plot_scalability_benchmark(results)
        assert result is not None
        plt.close()
    
    def test_plot_scalability_benchmark_complete(self):
        """Test scalability benchmark with all data columns"""
        visualizer = DBSCANVisualizer()
        
        results = pd.DataFrame({
            'n_samples': [100, 500, 1000, 5000, 10000],
            'runtime': [0.01, 0.25, 1.0, 25.0, 100.0],
            'memory_mb': [10, 15, 25, 80, 200],
            'n_clusters': [3, 3, 3, 3, 3],
            'n_noise': [5, 20, 40, 200, 400]
        })
        
        result = visualizer.plot_scalability_benchmark(results)
        assert result is not None
        plt.close()
    
    def test_plot_scalability_benchmark_custom_title(self):
        """Test scalability benchmark with custom title"""
        visualizer = DBSCANVisualizer()
        
        results = pd.DataFrame({
            'n_samples': [100, 500, 1000],
            'runtime': [0.01, 0.25, 1.0]
        })
        
        result = visualizer.plot_scalability_benchmark(
            results,
            title="Custom Benchmark Results"
        )
        assert result is not None
        plt.close()
    
    def test_plot_scalability_benchmark_invalid_input(self):
        """Test scalability benchmark with invalid input"""
        visualizer = DBSCANVisualizer()
        
        # Not a DataFrame
        with pytest.raises(TypeError):
            visualizer.plot_scalability_benchmark([1, 2, 3])
    
    def test_plot_scalability_benchmark_missing_columns(self):
        """Test scalability benchmark with missing required columns"""
        visualizer = DBSCANVisualizer()
        
        # Missing 'runtime' column
        results = pd.DataFrame({
            'n_samples': [100, 500, 1000]
        })
        
        with pytest.raises(ValueError):
            visualizer.plot_scalability_benchmark(results)
    
    def test_plot_scalability_benchmark_single_datapoint(self):
        """Test scalability benchmark with single data point"""
        visualizer = DBSCANVisualizer()
        
        results = pd.DataFrame({
            'n_samples': [100],
            'runtime': [0.01]
        })
        
        result = visualizer.plot_scalability_benchmark(results)
        assert result is not None
        plt.close()


class TestMathematicalVisualizationsIntegration:
    """Integration tests for all mathematical and performance visualizations"""
    
    def test_all_methods_work_together(self):
        """Test that all three new methods work on related data"""
        visualizer = DBSCANVisualizer()
        
        # 1. Distance metrics
        point_a = np.array([0, 0])
        point_b = np.array([3, 4])
        visualizer.plot_distance_metrics(point_a, point_b)
        plt.close()
        
        # 2. Complexity analysis
        sizes = np.array([100, 500, 1000, 5000])
        times = np.array([0.01, 0.25, 1.0, 25.0])
        visualizer.plot_complexity_analysis(sizes, times)
        plt.close()
        
        # 3. Scalability benchmark
        results = pd.DataFrame({
            'n_samples': sizes,
            'runtime': times,
            'memory_mb': [10, 15, 25, 80],
            'n_clusters': [3, 3, 3, 3],
            'n_noise': [5, 20, 40, 200]
        })
        visualizer.plot_scalability_benchmark(results)
        plt.close()
        
        # All should complete without errors
        assert True
    
    def test_with_custom_config(self):
        """Test mathematical visualizations with custom configuration"""
        from src.visualization import VisualizationConfig
        
        custom_config = VisualizationConfig(
            figsize=(12, 8),
            dpi=150,
            show_grid=True
        )
        visualizer = DBSCANVisualizer(config=custom_config)
        
        # Test all methods with custom config
        point_a = np.array([0, 0])
        point_b = np.array([5, 5])
        visualizer.plot_distance_metrics(point_a, point_b)
        plt.close()
        
        sizes = np.array([100, 500, 1000])
        times = np.array([0.01, 0.25, 1.0])
        visualizer.plot_complexity_analysis(sizes, times)
        plt.close()
        
        results = pd.DataFrame({
            'n_samples': sizes,
            'runtime': times
        })
        visualizer.plot_scalability_benchmark(results)
        plt.close()
        
        assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
