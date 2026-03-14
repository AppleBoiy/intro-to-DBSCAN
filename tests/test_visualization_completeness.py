"""
Property-Based Test for Visualization System Completeness (Task 3.6)

**Property 6: Visualization System Completeness**
**Validates: Requirements 3.1-3.10, 8.1-8.8, 10.3, 10.8**

Tests that all core DBSCAN visualization functions exist and execute without error
on sample data. This ensures comprehensive visual coverage of all key concepts.
"""
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs, make_circles
from src.visualization import DBSCANVisualizer
from src.dbscan_from_scratch import DBSCAN


class TestVisualizationSystemCompleteness:
    """
    Property 6: Visualization System Completeness
    
    For all core DBSCAN concepts, the visualization system should provide
    at least one working visualization function that executes without error
    on sample data.
    """
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample dataset for testing"""
        X, _ = make_moons(n_samples=100, noise=0.05, random_state=42)
        return X
    
    @pytest.fixture
    def clustered_data(self, sample_data):
        """Generate clustered data with labels"""
        dbscan = DBSCAN(eps=0.3, min_pts=5)
        labels = dbscan.fit_predict(sample_data)
        return sample_data, labels, dbscan
    
    @pytest.fixture
    def visualizer(self):
        """Create visualizer instance"""
        return DBSCANVisualizer()

    
    # Test 1: Epsilon-Neighborhood Visualization (Requirement 3.4, 8.1)
    def test_plot_epsilon_neighborhood_executes(self, sample_data, visualizer):
        """Test plot_epsilon_neighborhood() executes without error"""
        try:
            result = visualizer.plot_epsilon_neighborhood(
                X=sample_data,
                point_idx=0,
                eps=0.3
            )
            assert result is not None
            plt.close()
        except Exception as e:
            pytest.fail(f"plot_epsilon_neighborhood() failed: {e}")
    
    # Test 2: Point Types Visualization (Requirement 3.5, 8.4)
    def test_plot_point_types_executes(self, clustered_data, visualizer):
        """Test plot_point_types() executes without error"""
        X, labels, dbscan = clustered_data
        try:
            result = visualizer.plot_point_types(
                X=X,
                labels=labels,
                core_sample_indices=dbscan.core_sample_indices_,
                eps=0.3
            )
            assert result is not None
            plt.close()
        except Exception as e:
            pytest.fail(f"plot_point_types() failed: {e}")
    
    # Test 3: Density-Reachability Visualization (Requirement 8.2)
    def test_plot_density_reachability_executes(self, sample_data, visualizer):
        """Test plot_density_reachability() executes without error"""
        try:
            # Create a simple chain of points
            point_chain = [0, 5, 12, 18]
            result = visualizer.plot_density_reachability(
                X=sample_data,
                point_chain=point_chain,
                eps=0.3
            )
            assert result is not None
            plt.close()
        except Exception as e:
            pytest.fail(f"plot_density_reachability() failed: {e}")

    
    # Test 4: Density-Connectivity Visualization (Requirement 8.3)
    def test_plot_density_connectivity_executes(self, sample_data, visualizer):
        """Test plot_density_connectivity() executes without error"""
        try:
            result = visualizer.plot_density_connectivity(
                X=sample_data,
                point_a=5,
                point_b=12,
                connecting_point=8,
                eps=0.3
            )
            assert result is not None
            plt.close()
        except Exception as e:
            pytest.fail(f"plot_density_connectivity() failed: {e}")
    
    # Test 5: K-Distance Graph (Requirement 3.8, 4.1)
    def test_plot_k_distance_graph_executes(self, sample_data, visualizer):
        """Test plot_k_distance_graph() executes without error"""
        try:
            result = visualizer.plot_k_distance_graph(
                X=sample_data,
                k=4,
                show_elbow=True
            )
            assert result is not None
            plt.close()
        except Exception as e:
            pytest.fail(f"plot_k_distance_graph() failed: {e}")
    
    # Test 6: Algorithm Step Visualization (Requirement 3.2, 3.6)
    def test_plot_algorithm_step_executes(self, sample_data, visualizer):
        """Test plot_algorithm_step() executes without error"""
        try:
            visited = np.zeros(len(sample_data), dtype=bool)
            visited[0] = True
            labels = np.zeros(len(sample_data), dtype=int)
            
            result = visualizer.plot_algorithm_step(
                X=sample_data,
                current_point=0,
                visited=visited,
                labels=labels,
                eps=0.3,
                step_num=1
            )
            assert result is not None
            plt.close()
        except Exception as e:
            pytest.fail(f"plot_algorithm_step() failed: {e}")

    
    # Test 7: Algorithm Animation (Requirement 3.6)
    def test_animate_algorithm_steps_executes(self, visualizer):
        """Test animate_algorithm_steps() executes without error"""
        try:
            # Use small dataset for faster animation
            X, _ = make_blobs(n_samples=20, centers=2, random_state=42)
            
            anim = visualizer.animate_algorithm_steps(
                X=X,
                eps=0.8,
                min_pts=3,
                save_path=None,
                interval=100
            )
            assert anim is not None
            plt.close('all')
        except Exception as e:
            pytest.fail(f"animate_algorithm_steps() failed: {e}")
    
    # Test 8: Algorithm Comparison (Requirement 3.7, 10.3, 10.8)
    def test_plot_algorithm_comparison_executes(self, sample_data, visualizer):
        """Test plot_algorithm_comparison() executes without error"""
        try:
            from sklearn.cluster import KMeans
            
            dbscan = DBSCAN(eps=0.3, min_pts=5)
            kmeans = KMeans(n_clusters=2, random_state=42)
            
            algorithms = {
                'DBSCAN': dbscan.fit_predict(sample_data),
                'K-Means': kmeans.fit_predict(sample_data)
            }
            
            result = visualizer.plot_algorithm_comparison(sample_data, algorithms)
            assert result is not None
            plt.close()
        except Exception as e:
            pytest.fail(f"plot_algorithm_comparison() failed: {e}")
    
    # Test 9: Parameter Sensitivity (Requirement 3.9, 4.3)
    def test_plot_parameter_sensitivity_executes(self, sample_data, visualizer):
        """Test plot_parameter_sensitivity() executes without error"""
        try:
            eps_range = [0.2, 0.3]
            minpts_range = [3, 5]
            
            result = visualizer.plot_parameter_sensitivity(
                sample_data, eps_range, minpts_range
            )
            assert result is not None
            plt.close()
        except Exception as e:
            pytest.fail(f"plot_parameter_sensitivity() failed: {e}")

    
    # Test 10: Cluster Shapes (Requirement 3.9, 10.8)
    def test_plot_cluster_shapes_executes(self, visualizer):
        """Test plot_cluster_shapes() executes without error"""
        try:
            moons, _ = make_moons(n_samples=100, noise=0.05, random_state=42)
            circles, _ = make_circles(n_samples=100, noise=0.05, factor=0.5, random_state=42)
            
            datasets = {
                'Moons': moons,
                'Circles': circles
            }
            
            result = visualizer.plot_cluster_shapes(datasets, eps=0.3, min_pts=5)
            assert result is not None
            plt.close()
        except Exception as e:
            pytest.fail(f"plot_cluster_shapes() failed: {e}")
    
    # Test 11: Density Variations (Requirement 3.10, 10.8)
    def test_plot_density_variations_executes(self, clustered_data, visualizer):
        """Test plot_density_variations() executes without error"""
        X, labels, _ = clustered_data
        try:
            result = visualizer.plot_density_variations(X, labels, eps=0.3)
            assert result is not None
            plt.close()
        except Exception as e:
            pytest.fail(f"plot_density_variations() failed: {e}")
    
    # Test 12: Distance Metrics (Requirement 8.6, 8.8)
    def test_plot_distance_metrics_executes(self, visualizer):
        """Test plot_distance_metrics() executes without error"""
        try:
            point_a = np.array([0, 0])
            point_b = np.array([3, 4])
            
            result = visualizer.plot_distance_metrics(point_a, point_b)
            assert result is not None
            plt.close()
        except Exception as e:
            pytest.fail(f"plot_distance_metrics() failed: {e}")
    
    # Test 13: Complexity Analysis (Requirement 8.7, 15.1)
    def test_plot_complexity_analysis_executes(self, visualizer):
        """Test plot_complexity_analysis() executes without error"""
        try:
            sizes = np.array([100, 500, 1000, 2000])
            times = np.array([0.01, 0.25, 1.0, 4.0])
            
            result = visualizer.plot_complexity_analysis(sizes, times)
            assert result is not None
            plt.close()
        except Exception as e:
            pytest.fail(f"plot_complexity_analysis() failed: {e}")

    
    # Test 14: Scalability Benchmark (Requirement 8.8, 15.6)
    def test_plot_scalability_benchmark_executes(self, visualizer):
        """Test plot_scalability_benchmark() executes without error"""
        try:
            results = pd.DataFrame({
                'n_samples': [100, 500, 1000, 5000],
                'runtime': [0.01, 0.25, 1.0, 25.0],
                'memory_mb': [10, 15, 25, 80],
                'n_clusters': [3, 3, 3, 3],
                'n_noise': [5, 20, 40, 200]
            })
            
            result = visualizer.plot_scalability_benchmark(results)
            assert result is not None
            plt.close()
        except Exception as e:
            pytest.fail(f"plot_scalability_benchmark() failed: {e}")
    
    # Test 15: Basic Cluster Visualization (Requirement 3.1)
    def test_plot_clusters_executes(self, clustered_data, visualizer):
        """Test plot_clusters() executes without error"""
        X, labels, _ = clustered_data
        try:
            result = visualizer.plot_clusters(X, labels, title="Test Clusters")
            assert result is not None
            plt.close()
        except Exception as e:
            pytest.fail(f"plot_clusters() failed: {e}")


class TestVisualizationEdgeCases:
    """Test visualization methods with edge cases"""
    
    @pytest.fixture
    def visualizer(self):
        return DBSCANVisualizer()
    
    def test_epsilon_neighborhood_with_no_neighbors(self, visualizer):
        """Test epsilon neighborhood when point has no neighbors"""
        X = np.array([[0, 0], [10, 10], [20, 20]])
        try:
            result = visualizer.plot_epsilon_neighborhood(X, point_idx=0, eps=0.5)
            assert result is not None
            plt.close()
        except Exception as e:
            pytest.fail(f"Failed with no neighbors: {e}")
    
    def test_single_cluster_visualization(self, visualizer):
        """Test visualization with single cluster"""
        X, _ = make_blobs(n_samples=50, centers=1, random_state=42)
        dbscan = DBSCAN(eps=1.0, min_pts=3)
        labels = dbscan.fit_predict(X)
        
        try:
            result = visualizer.plot_clusters(X, labels)
            assert result is not None
            plt.close()
        except Exception as e:
            pytest.fail(f"Failed with single cluster: {e}")

    
    def test_all_noise_visualization(self, visualizer):
        """Test visualization when all points are noise"""
        X = np.array([[0, 0], [10, 10], [20, 20], [30, 30]])
        dbscan = DBSCAN(eps=0.5, min_pts=5)
        labels = dbscan.fit_predict(X)
        
        try:
            result = visualizer.plot_clusters(X, labels)
            assert result is not None
            plt.close()
        except Exception as e:
            pytest.fail(f"Failed with all noise: {e}")
    
    def test_parameter_sensitivity_single_value(self, visualizer):
        """Test parameter sensitivity with single parameter value"""
        X, _ = make_blobs(n_samples=50, centers=2, random_state=42)
        
        try:
            result = visualizer.plot_parameter_sensitivity(X, [0.5], [5])
            assert result is not None
            plt.close()
        except Exception as e:
            pytest.fail(f"Failed with single parameter: {e}")
    
    def test_algorithm_comparison_single_algorithm(self, visualizer):
        """Test algorithm comparison with single algorithm"""
        X, _ = make_blobs(n_samples=50, centers=2, random_state=42)
        dbscan = DBSCAN(eps=0.8, min_pts=5)
        labels = dbscan.fit_predict(X)
        
        try:
            result = visualizer.plot_algorithm_comparison(X, {'DBSCAN': labels})
            assert result is not None
            plt.close()
        except Exception as e:
            pytest.fail(f"Failed with single algorithm: {e}")


class TestVisualizationIntegration:
    """Integration tests ensuring all visualizations work together"""
    
    @pytest.fixture
    def visualizer(self):
        return DBSCANVisualizer()
    
    @pytest.fixture
    def test_data(self):
        X, _ = make_moons(n_samples=100, noise=0.05, random_state=42)
        dbscan = DBSCAN(eps=0.3, min_pts=5)
        labels = dbscan.fit_predict(X)
        return X, labels, dbscan
    
    def test_all_concept_visualizations_work(self, test_data, visualizer):
        """Test that all concept visualization methods work on same dataset"""
        X, labels, dbscan = test_data
        
        try:
            # Epsilon neighborhood
            visualizer.plot_epsilon_neighborhood(X, point_idx=0, eps=0.3)
            plt.close()
            
            # Point types
            visualizer.plot_point_types(X, labels, dbscan.core_sample_indices_, eps=0.3)
            plt.close()
            
            # Density reachability
            visualizer.plot_density_reachability(X, [0, 5, 12], eps=0.3)
            plt.close()
            
            # Density connectivity
            visualizer.plot_density_connectivity(X, 5, 12, 8, eps=0.3)
            plt.close()
            
            assert True  # All visualizations completed successfully
        except Exception as e:
            pytest.fail(f"Concept visualizations integration failed: {e}")

    
    def test_all_algorithm_visualizations_work(self, test_data, visualizer):
        """Test that all algorithm visualization methods work on same dataset"""
        X, labels, dbscan = test_data
        
        try:
            # K-distance graph
            visualizer.plot_k_distance_graph(X, k=4, show_elbow=True)
            plt.close()
            
            # Algorithm step
            visited = np.zeros(len(X), dtype=bool)
            visited[0] = True
            step_labels = np.zeros(len(X), dtype=int)
            visualizer.plot_algorithm_step(X, 0, visited, step_labels, eps=0.3, step_num=1)
            plt.close()
            
            # Animation (small subset)
            X_small = X[:20]
            anim = visualizer.animate_algorithm_steps(X_small, eps=0.3, min_pts=3, interval=100)
            plt.close('all')
            
            assert True  # All visualizations completed successfully
        except Exception as e:
            pytest.fail(f"Algorithm visualizations integration failed: {e}")
    
    def test_all_comparison_visualizations_work(self, test_data, visualizer):
        """Test that all comparison visualization methods work on same dataset"""
        X, labels, dbscan = test_data
        
        try:
            from sklearn.cluster import KMeans
            
            # Algorithm comparison
            kmeans = KMeans(n_clusters=2, random_state=42)
            algorithms = {
                'DBSCAN': labels,
                'K-Means': kmeans.fit_predict(X)
            }
            visualizer.plot_algorithm_comparison(X, algorithms)
            plt.close()
            
            # Parameter sensitivity
            visualizer.plot_parameter_sensitivity(X, [0.2, 0.3], [3, 5])
            plt.close()
            
            # Cluster shapes
            circles, _ = make_circles(n_samples=100, noise=0.05, factor=0.5, random_state=42)
            datasets = {'Moons': X, 'Circles': circles}
            visualizer.plot_cluster_shapes(datasets, eps=0.3, min_pts=5)
            plt.close()
            
            # Density variations
            visualizer.plot_density_variations(X, labels, eps=0.3)
            plt.close()
            
            assert True  # All visualizations completed successfully
        except Exception as e:
            pytest.fail(f"Comparison visualizations integration failed: {e}")
    
    def test_all_mathematical_visualizations_work(self, visualizer):
        """Test that all mathematical visualization methods work"""
        try:
            # Distance metrics
            point_a = np.array([0, 0])
            point_b = np.array([3, 4])
            visualizer.plot_distance_metrics(point_a, point_b)
            plt.close()
            
            # Complexity analysis
            sizes = np.array([100, 500, 1000, 2000])
            times = np.array([0.01, 0.25, 1.0, 4.0])
            visualizer.plot_complexity_analysis(sizes, times)
            plt.close()
            
            # Scalability benchmark
            results = pd.DataFrame({
                'n_samples': [100, 500, 1000],
                'runtime': [0.01, 0.25, 1.0],
                'memory_mb': [10, 15, 25],
                'n_clusters': [3, 3, 3],
                'n_noise': [5, 20, 40]
            })
            visualizer.plot_scalability_benchmark(results)
            plt.close()
            
            assert True  # All visualizations completed successfully
        except Exception as e:
            pytest.fail(f"Mathematical visualizations integration failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
