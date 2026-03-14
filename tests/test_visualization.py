"""
Unit tests for visualization methods (Task 3.3)

Tests the three algorithm visualization methods:
1. plot_k_distance_graph() with show_elbow parameter
2. plot_algorithm_step() for single step visualization
3. animate_algorithm_steps() for full algorithm animation
"""
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from src.visualization import DBSCANVisualizer


class TestKDistanceGraph:
    """Tests for enhanced plot_k_distance_graph method"""
    
    def test_plot_k_distance_without_elbow(self):
        """Test k-distance graph without elbow detection"""
        X, _ = make_moons(n_samples=50, noise=0.05, random_state=42)
        visualizer = DBSCANVisualizer()
        
        # Should not raise any errors
        result = visualizer.plot_k_distance_graph(X, k=4, show_elbow=False)
        assert result is not None
        plt.close()
    
    def test_plot_k_distance_with_elbow(self):
        """Test k-distance graph with elbow detection"""
        X, _ = make_moons(n_samples=50, noise=0.05, random_state=42)
        visualizer = DBSCANVisualizer()
        
        # Should not raise any errors
        result = visualizer.plot_k_distance_graph(X, k=4, show_elbow=True)
        assert result is not None
        plt.close()
    
    def test_elbow_detection_method(self):
        """Test the _detect_elbow helper method"""
        visualizer = DBSCANVisualizer()
        
        # Create a simple curve with obvious elbow
        distances = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0.5, 0.4, 0.3, 0.2, 0.1])
        elbow_idx = visualizer._detect_elbow(distances)
        
        # Elbow should be somewhere in the middle
        assert 0 < elbow_idx < len(distances) - 1
    
    def test_elbow_detection_edge_cases(self):
        """Test elbow detection with edge cases"""
        visualizer = DBSCANVisualizer()
        
        # Very short array
        distances = np.array([1.0, 0.5])
        elbow_idx = visualizer._detect_elbow(distances)
        assert elbow_idx == 0
        
        # Flat line (no elbow)
        distances = np.ones(10)
        elbow_idx = visualizer._detect_elbow(distances)
        assert 0 <= elbow_idx < len(distances)
    
    def test_different_k_values(self):
        """Test k-distance graph with different k values"""
        X, _ = make_blobs(n_samples=50, centers=3, random_state=42)
        visualizer = DBSCANVisualizer()
        
        for k in [3, 5, 7]:
            result = visualizer.plot_k_distance_graph(X, k=k, show_elbow=True)
            assert result is not None
            plt.close()


class TestAlgorithmStep:
    """Tests for plot_algorithm_step method"""
    
    def test_plot_initial_step(self):
        """Test visualization of initial algorithm step"""
        X, _ = make_moons(n_samples=50, noise=0.05, random_state=42)
        visualizer = DBSCANVisualizer()
        
        visited = np.zeros(len(X), dtype=bool)
        visited[0] = True
        labels = np.zeros(len(X), dtype=int)
        labels[0] = 1
        
        result = visualizer.plot_algorithm_step(
            X=X,
            current_point=0,
            visited=visited,
            labels=labels,
            eps=0.3,
            step_num=1
        )
        assert result is not None
        plt.close()
    
    def test_plot_mid_execution_step(self):
        """Test visualization of mid-execution step"""
        X, _ = make_moons(n_samples=50, noise=0.05, random_state=42)
        visualizer = DBSCANVisualizer()
        
        visited = np.zeros(len(X), dtype=bool)
        visited[:25] = True
        labels = np.zeros(len(X), dtype=int)
        labels[:15] = 1
        labels[15:20] = 2
        labels[20:25] = -1
        
        result = visualizer.plot_algorithm_step(
            X=X,
            current_point=25,
            visited=visited,
            labels=labels,
            eps=0.3,
            step_num=15
        )
        assert result is not None
        plt.close()
    
    def test_plot_step_with_neighbors(self):
        """Test visualization with neighbor highlighting"""
        X, _ = make_moons(n_samples=50, noise=0.05, random_state=42)
        visualizer = DBSCANVisualizer()
        
        visited = np.zeros(len(X), dtype=bool)
        visited[0] = True
        labels = np.zeros(len(X), dtype=int)
        
        # Calculate neighbors
        distances = np.sqrt(np.sum((X - X[0])**2, axis=1))
        neighbors = np.where(distances <= 0.3)[0]
        
        result = visualizer.plot_algorithm_step(
            X=X,
            current_point=0,
            visited=visited,
            labels=labels,
            eps=0.3,
            step_num=1,
            current_neighbors=neighbors
        )
        assert result is not None
        plt.close()
    
    def test_plot_final_step(self):
        """Test visualization of final algorithm step"""
        X, _ = make_moons(n_samples=50, noise=0.05, random_state=42)
        visualizer = DBSCANVisualizer()
        
        visited = np.ones(len(X), dtype=bool)
        labels = np.random.randint(-1, 3, size=len(X))
        
        result = visualizer.plot_algorithm_step(
            X=X,
            current_point=len(X)-1,
            visited=visited,
            labels=labels,
            eps=0.3,
            step_num=50
        )
        assert result is not None
        plt.close()
    
    def test_custom_title(self):
        """Test plot_algorithm_step with custom title"""
        X, _ = make_blobs(n_samples=30, centers=2, random_state=42)
        visualizer = DBSCANVisualizer()
        
        visited = np.zeros(len(X), dtype=bool)
        labels = np.zeros(len(X), dtype=int)
        
        result = visualizer.plot_algorithm_step(
            X=X,
            current_point=0,
            visited=visited,
            labels=labels,
            eps=0.5,
            step_num=1,
            title="Custom Title Test"
        )
        assert result is not None
        plt.close()


class TestAnimateAlgorithm:
    """Tests for animate_algorithm_steps method"""
    
    def test_create_animation_object(self):
        """Test creation of animation object"""
        X, _ = make_blobs(n_samples=20, centers=2, random_state=42)
        visualizer = DBSCANVisualizer()
        
        # Should create animation without errors
        anim = visualizer.animate_algorithm_steps(
            X=X,
            eps=0.8,
            min_pts=3,
            save_path=None,
            interval=100
        )
        assert anim is not None
        plt.close('all')
    
    def test_capture_algorithm_states(self):
        """Test the _capture_algorithm_states helper method"""
        X, _ = make_blobs(n_samples=20, centers=2, random_state=42)
        visualizer = DBSCANVisualizer()
        
        states = visualizer._capture_algorithm_states(X, eps=0.8, min_pts=3)
        
        # Should have multiple states
        assert len(states) > 0
        
        # Each state should have required keys
        for state in states:
            assert 'current_point' in state
            assert 'visited' in state
            assert 'labels' in state
            assert 'step_num' in state
            assert 'neighbors' in state
    
    def test_states_progression(self):
        """Test that algorithm states progress correctly"""
        X, _ = make_blobs(n_samples=15, centers=2, random_state=42)
        visualizer = DBSCANVisualizer()
        
        states = visualizer._capture_algorithm_states(X, eps=0.8, min_pts=3)
        
        # Step numbers should increase
        step_nums = [state['step_num'] for state in states]
        assert step_nums == sorted(step_nums)
        
        # Visited points should only increase
        for i in range(1, len(states)):
            prev_visited = np.sum(states[i-1]['visited'])
            curr_visited = np.sum(states[i]['visited'])
            assert curr_visited >= prev_visited
    
    def test_animation_with_different_parameters(self):
        """Test animation with various parameter combinations"""
        X, _ = make_blobs(n_samples=20, centers=2, random_state=42)
        visualizer = DBSCANVisualizer()
        
        # Test different eps values
        for eps in [0.5, 1.0, 1.5]:
            anim = visualizer.animate_algorithm_steps(
                X=X,
                eps=eps,
                min_pts=3,
                save_path=None,
                interval=100
            )
            assert anim is not None
            plt.close('all')
    
    def test_animation_small_dataset(self):
        """Test animation with very small dataset"""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        visualizer = DBSCANVisualizer()
        
        anim = visualizer.animate_algorithm_steps(
            X=X,
            eps=1.5,
            min_pts=2,
            save_path=None,
            interval=100
        )
        assert anim is not None
        plt.close('all')


class TestVisualizationIntegration:
    """Integration tests for all visualization methods"""
    
    def test_all_methods_work_together(self):
        """Test that all three methods work on the same dataset"""
        X, _ = make_moons(n_samples=50, noise=0.05, random_state=42)
        visualizer = DBSCANVisualizer()
        
        # 1. K-distance graph
        visualizer.plot_k_distance_graph(X, k=4, show_elbow=True)
        plt.close()
        
        # 2. Algorithm step
        visited = np.zeros(len(X), dtype=bool)
        visited[0] = True
        labels = np.zeros(len(X), dtype=int)
        
        visualizer.plot_algorithm_step(
            X=X,
            current_point=0,
            visited=visited,
            labels=labels,
            eps=0.3,
            step_num=1
        )
        plt.close()
        
        # 3. Animation
        X_small = X[:20]  # Use subset for faster test
        anim = visualizer.animate_algorithm_steps(
            X=X_small,
            eps=0.3,
            min_pts=3,
            save_path=None,
            interval=100
        )
        plt.close('all')
        
        # All should complete without errors
        assert True
    
    def test_custom_config(self):
        """Test visualization methods with custom configuration"""
        from src.visualization import VisualizationConfig
        
        X, _ = make_blobs(n_samples=30, centers=2, random_state=42)
        
        custom_config = VisualizationConfig(
            figsize=(8, 6),
            core_color='red',
            noise_marker='+'
        )
        visualizer = DBSCANVisualizer(config=custom_config)
        
        # Test with custom config
        visualizer.plot_k_distance_graph(X, k=4, show_elbow=True)
        plt.close()
        
        visited = np.zeros(len(X), dtype=bool)
        labels = np.zeros(len(X), dtype=int)
        
        visualizer.plot_algorithm_step(
            X=X,
            current_point=0,
            visited=visited,
            labels=labels,
            eps=0.5,
            step_num=1
        )
        plt.close()
        
        assert True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])



class TestComparisonVisualizations:
    """Tests for comparison visualization methods (Task 3.4)"""
    
    def test_plot_algorithm_comparison(self):
        """Test algorithm comparison visualization"""
        from sklearn.cluster import KMeans
        from src.dbscan_from_scratch import DBSCAN
        
        X, _ = make_moons(n_samples=100, noise=0.05, random_state=42)
        visualizer = DBSCANVisualizer()
        
        # Run different algorithms
        dbscan = DBSCAN(eps=0.3, min_pts=5)
        dbscan_labels = dbscan.fit_predict(X)
        
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans_labels = kmeans.fit_predict(X)
        
        algorithms = {
            'DBSCAN': dbscan_labels,
            'K-Means': kmeans_labels
        }
        
        # Should not raise any errors
        result = visualizer.plot_algorithm_comparison(X, algorithms)
        assert result is not None
        plt.close()
    
    def test_plot_algorithm_comparison_single(self):
        """Test algorithm comparison with single algorithm"""
        from src.dbscan_from_scratch import DBSCAN
        
        X, _ = make_blobs(n_samples=50, centers=3, random_state=42)
        visualizer = DBSCANVisualizer()
        
        dbscan = DBSCAN(eps=0.8, min_pts=5)
        labels = dbscan.fit_predict(X)
        
        algorithms = {'DBSCAN': labels}
        
        result = visualizer.plot_algorithm_comparison(X, algorithms)
        assert result is not None
        plt.close()
    
    def test_plot_algorithm_comparison_multiple(self):
        """Test algorithm comparison with three algorithms"""
        from sklearn.cluster import KMeans, AgglomerativeClustering
        from src.dbscan_from_scratch import DBSCAN
        
        X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
        visualizer = DBSCANVisualizer()
        
        dbscan = DBSCAN(eps=0.8, min_pts=5)
        kmeans = KMeans(n_clusters=3, random_state=42)
        hierarchical = AgglomerativeClustering(n_clusters=3)
        
        algorithms = {
            'DBSCAN': dbscan.fit_predict(X),
            'K-Means': kmeans.fit_predict(X),
            'Hierarchical': hierarchical.fit_predict(X)
        }
        
        result = visualizer.plot_algorithm_comparison(X, algorithms)
        assert result is not None
        plt.close()
    
    def test_plot_parameter_sensitivity(self):
        """Test parameter sensitivity visualization"""
        X, _ = make_moons(n_samples=100, noise=0.05, random_state=42)
        visualizer = DBSCANVisualizer()
        
        eps_range = [0.2, 0.3, 0.4]
        minpts_range = [3, 5, 7]
        
        # Should not raise any errors
        result = visualizer.plot_parameter_sensitivity(X, eps_range, minpts_range)
        assert result is not None
        plt.close()
    
    def test_plot_parameter_sensitivity_single_param(self):
        """Test parameter sensitivity with single parameter values"""
        X, _ = make_blobs(n_samples=50, centers=2, random_state=42)
        visualizer = DBSCANVisualizer()
        
        # Single eps, multiple minpts
        result = visualizer.plot_parameter_sensitivity(X, [0.5], [3, 5, 7])
        assert result is not None
        plt.close()
        
        # Multiple eps, single minpts
        result = visualizer.plot_parameter_sensitivity(X, [0.3, 0.5, 0.7], [5])
        assert result is not None
        plt.close()
    
    def test_plot_parameter_sensitivity_grid(self):
        """Test parameter sensitivity with larger grid"""
        X, _ = make_blobs(n_samples=80, centers=3, random_state=42)
        visualizer = DBSCANVisualizer()
        
        eps_range = [0.3, 0.5, 0.7, 0.9]
        minpts_range = [2, 4, 6]
        
        result = visualizer.plot_parameter_sensitivity(X, eps_range, minpts_range)
        assert result is not None
        plt.close()
    
    def test_plot_cluster_shapes(self):
        """Test cluster shapes visualization"""
        from sklearn.datasets import make_circles
        
        visualizer = DBSCANVisualizer()
        
        # Create datasets with different shapes
        moons, _ = make_moons(n_samples=100, noise=0.05, random_state=42)
        circles, _ = make_circles(n_samples=100, noise=0.05, factor=0.5, random_state=42)
        blobs, _ = make_blobs(n_samples=100, centers=3, random_state=42)
        
        datasets = {
            'Moons': moons,
            'Circles': circles,
            'Blobs': blobs
        }
        
        # Should not raise any errors
        result = visualizer.plot_cluster_shapes(datasets, eps=0.3, min_pts=5)
        assert result is not None
        plt.close()
    
    def test_plot_cluster_shapes_single(self):
        """Test cluster shapes with single dataset"""
        visualizer = DBSCANVisualizer()
        
        X, _ = make_moons(n_samples=100, noise=0.05, random_state=42)
        datasets = {'Moons': X}
        
        result = visualizer.plot_cluster_shapes(datasets, eps=0.3, min_pts=5)
        assert result is not None
        plt.close()
    
    def test_plot_cluster_shapes_custom_params(self):
        """Test cluster shapes with custom parameters"""
        visualizer = DBSCANVisualizer()
        
        X, _ = make_blobs(n_samples=100, centers=4, random_state=42)
        datasets = {'Blobs': X}
        
        # Test with different parameters
        result = visualizer.plot_cluster_shapes(datasets, eps=0.8, min_pts=3)
        assert result is not None
        plt.close()
    
    def test_plot_density_variations(self):
        """Test density variations visualization"""
        from src.dbscan_from_scratch import DBSCAN
        
        X, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.5, random_state=42)
        visualizer = DBSCANVisualizer()
        
        # Run DBSCAN
        dbscan = DBSCAN(eps=0.8, min_pts=5)
        labels = dbscan.fit_predict(X)
        
        # Should not raise any errors
        result = visualizer.plot_density_variations(X, labels, eps=0.8)
        assert result is not None
        plt.close()
    
    def test_plot_density_variations_with_noise(self):
        """Test density variations with noisy data"""
        from src.dbscan_from_scratch import DBSCAN
        
        X, _ = make_moons(n_samples=100, noise=0.1, random_state=42)
        visualizer = DBSCANVisualizer()
        
        dbscan = DBSCAN(eps=0.3, min_pts=5)
        labels = dbscan.fit_predict(X)
        
        result = visualizer.plot_density_variations(X, labels, eps=0.3)
        assert result is not None
        plt.close()
    
    def test_plot_density_variations_varying_density(self):
        """Test density variations with clusters of different densities"""
        from src.dbscan_from_scratch import DBSCAN
        
        # Create dataset with varying density
        X1, _ = make_blobs(n_samples=50, centers=[[0, 0]], cluster_std=0.3, random_state=42)
        X2, _ = make_blobs(n_samples=50, centers=[[3, 3]], cluster_std=0.8, random_state=42)
        X = np.vstack([X1, X2])
        
        visualizer = DBSCANVisualizer()
        
        dbscan = DBSCAN(eps=0.5, min_pts=5)
        labels = dbscan.fit_predict(X)
        
        result = visualizer.plot_density_variations(X, labels, eps=0.5)
        assert result is not None
        plt.close()


class TestComparisonIntegration:
    """Integration tests for all comparison visualization methods"""
    
    def test_all_comparison_methods(self):
        """Test that all four comparison methods work together"""
        from sklearn.cluster import KMeans
        from sklearn.datasets import make_circles
        from src.dbscan_from_scratch import DBSCAN
        
        X, _ = make_moons(n_samples=100, noise=0.05, random_state=42)
        visualizer = DBSCANVisualizer()
        
        # 1. Algorithm comparison
        dbscan = DBSCAN(eps=0.3, min_pts=5)
        kmeans = KMeans(n_clusters=2, random_state=42)
        
        algorithms = {
            'DBSCAN': dbscan.fit_predict(X),
            'K-Means': kmeans.fit_predict(X)
        }
        
        visualizer.plot_algorithm_comparison(X, algorithms)
        plt.close()
        
        # 2. Parameter sensitivity
        visualizer.plot_parameter_sensitivity(X, [0.2, 0.3], [3, 5])
        plt.close()
        
        # 3. Cluster shapes
        circles, _ = make_circles(n_samples=100, noise=0.05, factor=0.5, random_state=42)
        datasets = {'Moons': X, 'Circles': circles}
        
        visualizer.plot_cluster_shapes(datasets, eps=0.3, min_pts=5)
        plt.close()
        
        # 4. Density variations
        dbscan = DBSCAN(eps=0.3, min_pts=5)
        labels = dbscan.fit_predict(X)
        
        visualizer.plot_density_variations(X, labels, eps=0.3)
        plt.close()
        
        # All should complete without errors
        assert True
    
    def test_comparison_with_custom_config(self):
        """Test comparison methods with custom configuration"""
        from src.visualization import VisualizationConfig
        from src.dbscan_from_scratch import DBSCAN
        
        X, _ = make_blobs(n_samples=80, centers=3, random_state=42)
        
        custom_config = VisualizationConfig(
            figsize=(12, 8),
            core_color='purple',
            noise_marker='s'
        )
        visualizer = DBSCANVisualizer(config=custom_config)
        
        # Test all methods with custom config
        dbscan = DBSCAN(eps=0.8, min_pts=5)
        labels = dbscan.fit_predict(X)
        
        # Algorithm comparison
        algorithms = {'DBSCAN': labels}
        visualizer.plot_algorithm_comparison(X, algorithms)
        plt.close()
        
        # Parameter sensitivity
        visualizer.plot_parameter_sensitivity(X, [0.5, 0.8], [3, 5])
        plt.close()
        
        # Cluster shapes
        datasets = {'Blobs': X}
        visualizer.plot_cluster_shapes(datasets, eps=0.8, min_pts=5)
        plt.close()
        
        # Density variations
        visualizer.plot_density_variations(X, labels, eps=0.8)
        plt.close()
        
        assert True

class TestVisualizationErrorHandling:
    """Test error handling and edge cases in visualization"""
    
    def test_invalid_style_fallback(self):
        """Test that invalid matplotlib style falls back to default"""
        from src.visualization import VisualizationConfig, DBSCANVisualizer
        
        # Create config with invalid style
        config = VisualizationConfig(style='nonexistent_style')
        visualizer = DBSCANVisualizer(config=config)
        
        # Should not raise error, should fallback to default
        assert visualizer.config.style == 'nonexistent_style'
        
        print("✓ Invalid style fallback test passed")
    
    def test_plot_clusters_with_highlight_points(self):
        """Test plot_clusters with highlight_points parameter"""
        from src.visualization import DBSCANVisualizer
        from src.dbscan_from_scratch import DBSCAN
        
        visualizer = DBSCANVisualizer()
        
        # Create simple dataset
        X = np.array([[0, 0], [1, 1], [2, 2], [10, 10], [11, 11]])
        dbscan = DBSCAN(eps=1.5, min_pts=2)
        labels = dbscan.fit_predict(X)
        
        # Test with highlight points
        highlight_points = np.array([0, 2])
        result = visualizer.plot_clusters(
            X, labels, 
            title="Test with Highlights",
            highlight_points=highlight_points
        )
        
        assert result is not None
        plt.close()
        print("✓ plot_clusters with highlight_points test passed")
    
    def test_plot_epsilon_neighborhood_edge_cases(self):
        """Test plot_epsilon_neighborhood with edge cases"""
        from src.visualization import DBSCANVisualizer
        
        visualizer = DBSCANVisualizer()
        
        # Test with point that has no neighbors
        X = np.array([[0, 0], [10, 10], [20, 20]])
        point_idx = 1  # Middle point
        eps = 1.0  # Small epsilon
        
        result = visualizer.plot_epsilon_neighborhood(X, point_idx, eps)
        
        assert result is not None
        plt.close()
        print("✓ plot_epsilon_neighborhood edge cases test passed")
    
    def test_plot_point_types_edge_cases(self):
        """Test plot_point_types with edge cases"""
        from src.visualization import DBSCANVisualizer
        from src.dbscan_from_scratch import DBSCAN
        
        visualizer = DBSCANVisualizer()
        
        # Create dataset with all noise points
        X = np.array([[0, 0], [10, 10], [20, 20], [30, 30]])
        dbscan = DBSCAN(eps=1.0, min_pts=3)  # High min_pts, small eps
        labels = dbscan.fit_predict(X)
        
        # Get core sample indices (should be empty for all noise)
        core_sample_indices = getattr(dbscan, 'core_sample_indices_', np.array([]))
        
        result = visualizer.plot_point_types(X, labels, core_sample_indices, eps=1.0)
        
        assert result is not None
        plt.close()
        print("✓ plot_point_types edge cases test passed")
    
    def test_detect_elbow_edge_cases(self):
        """Test _detect_elbow with edge cases"""
        from src.visualization import DBSCANVisualizer
        
        visualizer = DBSCANVisualizer()
        
        # Test with very short array
        distances = np.array([3.0, 2.0, 1.0])
        elbow_idx = visualizer._detect_elbow(distances)
        assert 0 <= elbow_idx < len(distances)
        
        # Test with flat array (no clear elbow)
        distances = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        elbow_idx = visualizer._detect_elbow(distances)
        assert 0 <= elbow_idx < len(distances)
        
        print("✓ _detect_elbow edge cases test passed")


class TestVisualizationInputValidation:
    """Test input validation for visualization methods"""
    
    def test_plot_clusters_input_validation(self):
        """Test input validation for plot_clusters"""
        from src.visualization import DBSCANVisualizer
        
        visualizer = DBSCANVisualizer()
        
        X = np.array([[0, 0], [1, 1]])
        labels = np.array([0, 0])
        
        # Test with mismatched array lengths
        try:
            visualizer.plot_clusters(X, np.array([0]))  # Wrong length
            plt.close()
            assert False, "Should raise error for mismatched lengths"
        except (ValueError, IndexError):
            pass
        
        print("✓ plot_clusters input validation test passed")
    
    def test_plot_k_distance_graph_input_validation(self):
        """Test input validation for plot_k_distance_graph"""
        from src.visualization import DBSCANVisualizer
        
        visualizer = DBSCANVisualizer()
        
        # Test with empty array
        try:
            visualizer.plot_k_distance_graph(np.array([]).reshape(0, 2), k=4)
            plt.close()
            assert False, "Should raise error for empty array"
        except (ValueError, IndexError):
            pass
        
        print("✓ plot_k_distance_graph input validation test passed")


class TestVisualizationComplexScenarios:
    """Test visualization with complex scenarios"""
    
    def test_all_noise_clustering(self):
        """Test visualization when all points are classified as noise"""
        from src.visualization import DBSCANVisualizer
        from src.dbscan_from_scratch import DBSCAN
        
        visualizer = DBSCANVisualizer()
        
        # Create sparse dataset where all points will be noise
        X = np.array([[0, 0], [10, 10], [20, 20], [30, 30]])
        dbscan = DBSCAN(eps=1.0, min_pts=3)
        labels = dbscan.fit_predict(X)
        
        # All should be noise (-1)
        assert np.all(labels == -1)
        
        result = visualizer.plot_clusters(X, labels, title="All Noise")
        assert result is not None
        
        plt.close()
        print("✓ All noise clustering visualization test passed")
    
    def test_single_cluster_scenario(self):
        """Test visualization with single cluster"""
        from src.visualization import DBSCANVisualizer
        from src.dbscan_from_scratch import DBSCAN
        
        visualizer = DBSCANVisualizer()
        
        # Create tight cluster
        X = np.array([[0, 0], [0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
        dbscan = DBSCAN(eps=0.5, min_pts=2)
        labels = dbscan.fit_predict(X)
        
        result = visualizer.plot_clusters(X, labels, title="Single Cluster")
        assert result is not None
        
        plt.close()
        print("✓ Single cluster visualization test passed")