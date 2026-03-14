"""
Test Fixtures Validation

This test file validates that all fixtures in conftest.py work correctly.
These tests ensure the fixtures provide the expected data and instances.
"""
import pytest
import numpy as np
from src.dbscan_from_scratch import DBSCAN, PointType
from src.visualization import DBSCANVisualizer
from src.data_loader import DatasetGenerator


class TestDataFixtures:
    """Test data fixtures from conftest.py"""
    
    def test_sample_data(self, sample_data):
        """Test sample_data fixture"""
        assert sample_data.shape == (100, 2)
        assert np.issubdtype(sample_data.dtype, np.number)
        assert not np.any(np.isnan(sample_data))
    
    def test_sample_data_blobs(self, sample_data_blobs):
        """Test sample_data_blobs fixture"""
        assert sample_data_blobs.shape == (150, 2)
        assert np.issubdtype(sample_data_blobs.dtype, np.number)
    
    def test_sample_data_circles(self, sample_data_circles):
        """Test sample_data_circles fixture"""
        assert sample_data_circles.shape == (100, 2)
        assert np.issubdtype(sample_data_circles.dtype, np.number)
    
    def test_sample_data_small(self, sample_data_small):
        """Test sample_data_small fixture"""
        assert sample_data_small.shape == (20, 2)
        assert np.issubdtype(sample_data_small.dtype, np.number)
    
    def test_sample_data_with_noise(self, sample_data_with_noise):
        """Test sample_data_with_noise fixture"""
        assert sample_data_with_noise.shape == (13, 2)
        assert np.issubdtype(sample_data_with_noise.dtype, np.number)
    
    def test_empty_dataset(self, empty_dataset):
        """Test empty_dataset fixture"""
        assert empty_dataset.shape == (0, 2)
    
    def test_single_point_dataset(self, single_point_dataset):
        """Test single_point_dataset fixture"""
        assert single_point_dataset.shape == (1, 2)
        assert np.allclose(single_point_dataset[0], [0, 0])


class TestDBSCANFixtures:
    """Test DBSCAN instance fixtures from conftest.py"""
    
    def test_dbscan_instance(self, dbscan_instance):
        """Test dbscan_instance fixture"""
        assert isinstance(dbscan_instance, DBSCAN)
        assert dbscan_instance.eps == 0.5
        assert dbscan_instance.min_pts == 5
        assert dbscan_instance.metric == 'euclidean'
    
    def test_dbscan_instance_tight(self, dbscan_instance_tight):
        """Test dbscan_instance_tight fixture"""
        assert isinstance(dbscan_instance_tight, DBSCAN)
        assert dbscan_instance_tight.eps == 0.3
        assert dbscan_instance_tight.min_pts == 5
    
    def test_dbscan_instance_loose(self, dbscan_instance_loose):
        """Test dbscan_instance_loose fixture"""
        assert isinstance(dbscan_instance_loose, DBSCAN)
        assert dbscan_instance_loose.eps == 1.0
        assert dbscan_instance_loose.min_pts == 3
    
    def test_dbscan_manhattan(self, dbscan_manhattan):
        """Test dbscan_manhattan fixture"""
        assert isinstance(dbscan_manhattan, DBSCAN)
        assert dbscan_manhattan.metric == 'manhattan'
    
    def test_dbscan_chebyshev(self, dbscan_chebyshev):
        """Test dbscan_chebyshev fixture"""
        assert isinstance(dbscan_chebyshev, DBSCAN)
        assert dbscan_chebyshev.metric == 'chebyshev'


class TestVisualizationFixtures:
    """Test visualization fixtures from conftest.py"""
    
    def test_visualizer(self, visualizer):
        """Test visualizer fixture"""
        assert isinstance(visualizer, DBSCANVisualizer)
        assert visualizer.config is not None
    
    def test_visualizer_custom(self, visualizer_custom):
        """Test visualizer_custom fixture"""
        assert isinstance(visualizer_custom, DBSCANVisualizer)
        assert visualizer_custom.config.core_color == 'red'
        assert visualizer_custom.config.noise_marker == '+'


class TestDataGeneratorFixture:
    """Test data generator fixture from conftest.py"""
    
    def test_data_generator(self, data_generator):
        """Test data_generator fixture"""
        assert isinstance(data_generator, DatasetGenerator)


class TestFittedDBSCANFixtures:
    """Test fitted DBSCAN fixtures from conftest.py"""
    
    def test_fitted_dbscan(self, fitted_dbscan):
        """Test fitted_dbscan fixture"""
        dbscan, labels, X = fitted_dbscan
        
        assert isinstance(dbscan, DBSCAN)
        assert isinstance(labels, np.ndarray)
        assert isinstance(X, np.ndarray)
        assert len(labels) == len(X)
        assert dbscan.labels_ is not None
        assert dbscan.core_sample_indices_ is not None
    
    def test_fitted_dbscan_blobs(self, fitted_dbscan_blobs):
        """Test fitted_dbscan_blobs fixture"""
        dbscan, labels, X = fitted_dbscan_blobs
        
        assert isinstance(dbscan, DBSCAN)
        assert isinstance(labels, np.ndarray)
        assert isinstance(X, np.ndarray)
        assert len(labels) == len(X)
        assert dbscan.labels_ is not None


class TestFixtureIntegration:
    """Test that fixtures work together in realistic scenarios"""
    
    def test_dbscan_with_sample_data(self, dbscan_instance, sample_data):
        """Test using DBSCAN instance with sample data"""
        labels = dbscan_instance.fit_predict(sample_data)
        
        assert len(labels) == len(sample_data)
        assert dbscan_instance.labels_ is not None
        assert dbscan_instance.core_sample_indices_ is not None
    
    def test_visualizer_with_fitted_dbscan(self, visualizer, fitted_dbscan):
        """Test using visualizer with fitted DBSCAN"""
        dbscan, labels, X = fitted_dbscan
        
        # Should not raise any errors
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        visualizer.plot_clusters(X, labels, "Test Plot")
        plt.close()
    
    def test_multiple_metrics_same_data(self, sample_data, dbscan_instance, 
                                       dbscan_manhattan, dbscan_chebyshev):
        """Test different distance metrics on same data"""
        labels_euclidean = dbscan_instance.fit_predict(sample_data)
        labels_manhattan = dbscan_manhattan.fit_predict(sample_data)
        labels_chebyshev = dbscan_chebyshev.fit_predict(sample_data)
        
        # All should produce valid labels
        assert len(labels_euclidean) == len(sample_data)
        assert len(labels_manhattan) == len(sample_data)
        assert len(labels_chebyshev) == len(sample_data)
    
    def test_edge_cases(self, dbscan_instance, empty_dataset, single_point_dataset):
        """Test DBSCAN with edge case datasets"""
        # Empty dataset
        labels_empty = dbscan_instance.fit_predict(empty_dataset)
        assert len(labels_empty) == 0
        
        # Single point
        labels_single = dbscan_instance.fit_predict(single_point_dataset)
        assert len(labels_single) == 1
        assert labels_single[0] == -1  # Should be noise


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
