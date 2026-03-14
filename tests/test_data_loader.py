"""
Unit Tests for Data Loader Module
Tests for dataset loading utilities including real-world datasets
"""
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import DatasetGenerator, DatasetMetadata


def test_load_gps_tracks():
    """Test loading GPS tracks dataset"""
    gen = DatasetGenerator()
    X, metadata = gen.load_real_world_dataset('gps_tracks')
    
    # Check data shape
    assert X.shape == (225, 2), f"Expected shape (225, 2), got {X.shape}"
    assert X.shape[0] == metadata.n_samples, "n_samples should match data shape"
    assert X.shape[1] == metadata.n_features, "n_features should match data shape"
    
    # Check metadata
    assert metadata.name == 'GPS Vehicle Tracks'
    assert metadata.expected_clusters == 10
    assert metadata.suggested_eps == 0.003
    assert metadata.suggested_minpts == 3
    assert metadata.difficulty == 'intermediate'
    assert metadata.source == 'real-world'
    
    # Check data is numeric
    assert np.issubdtype(X.dtype, np.number), "Data should be numeric"
    
    # Check no NaN values
    assert not np.any(np.isnan(X)), "Data should not contain NaN values"
    
    print("✓ GPS tracks loading test passed")


def test_load_customer_locations():
    """Test loading customer locations dataset"""
    gen = DatasetGenerator()
    X, metadata = gen.load_real_world_dataset('customer_locations')
    
    # Check data shape
    assert X.shape == (200, 2), f"Expected shape (200, 2), got {X.shape}"
    assert X.shape[0] == metadata.n_samples, "n_samples should match data shape"
    assert X.shape[1] == metadata.n_features, "n_features should match data shape"
    
    # Check metadata
    assert metadata.name == 'Retail Customer Locations'
    assert metadata.expected_clusters == 8
    assert metadata.suggested_eps == 0.005
    assert metadata.suggested_minpts == 4
    assert metadata.difficulty == 'intermediate'
    assert metadata.source == 'real-world'
    
    # Check data is numeric
    assert np.issubdtype(X.dtype, np.number), "Data should be numeric"
    
    # Check no NaN values
    assert not np.any(np.isnan(X)), "Data should not contain NaN values"
    
    print("✓ Customer locations loading test passed")


def test_load_sensor_readings():
    """Test loading sensor readings dataset"""
    gen = DatasetGenerator()
    X, metadata = gen.load_real_world_dataset('sensor_readings')
    
    # Check data shape
    assert X.shape == (185, 4), f"Expected shape (185, 4), got {X.shape}"
    assert X.shape[0] == metadata.n_samples, "n_samples should match data shape"
    assert X.shape[1] == metadata.n_features, "n_features should match data shape"
    
    # Check metadata
    assert metadata.name == 'IoT Sensor Readings'
    assert metadata.expected_clusters == 15
    assert metadata.suggested_eps == 2.5
    assert metadata.suggested_minpts == 3
    assert metadata.difficulty == 'advanced'
    assert metadata.source == 'real-world'
    
    # Check data is numeric
    assert np.issubdtype(X.dtype, np.number), "Data should be numeric"
    
    # Check no NaN values
    assert not np.any(np.isnan(X)), "Data should not contain NaN values"
    
    print("✓ Sensor readings loading test passed")


def test_invalid_dataset_name():
    """Test error handling for invalid dataset name"""
    gen = DatasetGenerator()
    
    try:
        X, metadata = gen.load_real_world_dataset('invalid_dataset')
        assert False, "Should raise ValueError for invalid dataset name"
    except ValueError as e:
        error_msg = str(e)
        assert 'Unknown dataset' in error_msg, "Error message should mention unknown dataset"
        assert 'gps_tracks' in error_msg, "Error message should list available datasets"
        assert 'customer_locations' in error_msg, "Error message should list available datasets"
        assert 'sensor_readings' in error_msg, "Error message should list available datasets"
    
    print("✓ Invalid dataset name test passed")


def test_metadata_structure():
    """Test DatasetMetadata structure and attributes"""
    gen = DatasetGenerator()
    X, metadata = gen.load_real_world_dataset('gps_tracks')
    
    # Check all required attributes exist
    assert hasattr(metadata, 'name'), "Metadata should have 'name' attribute"
    assert hasattr(metadata, 'description'), "Metadata should have 'description' attribute"
    assert hasattr(metadata, 'n_samples'), "Metadata should have 'n_samples' attribute"
    assert hasattr(metadata, 'n_features'), "Metadata should have 'n_features' attribute"
    assert hasattr(metadata, 'expected_clusters'), "Metadata should have 'expected_clusters' attribute"
    assert hasattr(metadata, 'suggested_eps'), "Metadata should have 'suggested_eps' attribute"
    assert hasattr(metadata, 'suggested_minpts'), "Metadata should have 'suggested_minpts' attribute"
    assert hasattr(metadata, 'difficulty'), "Metadata should have 'difficulty' attribute"
    assert hasattr(metadata, 'source'), "Metadata should have 'source' attribute"
    assert hasattr(metadata, 'paper_reference'), "Metadata should have 'paper_reference' attribute"
    
    # Check types
    assert isinstance(metadata.name, str), "name should be string"
    assert isinstance(metadata.description, str), "description should be string"
    assert isinstance(metadata.n_samples, int), "n_samples should be int"
    assert isinstance(metadata.n_features, int), "n_features should be int"
    assert isinstance(metadata.expected_clusters, int), "expected_clusters should be int"
    assert isinstance(metadata.suggested_eps, (int, float)), "suggested_eps should be numeric"
    assert isinstance(metadata.suggested_minpts, int), "suggested_minpts should be int"
    assert isinstance(metadata.difficulty, str), "difficulty should be string"
    assert isinstance(metadata.source, str), "source should be string"
    
    # Check valid values
    assert metadata.difficulty in ['beginner', 'intermediate', 'advanced'], \
        "difficulty should be beginner, intermediate, or advanced"
    assert metadata.source in ['synthetic', 'real-world'], \
        "source should be synthetic or real-world"
    
    print("✓ Metadata structure test passed")


def test_metadata_to_markdown():
    """Test DatasetMetadata to_markdown method"""
    gen = DatasetGenerator()
    X, metadata = gen.load_real_world_dataset('gps_tracks')
    
    markdown = metadata.to_markdown()
    
    # Check markdown contains key information
    assert isinstance(markdown, str), "to_markdown should return string"
    assert metadata.name in markdown, "Markdown should contain dataset name"
    assert metadata.description in markdown, "Markdown should contain description"
    assert str(metadata.n_samples) in markdown, "Markdown should contain n_samples"
    assert str(metadata.n_features) in markdown, "Markdown should contain n_features"
    assert str(metadata.expected_clusters) in markdown, "Markdown should contain expected_clusters"
    assert str(metadata.suggested_eps) in markdown, "Markdown should contain suggested_eps"
    assert str(metadata.suggested_minpts) in markdown, "Markdown should contain suggested_minpts"
    assert metadata.difficulty in markdown, "Markdown should contain difficulty"
    assert metadata.source in markdown, "Markdown should contain source"
    
    # Check markdown formatting
    assert '###' in markdown or '##' in markdown, "Markdown should have headers"
    assert '**' in markdown, "Markdown should have bold text"
    
    print("✓ Metadata to_markdown test passed")


def test_all_datasets_have_paper_references():
    """Test that all datasets have paper references"""
    gen = DatasetGenerator()
    datasets = ['gps_tracks', 'customer_locations', 'sensor_readings']
    
    for dataset_name in datasets:
        X, metadata = gen.load_real_world_dataset(dataset_name)
        assert metadata.paper_reference is not None, \
            f"{dataset_name} should have paper_reference"
        assert isinstance(metadata.paper_reference, str), \
            f"{dataset_name} paper_reference should be string"
        assert len(metadata.paper_reference) > 0, \
            f"{dataset_name} paper_reference should not be empty"
        assert 'Section' in metadata.paper_reference, \
            f"{dataset_name} paper_reference should reference a section"
    
    print("✓ Paper references test passed")


def test_dataset_validation():
    """Test that loaded datasets are valid for DBSCAN"""
    gen = DatasetGenerator()
    datasets = ['gps_tracks', 'customer_locations', 'sensor_readings']
    
    for dataset_name in datasets:
        X, metadata = gen.load_real_world_dataset(dataset_name)
        
        # Check minimum samples
        assert X.shape[0] >= 10, f"{dataset_name} should have at least 10 samples"
        
        # Check suggested parameters are valid
        assert metadata.suggested_eps > 0, \
            f"{dataset_name} suggested_eps should be positive"
        assert metadata.suggested_minpts >= 1, \
            f"{dataset_name} suggested_minpts should be >= 1"
        assert metadata.expected_clusters >= 0, \
            f"{dataset_name} expected_clusters should be non-negative"
        
        # Check data range is reasonable (not all zeros)
        assert np.std(X) > 0, f"{dataset_name} should have non-zero variance"
        
        # Check no infinite values
        assert not np.any(np.isinf(X)), f"{dataset_name} should not contain infinite values"
    
    print("✓ Dataset validation test passed")


def test_generate_basic_shapes():
    """Test generate_basic_shapes method"""
    gen = DatasetGenerator()
    
    # Test moons
    X = gen.generate_basic_shapes('moons', n_samples=200, noise_level=0.1)
    assert X.shape == (200, 2), "Moons should have correct shape"
    assert np.issubdtype(X.dtype, np.number), "Moons data should be numeric"
    
    # Test circles
    X = gen.generate_basic_shapes('circles', n_samples=200, noise_level=0.1)
    assert X.shape == (200, 2), "Circles should have correct shape"
    
    # Test blobs
    X = gen.generate_basic_shapes('blobs', n_samples=200, noise_level=0.1)
    assert X.shape == (200, 2), "Blobs should have correct shape"
    
    # Test invalid shape
    try:
        X = gen.generate_basic_shapes('invalid', n_samples=200)
        assert False, "Should raise ValueError for invalid shape"
    except ValueError as e:
        assert 'Unknown shape' in str(e), "Error should mention unknown shape"
    
    print("✓ Generate basic shapes test passed")


def test_generate_varying_density():
    """Test generate_varying_density method"""
    gen = DatasetGenerator()
    
    X = gen.generate_varying_density(n_samples=600, density_ratios=[0.3, 0.6, 1.2])
    
    assert X.shape == (600, 2), "Should have correct shape"
    assert np.issubdtype(X.dtype, np.number), "Data should be numeric"
    assert not np.any(np.isnan(X)), "Data should not contain NaN"
    
    # Test with default density ratios
    X = gen.generate_varying_density(n_samples=600)
    assert X.shape == (600, 2), "Should work with default density ratios"
    
    print("✓ Generate varying density test passed")


def test_generate_spatial_data():
    """Test generate_spatial_data method"""
    gen = DatasetGenerator()
    
    X = gen.generate_spatial_data(bounds=(-10, 10, -10, 10), n_points=500)
    
    assert X.shape == (500, 2), "Should have correct shape"
    assert np.issubdtype(X.dtype, np.number), "Data should be numeric"
    
    # Check bounds are respected
    assert np.all(X[:, 0] >= -10) and np.all(X[:, 0] <= 10), "X coordinates should be within bounds"
    assert np.all(X[:, 1] >= -10) and np.all(X[:, 1] <= 10), "Y coordinates should be within bounds"
    
    print("✓ Generate spatial data test passed")


def test_generate_anomaly_dataset():
    """Test generate_anomaly_dataset method"""
    gen = DatasetGenerator()
    
    X = gen.generate_anomaly_dataset(n_normal=400, n_anomalies=20)
    
    assert X.shape == (420, 2), "Should have correct total samples"
    assert np.issubdtype(X.dtype, np.number), "Data should be numeric"
    assert not np.any(np.isnan(X)), "Data should not contain NaN"
    
    print("✓ Generate anomaly dataset test passed")


def test_dataset_reproducibility():
    """Test that datasets are reproducible with same random_state"""
    gen = DatasetGenerator()
    
    # Test basic shapes reproducibility
    X1 = gen.generate_basic_shapes('moons', n_samples=200, random_state=42)
    X2 = gen.generate_basic_shapes('moons', n_samples=200, random_state=42)
    assert np.allclose(X1, X2), "Same random_state should produce identical results"
    
    # Test varying density reproducibility
    X1 = gen.generate_varying_density(n_samples=600, random_state=42)
    X2 = gen.generate_varying_density(n_samples=600, random_state=42)
    assert np.allclose(X1, X2), "Same random_state should produce identical results"
    
    # Test spatial data reproducibility
    X1 = gen.generate_spatial_data(n_points=500, random_state=42)
    X2 = gen.generate_spatial_data(n_points=500, random_state=42)
    assert np.allclose(X1, X2), "Same random_state should produce identical results"
    
    # Test anomaly dataset reproducibility
    X1 = gen.generate_anomaly_dataset(n_normal=400, n_anomalies=20, random_state=42)
    X2 = gen.generate_anomaly_dataset(n_normal=400, n_anomalies=20, random_state=42)
    assert np.allclose(X1, X2), "Same random_state should produce identical results"
    
    print("✓ Dataset reproducibility test passed")


def test_load_sample_data_invalid_type():
    """Test load_sample_data with invalid dataset type"""
    from src.data_loader import load_sample_data
    
    try:
        load_sample_data(dataset_type="invalid_type")
        assert False, "Should raise ValueError for invalid dataset type"
    except ValueError as e:
        assert "Unknown dataset_type" in str(e)
    
    print("✓ load_sample_data invalid type test passed")


def test_load_sample_data_all_types():
    """Test load_sample_data with all valid dataset types"""
    from src.data_loader import load_sample_data
    
    # Test moons
    X_moons = load_sample_data("moons", n_samples=100, noise=0.1)
    assert X_moons.shape == (100, 2)
    
    # Test circles  
    X_circles = load_sample_data("circles", n_samples=100, noise=0.1)
    assert X_circles.shape == (100, 2)
    
    # Test blobs
    X_blobs = load_sample_data("blobs", n_samples=100, noise=0.1)
    assert X_blobs.shape == (100, 2)
    
    print("✓ load_sample_data all types test passed")


def test_load_spatial_data_structure():
    """Test load_spatial_data generates expected structure"""
    from src.data_loader import load_spatial_data
    
    X = load_spatial_data(n_points=600)
    
    # Should have expected number of points (approximately)
    # 600//2 + 600//3 + 600//6 + 600//10 = 300 + 200 + 100 + 60 = 660
    expected_points = 600//2 + 600//3 + 600//6 + 600//10
    assert X.shape[0] == expected_points
    assert X.shape[1] == 2
    
    print("✓ load_spatial_data structure test passed")


def test_generate_varying_density_edge_cases():
    """Test generate_varying_density with edge cases"""
    generator = DatasetGenerator()
    
    # Test with more clusters than predefined centers - use correct parameter name
    X = generator.generate_varying_density(n_samples=100, density_ratios=[0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7])
    assert X.shape[0] == 100
    assert X.shape[1] == 2
    
    # Test with remaining points handling
    X = generator.generate_varying_density(n_samples=103, density_ratios=[0.3, 0.5, 0.7])  # 103 not divisible by 3
    assert X.shape[0] == 103
    
    print("✓ generate_varying_density edge cases test passed")


def test_load_real_world_dataset_file_not_found():
    """Test load_real_world_dataset with missing files"""
    from src.data_loader import DatasetGenerator
    import tempfile
    import shutil
    from pathlib import Path
    
    generator = DatasetGenerator()
    
    # Temporarily move data files to test FileNotFoundError
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Move files temporarily
        files_to_move = ['gps_tracks.csv', 'customer_locations.csv', 'sensor_readings.csv']
        moved_files = []
        
        for filename in files_to_move:
            src = data_dir / filename
            if src.exists():
                dst = temp_dir / filename
                shutil.move(str(src), str(dst))
                moved_files.append((src, dst))
        
        # Test FileNotFoundError for each dataset
        for dataset_name in ['gps_tracks', 'customer_locations', 'sensor_readings']:
            try:
                generator.load_real_world_dataset(dataset_name)
                assert False, f"Should raise FileNotFoundError for {dataset_name}"
            except FileNotFoundError as e:
                assert "Dataset file not found" in str(e)
        
        print("✓ load_real_world_dataset file not found test passed")
        
    finally:
        # Restore files
        for src, dst in moved_files:
            if dst.exists():
                shutil.move(str(dst), str(src))
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_load_gps_tracks()
    test_load_customer_locations()
    test_load_sensor_readings()
    test_invalid_dataset_name()
    test_metadata_structure()
    test_metadata_to_markdown()
    test_all_datasets_have_paper_references()
    test_dataset_validation()
    test_generate_basic_shapes()
    test_generate_varying_density()
    test_generate_spatial_data()
    test_generate_anomaly_dataset()
    test_dataset_reproducibility()
    test_load_sample_data_invalid_type()
    test_load_sample_data_all_types()
    test_load_spatial_data_structure()
    test_generate_varying_density_edge_cases()
    test_load_real_world_dataset_file_not_found()
    print("\n✓ All data loader tests passed!")
