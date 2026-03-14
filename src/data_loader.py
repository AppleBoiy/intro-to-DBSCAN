"""
Data Loading and Generation Functions
Functions for loading and generating test datasets
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from pathlib import Path
from sklearn.datasets import make_moons, make_blobs, make_circles


@dataclass
class DatasetMetadata:
    """
    Metadata for datasets used in examples
    
    Attributes:
    -----------
    name : str
        Dataset name
    description : str
        Description of the dataset and its characteristics
    n_samples : int
        Number of data points in the dataset
    n_features : int
        Number of features (dimensions)
    expected_clusters : int
        Expected number of clusters (excluding noise)
    suggested_eps : float
        Suggested epsilon parameter for DBSCAN
    suggested_minpts : int
        Suggested MinPts parameter for DBSCAN
    difficulty : str
        Difficulty level: "beginner", "intermediate", or "advanced"
    source : str
        Source type: "synthetic" or "real-world"
    paper_reference : Optional[str]
        Reference to paper section if applicable
    """
    name: str
    description: str
    n_samples: int
    n_features: int
    expected_clusters: int
    suggested_eps: float
    suggested_minpts: int
    difficulty: str
    source: str
    paper_reference: Optional[str] = None
    
    def to_markdown(self) -> str:
        """
        Generate markdown documentation for dataset
        
        Returns:
        --------
        str
            Markdown formatted dataset documentation
        """
        md = f"### {self.name}\n\n"
        md += f"**Description**: {self.description}\n\n"
        md += f"**Characteristics**:\n"
        md += f"- Samples: {self.n_samples}\n"
        md += f"- Features: {self.n_features}\n"
        md += f"- Expected Clusters: {self.expected_clusters}\n"
        md += f"- Difficulty: {self.difficulty}\n"
        md += f"- Source: {self.source}\n\n"
        md += f"**Suggested DBSCAN Parameters**:\n"
        md += f"- eps: {self.suggested_eps}\n"
        md += f"- min_pts: {self.suggested_minpts}\n\n"
        if self.paper_reference:
            md += f"**Paper Reference**: {self.paper_reference}\n\n"
        return md


def load_sample_data(dataset_type: str = "moons", n_samples: int = 300, noise: float = 0.05):
    """
    Generate sample data for testing DBSCAN
    
    Parameters:
    -----------
    dataset_type : str
        Type of dataset: "moons", "circles", "blobs"
    n_samples : int
        Number of data points
    noise : float
        Noise level
    
    Returns:
    --------
    X : np.ndarray
        Data with shape (n_samples, 2)
    """
    if dataset_type == "moons":
        X, _ = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    elif dataset_type == "circles":
        X, _ = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
    elif dataset_type == "blobs":
        X, _ = make_blobs(n_samples=n_samples, centers=3, cluster_std=0.5, random_state=42)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    return X


def load_spatial_data(n_points: int = 500):
    """
    Generate synthetic spatial data
    Suitable for testing DBSCAN with varying density clusters
    """
    # Create 3 clusters with different densities
    cluster1 = np.random.randn(n_points // 2, 2) * 0.3 + [0, 0]
    cluster2 = np.random.randn(n_points // 3, 2) * 0.5 + [3, 3]
    cluster3 = np.random.randn(n_points // 6, 2) * 0.2 + [0, 3]
    
    # Add noise points
    noise = np.random.uniform(-2, 5, (n_points // 10, 2))
    
    X = np.vstack([cluster1, cluster2, cluster3, noise])
    return X


class DatasetGenerator:
    """
    Generate and load datasets for DBSCAN experimentation
    
    This class provides methods to generate various types of synthetic datasets
    for testing and demonstrating DBSCAN clustering capabilities.
    """
    
    @staticmethod
    def generate_basic_shapes(shape: str, n_samples: int = 300, 
                             noise_level: float = 0.05, 
                             random_state: int = 42) -> np.ndarray:
        """
        Generate standard test datasets (moons, circles, blobs)
        
        Parameters:
        -----------
        shape : str
            Type of dataset: "moons", "circles", or "blobs"
        n_samples : int, default=300
            Number of data points to generate
        noise_level : float, default=0.05
            Standard deviation of Gaussian noise added to the data
        random_state : int, default=42
            Random seed for reproducibility
        
        Returns:
        --------
        X : np.ndarray
            Generated data with shape (n_samples, 2)
        
        Examples:
        ---------
        >>> gen = DatasetGenerator()
        >>> X = gen.generate_basic_shapes('moons', n_samples=200, noise_level=0.1)
        >>> X.shape
        (200, 2)
        """
        if shape == "moons":
            X, _ = make_moons(n_samples=n_samples, noise=noise_level, 
                            random_state=random_state)
        elif shape == "circles":
            X, _ = make_circles(n_samples=n_samples, noise=noise_level, 
                              factor=0.5, random_state=random_state)
        elif shape == "blobs":
            X, _ = make_blobs(n_samples=n_samples, centers=3, 
                            cluster_std=noise_level * 10, 
                            random_state=random_state)
        else:
            raise ValueError(
                f"Unknown shape: {shape}. "
                f"Valid options are: 'moons', 'circles', 'blobs'"
            )
        
        return X
    
    @staticmethod
    def generate_varying_density(n_samples: int = 600, 
                                 density_ratios: List[float] = None,
                                 random_state: int = 42) -> np.ndarray:
        """
        Generate clusters with different densities
        
        This dataset tests DBSCAN's ability to handle clusters with varying
        density levels, which is a key advantage over algorithms like K-Means.
        
        Parameters:
        -----------
        n_samples : int, default=600
            Total number of data points to generate
        density_ratios : List[float], optional
            List of standard deviations for each cluster. Higher values create
            lower density clusters. If None, uses [0.3, 0.6, 1.2] for three
            clusters with progressively lower density.
        random_state : int, default=42
            Random seed for reproducibility
        
        Returns:
        --------
        X : np.ndarray
            Generated data with shape (n_samples, 2)
        
        Examples:
        ---------
        >>> gen = DatasetGenerator()
        >>> X = gen.generate_varying_density(n_samples=900, 
        ...                                   density_ratios=[0.2, 0.5, 1.0])
        >>> X.shape
        (900, 2)
        """
        if density_ratios is None:
            density_ratios = [0.3, 0.6, 1.2]
        
        np.random.seed(random_state)
        
        n_clusters = len(density_ratios)
        points_per_cluster = n_samples // n_clusters
        
        clusters = []
        centers = [
            [0, 0],
            [5, 5],
            [0, 5]
        ]
        
        # Ensure we have enough centers
        if n_clusters > len(centers):
            # Generate additional centers
            for i in range(n_clusters - len(centers)):
                centers.append([i * 5, (i + 1) * 5])
        
        for i, std in enumerate(density_ratios):
            cluster = np.random.randn(points_per_cluster, 2) * std + centers[i]
            clusters.append(cluster)
        
        # Handle remaining points (if n_samples not evenly divisible)
        remaining = n_samples - (points_per_cluster * n_clusters)
        if remaining > 0:
            extra_cluster = np.random.randn(remaining, 2) * density_ratios[0] + centers[0]
            clusters.append(extra_cluster)
        
        X = np.vstack(clusters)
        
        # Shuffle the data
        indices = np.random.permutation(len(X))
        X = X[indices]
        
        return X
    
    @staticmethod
    def generate_spatial_data(bounds: Tuple[float, float, float, float] = (-10, 10, -10, 10),
                             n_points: int = 500,
                             random_state: int = 42) -> np.ndarray:
        """
        Generate geographic-style spatial data
        
        Creates data that mimics real-world spatial patterns such as GPS
        coordinates, with clustered regions and scattered points.
        
        Parameters:
        -----------
        bounds : Tuple[float, float, float, float], default=(-10, 10, -10, 10)
            Spatial bounds as (min_x, max_x, min_y, max_y)
        n_points : int, default=500
            Total number of points to generate
        random_state : int, default=42
            Random seed for reproducibility
        
        Returns:
        --------
        X : np.ndarray
            Generated spatial data with shape (n_points, 2)
        
        Examples:
        ---------
        >>> gen = DatasetGenerator()
        >>> X = gen.generate_spatial_data(bounds=(0, 100, 0, 100), n_points=1000)
        >>> X.shape
        (1000, 2)
        """
        np.random.seed(random_state)
        
        min_x, max_x, min_y, max_y = bounds
        
        # Create 3 clusters with different densities (70% of points)
        n_clustered = int(n_points * 0.7)
        cluster1 = np.random.randn(n_clustered // 2, 2) * 0.5 + [
            (min_x + max_x) * 0.25, (min_y + max_y) * 0.25
        ]
        cluster2 = np.random.randn(n_clustered // 3, 2) * 0.8 + [
            (min_x + max_x) * 0.75, (min_y + max_y) * 0.75
        ]
        cluster3 = np.random.randn(n_clustered // 6, 2) * 0.3 + [
            (min_x + max_x) * 0.25, (min_y + max_y) * 0.75
        ]
        
        # Add uniformly distributed noise points (30% of points)
        n_noise = n_points - (len(cluster1) + len(cluster2) + len(cluster3))
        noise = np.random.uniform(
            [min_x, min_y], 
            [max_x, max_y], 
            (n_noise, 2)
        )
        
        X = np.vstack([cluster1, cluster2, cluster3, noise])
        
        # Clip to bounds
        X[:, 0] = np.clip(X[:, 0], min_x, max_x)
        X[:, 1] = np.clip(X[:, 1], min_y, max_y)
        
        # Shuffle the data
        indices = np.random.permutation(len(X))
        X = X[indices]
        
        return X
    
    @staticmethod
    def generate_anomaly_dataset(n_normal: int = 400, 
                                n_anomalies: int = 20,
                                random_state: int = 42) -> np.ndarray:
        """
        Generate dataset for anomaly detection demonstration
        
        Creates a dataset with a main cluster of "normal" points and scattered
        "anomaly" points, useful for demonstrating DBSCAN's noise detection
        capabilities.
        
        Parameters:
        -----------
        n_normal : int, default=400
            Number of normal (clustered) points
        n_anomalies : int, default=20
            Number of anomaly (outlier) points
        random_state : int, default=42
            Random seed for reproducibility
        
        Returns:
        --------
        X : np.ndarray
            Generated data with shape (n_normal + n_anomalies, 2)
            First n_normal points are normal, last n_anomalies are anomalies
        
        Examples:
        ---------
        >>> gen = DatasetGenerator()
        >>> X = gen.generate_anomaly_dataset(n_normal=500, n_anomalies=30)
        >>> X.shape
        (530, 2)
        """
        np.random.seed(random_state)
        
        # Generate normal points in 2-3 tight clusters
        n_cluster1 = int(n_normal * 0.6)
        n_cluster2 = n_normal - n_cluster1
        
        cluster1 = np.random.randn(n_cluster1, 2) * 0.4 + [0, 0]
        cluster2 = np.random.randn(n_cluster2, 2) * 0.3 + [3, 3]
        
        normal_points = np.vstack([cluster1, cluster2])
        
        # Generate anomaly points scattered far from clusters
        # Use a wider uniform distribution
        anomalies = np.random.uniform(-5, 8, (n_anomalies, 2))
        
        # Ensure anomalies are not too close to normal clusters
        # by rejecting points within a certain distance
        min_distance = 2.0
        filtered_anomalies = []
        
        for anomaly in anomalies:
            distances = np.linalg.norm(normal_points - anomaly, axis=1)
            if np.min(distances) > min_distance:
                filtered_anomalies.append(anomaly)
        
        # If we filtered out too many, add some guaranteed outliers
        while len(filtered_anomalies) < n_anomalies:
            # Add points at the corners
            corner_points = [
                [-5, -5], [8, 8], [-5, 8], [8, -5],
                [-4, 6], [7, -4], [-3, 7], [6, -3]
            ]
            for corner in corner_points:
                if len(filtered_anomalies) < n_anomalies:
                    noise = np.random.randn(2) * 0.2
                    filtered_anomalies.append(corner + noise)
        
        anomalies = np.array(filtered_anomalies[:n_anomalies])
        
        # Combine normal and anomaly points
        X = np.vstack([normal_points, anomalies])
        
        # Shuffle the data
        indices = np.random.permutation(len(X))
        X = X[indices]
        
        return X

    @staticmethod
    def load_real_world_dataset(name: str) -> Tuple[np.ndarray, DatasetMetadata]:
        """
        Load real-world datasets with metadata
        
        Available datasets:
        - 'gps_tracks': GPS trajectory data from vehicle tracking
        - 'customer_locations': Retail customer clustering data
        - 'sensor_readings': IoT sensor anomaly detection data
        
        Parameters:
        -----------
        name : str
            Name of the dataset to load
        
        Returns:
        --------
        X : np.ndarray
            Dataset as numpy array with shape (n_samples, n_features)
        metadata : DatasetMetadata
            Metadata object containing dataset information and suggested parameters
        
        Raises:
        -------
        ValueError
            If dataset name is not recognized
        FileNotFoundError
            If dataset file is not found
        
        Examples:
        ---------
        >>> gen = DatasetGenerator()
        >>> X, metadata = gen.load_real_world_dataset('gps_tracks')
        >>> print(f"Loaded {metadata.name}: {metadata.description}")
        >>> print(f"Suggested eps: {metadata.suggested_eps}")
        """
        # Define dataset metadata
        datasets_info = {
            'gps_tracks': DatasetMetadata(
                name='GPS Vehicle Tracks',
                description='GPS trajectory data from 10 vehicles in San Francisco. '
                           'Demonstrates spatial clustering of vehicle routes with varying '
                           'densities. Useful for understanding DBSCAN on geographic data.',
                n_samples=225,
                n_features=2,
                expected_clusters=10,
                suggested_eps=0.003,
                suggested_minpts=3,
                difficulty='intermediate',
                source='real-world',
                paper_reference='Section 7: Applications to spatial databases'
            ),
            'customer_locations': DatasetMetadata(
                name='Retail Customer Locations',
                description='Customer location data from retail stores showing geographic '
                           'clustering patterns. Includes purchase amounts and visit frequency. '
                           'Demonstrates market segmentation and location-based clustering.',
                n_samples=200,
                n_features=2,
                expected_clusters=8,
                suggested_eps=0.005,
                suggested_minpts=4,
                difficulty='intermediate',
                source='real-world',
                paper_reference='Section 7: Applications to marketing'
            ),
            'sensor_readings': DatasetMetadata(
                name='IoT Sensor Readings',
                description='Multi-dimensional sensor data from industrial IoT devices. '
                           'Contains normal operating conditions and anomalous readings. '
                           'Demonstrates DBSCAN for anomaly detection in time-series data.',
                n_samples=185,
                n_features=4,
                expected_clusters=15,
                suggested_eps=2.5,
                suggested_minpts=3,
                difficulty='advanced',
                source='real-world',
                paper_reference='Section 7: Applications to anomaly detection'
            )
        }
        
        if name not in datasets_info:
            available = ', '.join(datasets_info.keys())
            raise ValueError(
                f"Unknown dataset: '{name}'. "
                f"Available datasets: {available}"
            )
        
        # Load the dataset
        data_dir = Path(__file__).parent.parent / 'data' / 'raw'
        
        if name == 'gps_tracks':
            file_path = data_dir / 'gps_tracks.csv'
            if not file_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
            df = pd.read_csv(file_path)
            # Extract latitude and longitude as features
            X = df[['latitude', 'longitude']].values
            
        elif name == 'customer_locations':
            file_path = data_dir / 'customer_locations.csv'
            if not file_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
            df = pd.read_csv(file_path)
            # Extract latitude and longitude as features
            X = df[['latitude', 'longitude']].values
            
        elif name == 'sensor_readings':
            file_path = data_dir / 'sensor_readings.csv'
            if not file_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
            df = pd.read_csv(file_path)
            # Extract sensor measurements as features (temperature, humidity, pressure, vibration)
            X = df[['temperature', 'humidity', 'pressure', 'vibration']].values
        
        metadata = datasets_info[name]
        
        return X, metadata
