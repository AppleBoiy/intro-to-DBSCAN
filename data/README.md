# DBSCAN Datasets

This directory contains datasets for demonstrating and learning DBSCAN clustering algorithm. The datasets are organized into two categories: synthetic datasets (generated programmatically) and real-world datasets (CSV files with realistic data patterns).

## Directory Structure

```
data/
├── raw/              # Raw datasets (CSV files)
│   ├── gps_tracks.csv
│   ├── customer_locations.csv
│   └── sensor_readings.csv
├── processed/        # Processed datasets (generated during analysis)
└── README.md         # This file
```

## Real-World Datasets

### GPS Vehicle Tracks

**Description**: GPS trajectory data from 10 vehicles in San Francisco. Demonstrates spatial clustering of vehicle routes with varying densities. Useful for understanding DBSCAN on geographic data.

**Characteristics**:
- Samples: 225
- Features: 2 (latitude, longitude)
- Expected Clusters: 10
- Difficulty: intermediate
- Source: real-world

**Suggested DBSCAN Parameters**:
- eps: 0.003
- min_pts: 3

**Paper Reference**: Section 7: Applications to spatial databases

**File**: `data/raw/gps_tracks.csv`

**Columns**:
- `latitude`: GPS latitude coordinate
- `longitude`: GPS longitude coordinate
- `timestamp`: Time of GPS reading
- `vehicle_id`: Unique vehicle identifier

**Usage Example**:
```python
from src.data_loader import DatasetGenerator

gen = DatasetGenerator()
X, metadata = gen.load_real_world_dataset('gps_tracks')

# Use with DBSCAN
from src.dbscan_from_scratch import DBSCAN
dbscan = DBSCAN(eps=metadata.suggested_eps, min_pts=metadata.suggested_minpts)
labels = dbscan.fit_predict(X)
```

---

### Retail Customer Locations

**Description**: Customer location data from retail stores showing geographic clustering patterns. Includes purchase amounts and visit frequency. Demonstrates market segmentation and location-based clustering.

**Characteristics**:
- Samples: 200
- Features: 2 (latitude, longitude)
- Expected Clusters: 8
- Difficulty: intermediate
- Source: real-world

**Suggested DBSCAN Parameters**:
- eps: 0.005
- min_pts: 4

**Paper Reference**: Section 7: Applications to marketing

**File**: `data/raw/customer_locations.csv`

**Columns**:
- `customer_id`: Unique customer identifier
- `latitude`: Customer location latitude
- `longitude`: Customer location longitude
- `purchase_amount`: Total purchase amount in dollars
- `visit_frequency`: Number of store visits

**Usage Example**:
```python
from src.data_loader import DatasetGenerator

gen = DatasetGenerator()
X, metadata = gen.load_real_world_dataset('customer_locations')

# Use with DBSCAN
from src.dbscan_from_scratch import DBSCAN
dbscan = DBSCAN(eps=metadata.suggested_eps, min_pts=metadata.suggested_minpts)
labels = dbscan.fit_predict(X)
```

**Application**: This dataset is ideal for:
- Market segmentation analysis
- Store location optimization
- Customer behavior clustering
- Geographic market analysis

---

### IoT Sensor Readings

**Description**: Multi-dimensional sensor data from industrial IoT devices. Contains normal operating conditions and anomalous readings. Demonstrates DBSCAN for anomaly detection in time-series data.

**Characteristics**:
- Samples: 185
- Features: 4 (temperature, humidity, pressure, vibration)
- Expected Clusters: 15
- Difficulty: advanced
- Source: real-world

**Suggested DBSCAN Parameters**:
- eps: 2.5
- min_pts: 3

**Paper Reference**: Section 7: Applications to anomaly detection

**File**: `data/raw/sensor_readings.csv`

**Columns**:
- `sensor_id`: Unique sensor identifier
- `timestamp`: Time of sensor reading
- `temperature`: Temperature in Celsius
- `humidity`: Relative humidity percentage
- `pressure`: Atmospheric pressure in hPa
- `vibration`: Vibration level
- `status`: Operational status (normal/anomaly)

**Usage Example**:
```python
from src.data_loader import DatasetGenerator

gen = DatasetGenerator()
X, metadata = gen.load_real_world_dataset('sensor_readings')

# Use with DBSCAN for anomaly detection
from src.dbscan_from_scratch import DBSCAN
dbscan = DBSCAN(eps=metadata.suggested_eps, min_pts=metadata.suggested_minpts)
labels = dbscan.fit_predict(X)

# Points labeled as -1 are anomalies (noise)
anomalies = X[labels == -1]
print(f"Detected {len(anomalies)} anomalies")
```

**Application**: This dataset is ideal for:
- Anomaly detection in sensor networks
- Predictive maintenance
- Quality control monitoring
- Multi-dimensional outlier detection

---

## Synthetic Datasets

The repository also provides synthetic dataset generators through the `DatasetGenerator` class. These are useful for controlled experiments and understanding algorithm behavior.

### Available Synthetic Datasets

1. **Basic Shapes** (`generate_basic_shapes`)
   - Moons: Two interleaving half-circles
   - Circles: Concentric circles
   - Blobs: Gaussian clusters
   - Difficulty: Beginner
   - Use case: Understanding non-convex cluster shapes

2. **Varying Density** (`generate_varying_density`)
   - Multiple clusters with different densities
   - Difficulty: Intermediate
   - Use case: Testing DBSCAN's density-based approach

3. **Spatial Data** (`generate_spatial_data`)
   - Geographic-style spatial patterns
   - Difficulty: Intermediate
   - Use case: Simulating real-world spatial distributions

4. **Anomaly Dataset** (`generate_anomaly_dataset`)
   - Normal clusters with scattered outliers
   - Difficulty: Intermediate
   - Use case: Anomaly detection demonstrations

### Synthetic Dataset Usage

```python
from src.data_loader import DatasetGenerator

gen = DatasetGenerator()

# Generate moons dataset
X_moons = gen.generate_basic_shapes('moons', n_samples=300, noise_level=0.05)

# Generate varying density clusters
X_density = gen.generate_varying_density(n_samples=600, density_ratios=[0.3, 0.6, 1.2])

# Generate spatial data
X_spatial = gen.generate_spatial_data(bounds=(-10, 10, -10, 10), n_points=500)

# Generate anomaly detection dataset
X_anomaly = gen.generate_anomaly_dataset(n_normal=400, n_anomalies=20)
```

---

## Dataset Selection Guide

### By Learning Objective

**Understanding DBSCAN Basics**:
- Start with synthetic datasets (moons, circles)
- Progress to GPS tracks for real-world spatial clustering

**Parameter Tuning Practice**:
- Use varying density synthetic dataset
- Try customer locations with different eps values

**Anomaly Detection**:
- Use anomaly synthetic dataset first
- Progress to sensor readings for multi-dimensional anomaly detection

**Advanced Applications**:
- Sensor readings (high-dimensional)
- Custom datasets with domain-specific characteristics

### By Difficulty Level

**Beginner**:
- Synthetic basic shapes (moons, circles, blobs)
- 2D visualization friendly
- Clear cluster boundaries

**Intermediate**:
- GPS tracks
- Customer locations
- Varying density synthetic data
- Spatial synthetic data

**Advanced**:
- Sensor readings (4D)
- High-dimensional synthetic data
- Complex real-world patterns

---

## Data Quality and Reproducibility

All datasets in this repository are designed for educational purposes with the following characteristics:

1. **Reproducibility**: Synthetic datasets use fixed random seeds (default: 42)
2. **Realistic Patterns**: Real-world datasets exhibit realistic clustering patterns
3. **Documented Parameters**: Each dataset includes suggested DBSCAN parameters
4. **Multiple Difficulty Levels**: Progressive complexity for learning
5. **Paper Alignment**: Datasets align with concepts from the 1996 DBSCAN paper

---

## Adding Custom Datasets

To add your own dataset:

1. Place CSV file in `data/raw/`
2. Add metadata to `DatasetGenerator.load_real_world_dataset()`
3. Update this README with dataset documentation
4. Include suggested DBSCAN parameters based on experimentation

Example metadata structure:
```python
DatasetMetadata(
    name='Your Dataset Name',
    description='Detailed description of the dataset',
    n_samples=1000,
    n_features=2,
    expected_clusters=5,
    suggested_eps=0.5,
    suggested_minpts=5,
    difficulty='intermediate',
    source='real-world',
    paper_reference='Optional paper reference'
)
```

---

## References

- Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In KDD (Vol. 96, No. 34, pp. 226-231).

---

## License

These datasets are provided for educational purposes as part of the comprehensive DBSCAN learning repository.
