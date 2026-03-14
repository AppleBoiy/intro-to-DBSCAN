# DBSCAN Learning Repository

[![codecov](https://codecov.io/gh/AppleBoiy/intro-to-DBSCAN/graph/badge.svg?token=wv2l5Ej5vK)](https://codecov.io/gh/AppleBoiy/intro-to-DBSCAN)
![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A comprehensive, academically rigorous learning resource for mastering DBSCAN (Density-Based Spatial Clustering of Applications with Noise) from fundamentals to advanced applications. Based on the original 1996 KDD paper with complete mathematical foundations, interactive visualizations, and hands-on implementations.

## Why This Repository?

Learn DBSCAN the right way - from theory to practice:

- **Academically Rigorous**: Every concept traces back to the original 1996 paper with proper citations
- **Visually Rich**: Interactive visualizations for every key concept
- **Hands-On**: Implement DBSCAN from scratch with full understanding
- **Comprehensive**: From basic concepts to performance optimization
- **Self-Contained**: Everything you need in one place - no external resources required

## Quick Start (10 Minutes)

```bash
# Clone and setup
git clone https://github.com/AppleBoiy/intro-to-DBSCAN.git
cd dbscan-learning
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Launch notebooks
jupyter notebook notebooks/00_quick_start.ipynb
```

## Learning Paths

Choose your learning journey based on your goals and available time:

### Path 1: Quick Start (1-2 hours) - Beginner

**Perfect for**: Practitioners who need to understand and apply DBSCAN quickly

**You'll learn**:
- What DBSCAN does and how it differs from K-Means
- How to run DBSCAN and visualize results
- The role of eps and min_pts parameters
- When to use DBSCAN for your problem

**Start here**: `notebooks/00_quick_start.ipynb` → `notebooks/01_dbscan_basics.ipynb`

### Path 2: Comprehensive Understanding (8-12 hours) - Intermediate

**Perfect for**: Students and data scientists seeking deep mastery

**You'll learn**:
- Complete mathematical foundations and theory
- Density-reachability and density-connectivity concepts
- Parameter selection using k-distance graphs
- Algorithm implementation from scratch
- Comparison with other clustering algorithms
- Real-world applications (spatial clustering, anomaly detection)

**Start here**: `docs/learning_roadmap.md` → Follow Path 2 sequence

### Path 3: Advanced Topics (4-6 hours) - Advanced

**Perfect for**: Researchers and engineers optimizing for production

**You'll learn**:
- Time and space complexity analysis with proofs
- Spatial indexing optimizations (R-tree, KD-tree)
- Performance benchmarking and profiling
- Scaling to large datasets (100K+ points)
- Memory optimization techniques

**Prerequisites**: Complete Path 2 first

**Start here**: `docs/05_complexity_analysis.md` → `notebooks/10_performance_analysis.ipynb`

See `docs/learning_roadmap.md` for detailed learning paths with time estimates and checkpoints.

## Installation

### Requirements

- Python 3.8 or higher
- 2GB RAM minimum (4GB recommended for large datasets)
- Jupyter Notebook for interactive learning

### Setup

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from src.dbscan_from_scratch import DBSCAN; print('Installation successful!')"
```

## Project Structure

```
dbscan-learning/
├── docs/                          # Comprehensive documentation
│   ├── 00_how_to_read_the_paper.md    # Paper reading guide
│   ├── 01_theory_and_math.md          # Mathematical foundations
│   ├── 02_density_concepts.md         # Density-reachability & connectivity
│   ├── 03_algorithm_details.md        # Algorithm walkthrough
│   ├── 04_parameter_tuning.md         # Parameter selection guide
│   ├── 05_complexity_analysis.md      # Time/space complexity
│   ├── 06_algorithm_comparison.md     # DBSCAN vs other algorithms
│   ├── 07_applications.md             # Real-world use cases
│   ├── 08_performance_optimization.md # Optimization techniques
│   ├── glossary.md                    # Technical terms
│   ├── learning_roadmap.md            # Structured learning paths
│   └── references.md                  # Paper citations
│
├── notebooks/                     # Interactive Jupyter notebooks
│   ├── 00_quick_start.ipynb           # 10-minute introduction
│   ├── 01_dbscan_basics.ipynb         # Core concepts
│   ├── 02_mathematical_foundations.ipynb  # Formal definitions
│   ├── 03_epsilon_neighborhood.ipynb  # Interactive ε exploration
│   ├── 04_density_concepts.ipynb      # Density-reachability visualization
│   ├── 05_algorithm_walkthrough.ipynb # Step-by-step execution
│   ├── 06_parameter_tuning.ipynb      # K-distance graphs & tuning
│   ├── 07_comparing_algorithms.ipynb  # DBSCAN vs K-Means/OPTICS
│   ├── 08_spatial_clustering.ipynb    # GPS & geographic data
│   ├── 09_anomaly_detection.ipynb     # Outlier detection
│   ├── 10_performance_analysis.ipynb  # Benchmarking & profiling
│   └── 11_advanced_topics.ipynb       # Spatial indexing & optimization
│
├── src/                           # Source code implementation
│   ├── dbscan_from_scratch.py         # Core DBSCAN implementation
│   ├── parameter_tuning.py            # Parameter selection tools
│   ├── visualization.py               # Visualization utilities
│   └── data_loader.py                 # Dataset generation & loading
│
├── data/                          # Datasets for experiments
│   ├── raw/                           # Real-world datasets
│   │   ├── gps_tracks.csv             # GPS trajectory data
│   │   ├── customer_locations.csv     # Retail clustering data
│   │   └── sensor_readings.csv        # IoT anomaly data
│   └── README.md                      # Dataset documentation
│
├── tests/                         # Comprehensive test suite
│   ├── test_dbscan.py                 # Core algorithm tests
│   ├── test_properties.py             # Property-based tests
│   ├── test_sklearn_compatibility.py  # Sklearn equivalence tests
│   ├── test_edge_cases.py             # Edge case handling
│   └── test_performance.py            # Performance benchmarks
│
├── 1996-DBSCAN-KDD.pdf           # Original DBSCAN paper
└── requirements.txt               # Python dependencies
```

## Usage Examples

### Basic Clustering

```python
from src.dbscan_from_scratch import DBSCAN
from src.data_loader import DatasetGenerator
from src.visualization import DBSCANVisualizer

# Generate sample data
generator = DatasetGenerator(random_state=42)
X = generator.generate_basic_shapes('moons', n_samples=300, noise=0.05)

# Run DBSCAN
dbscan = DBSCAN(eps=0.3, min_pts=5)
labels = dbscan.fit_predict(X)

# Visualize results
visualizer = DBSCANVisualizer()
visualizer.plot_clusters(X, labels, title='DBSCAN Clustering Results')
```

### Parameter Tuning

```python
from src.parameter_tuning import ParameterSelector

# Automatic parameter selection
selector = ParameterSelector()
suggested_params = selector.suggest_parameters(X)
print(f"Suggested eps: {suggested_params['eps']:.3f}")
print(f"Suggested min_pts: {suggested_params['min_pts']}")

# Visualize k-distance graph
selector.plot_k_distance_graph(X, k=5, show_elbow=True)
```

### Visualizing Concepts

```python
# Visualize epsilon neighborhoods
visualizer.plot_epsilon_neighborhood(X, point_idx=10, eps=0.3)

# Show point types (core, border, noise)
visualizer.plot_point_types(X, dbscan)

# Visualize density-reachability
visualizer.plot_density_reachability(X, dbscan, start_point=5)
```

### Real-World Applications

```python
from src.data_loader import load_real_world_dataset

# Load GPS trajectory data
gps_data, metadata = load_real_world_dataset('gps_tracks')

# Apply DBSCAN with suggested parameters
dbscan = DBSCAN(
    eps=metadata['suggested_eps'],
    min_pts=metadata['suggested_min_pts']
)
clusters = dbscan.fit_predict(gps_data)

# Identify anomalies (noise points)
anomalies = gps_data[clusters == -1]
print(f"Found {len(anomalies)} anomalous GPS points")
```

### Performance Benchmarking

```python
from src.dbscan_from_scratch import DBSCAN
import time

# Benchmark on different dataset sizes
for n in [100, 500, 1000, 5000]:
    X = generator.generate_basic_shapes('blobs', n_samples=n)
    
    start = time.time()
    dbscan = DBSCAN(eps=0.5, min_pts=5)
    labels = dbscan.fit_predict(X)
    elapsed = time.time() - start
    
    print(f"n={n}: {elapsed:.3f}s")
```

## Learning Outcomes

After completing this repository, you will be able to:

### Foundational Understanding
- ✓ Explain DBSCAN's density-based clustering approach
- ✓ Define and distinguish core, border, and noise points
- ✓ Understand epsilon-neighborhoods and density-reachability
- ✓ Trace concepts back to the original 1996 paper

### Mathematical Mastery
- ✓ Work with formal definitions and mathematical notation
- ✓ Understand density-connectivity and its properties
- ✓ Analyze time and space complexity (O(n²) naive, O(n log n) optimized)
- ✓ Prove correctness properties of the algorithm

### Practical Skills
- ✓ Implement DBSCAN from scratch with full understanding
- ✓ Select optimal parameters using k-distance graphs
- ✓ Apply DBSCAN to spatial data and anomaly detection
- ✓ Compare DBSCAN with K-Means, OPTICS, and hierarchical clustering
- ✓ Diagnose and fix poor clustering results

### Advanced Techniques
- ✓ Optimize performance using spatial indexing (R-tree, KD-tree)
- ✓ Scale DBSCAN to large datasets (100K+ points)
- ✓ Profile and benchmark clustering performance
- ✓ Understand memory usage and optimization strategies

## Documentation

### Core Documentation
- **[Learning Roadmap](docs/learning_roadmap.md)**: Structured learning paths with time estimates
- **[Theory & Math](docs/01_theory_and_math.md)**: Mathematical foundations with paper citations
- **[Density Concepts](docs/02_density_concepts.md)**: Density-reachability and connectivity
- **[Algorithm Details](docs/03_algorithm_details.md)**: Complete algorithm walkthrough
- **[Parameter Tuning](docs/04_parameter_tuning.md)**: K-distance graphs and parameter selection

### Advanced Topics
- **[Complexity Analysis](docs/05_complexity_analysis.md)**: Time and space complexity proofs
- **[Algorithm Comparison](docs/06_algorithm_comparison.md)**: DBSCAN vs other algorithms
- **[Applications](docs/07_applications.md)**: Real-world use cases and strategies
- **[Performance Optimization](docs/08_performance_optimization.md)**: Spatial indexing and scaling

### Reference Materials
- **[Glossary](docs/glossary.md)**: Technical term definitions
- **[References](docs/references.md)**: Complete paper citations
- **[How to Read the Paper](docs/00_how_to_read_the_paper.md)**: Guide to the 1996 paper

## Interactive Notebooks

All notebooks include:
- Learning objectives and prerequisites
- Paper citations for every concept
- Interactive visualizations with parameter controls
- Exercises with solutions
- Summary and next steps

Launch the notebook server:
```bash
jupyter notebook notebooks/
```

Start with `00_quick_start.ipynb` for a 10-minute introduction.

## Testing

The repository includes comprehensive tests ensuring correctness:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m "not slow"  # Fast tests only
pytest tests/test_properties.py  # Property-based tests
pytest tests/test_sklearn_compatibility.py  # Sklearn equivalence
```

Test coverage: 90%+ across all modules

## Contributing

Contributions are welcome! Areas for contribution:
- Additional real-world datasets
- New visualization techniques
- Performance optimizations
- Additional language implementations
- Documentation improvements

Please open an issue to discuss major changes before submitting a pull request.

## Citation

If you use this repository for academic work, please cite the original DBSCAN paper:

```bibtex
@inproceedings{ester1996density,
  title={A density-based algorithm for discovering clusters in large spatial databases with noise},
  author={Ester, Martin and Kriegel, Hans-Peter and Sander, J{\"o}rg and Xu, Xiaowei},
  booktitle={Kdd},
  volume={96},
  number={34},
  pages={226--231},
  year={1996}
}
```

## License

MIT License with Research Attribution - see [LICENSE](LICENSE) file for details

### Terms of Use

This repository is provided for educational and research purposes with the following terms:

#### Educational Use
- **Primary Purpose**: Learning and understanding the DBSCAN algorithm
- **Academic Integrity**: Users must follow their institution's academic integrity policies
- **Citation Required**: Please cite the original DBSCAN paper (Ester et al., 1996) when using this material academically
- **No Plagiarism**: Do not present this work as your own original research

#### Research Attribution
- **Original Algorithm**: Based on the 1996 DBSCAN paper by Ester, Kriegel, Sander, and Xu
- **Research Materials**: Some datasets and methodologies may derive from biolab.si and other academic sources
- **Verification Required**: Users should verify original sources for specific datasets used in academic work
- **Proper Citation**: Acknowledge this repository if it significantly contributes to your research

#### Data Usage
- **Dataset Sources**: Real-world datasets may have specific attribution requirements
- **User Responsibility**: Verify licensing terms and original sources of datasets independently
- **Synthetic Data**: Some datasets are synthetic or derived from publicly available sources
- **Contact for Clarification**: Reach out to maintainers when uncertain about data sources

#### Commercial Use
- **Permitted**: Commercial use allowed under MIT license terms
- **No Warranty**: No warranty provided for commercial applications
- **Independent Verification**: Users must verify dataset licensing for commercial use

#### Modifications
- **Derivative Works**: Permitted under MIT license terms
- **Attribution Maintained**: Keep attribution to original sources in derivative works
- **Community Contribution**: Consider contributing improvements back to the project

For complete terms and conditions, see the [LICENSE](LICENSE) file.

## Acknowledgments

- Original DBSCAN paper by Ester et al. (1996)
- Scikit-learn for reference implementation
- The clustering research community

## Support

- **Issues**: Open a GitHub issue for bugs or questions
- **Discussions**: Use GitHub Discussions for general questions
- **Documentation**: Check `docs/learning_roadmap.md` for guidance

---

**Ready to start learning?** Choose your path in the [Learning Roadmap](docs/learning_roadmap.md) and begin your DBSCAN journey!
