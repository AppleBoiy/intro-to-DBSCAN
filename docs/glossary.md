# Glossary of Technical Terms

> **Purpose**: This glossary provides definitions for all technical terms used throughout the DBSCAN learning repository. Each term includes its formal definition, intuitive explanation, and references to the original paper where applicable.

## Table of Contents

- [Algorithm Terms](#algorithm-terms)
- [Point Classification](#point-classification)
- [Density Concepts](#density-concepts)
- [Parameters](#parameters)
- [Data Structures](#data-structures)
- [Complexity Terms](#complexity-terms)
- [Comparison Terms](#comparison-terms)

---

## Algorithm Terms

### DBSCAN
**Full Name**: Density-Based Spatial Clustering of Applications with Noise

**Definition** [Paper §1, p. 226]: A density-based clustering algorithm that discovers clusters of arbitrary shapes and identifies noise points based on local density.

**Intuition**: An algorithm that groups together points that are closely packed (high density) and marks points in low-density regions as outliers.

**Key Properties**:
- Discovers clusters without requiring the number of clusters as input
- Can find arbitrarily shaped clusters
- Explicitly identifies noise/outliers
- Deterministic (same input always produces same output)

---

### Cluster
**Definition** [Paper §4.1, p. 227]: A non-empty subset C of the database D satisfying:
1. **Maximality**: For all p, q: if p ∈ C and q is density-reachable from p, then q ∈ C
2. **Connectivity**: For all p, q ∈ C: p is density-connected to q

**Intuition**: A group of points that are all density-connected to each other, forming a dense region in the data space.

**Properties**:
- Can have arbitrary shapes (not limited to spherical)
- Defined by density, not geometric boundaries
- Contains both core points and border points

---

### Clustering
**Definition**: The task of grouping a set of objects such that objects in the same group (cluster) are more similar to each other than to those in other groups.

**In DBSCAN Context**: Partitioning the dataset into clusters based on density-connectivity, with some points marked as noise.

---

### Deterministic Algorithm
**Definition**: An algorithm that, given the same input, always produces the same output.

**DBSCAN Property**: DBSCAN is deterministic because it processes points in a fixed order and uses deterministic rules for cluster assignment (unlike K-Means, which uses random initialization).

---

## Point Classification

### Core Point
**Definition** [Paper §4.1, p. 227]: A point p is a core point if its ε-neighborhood contains at least MinPts points:
```
|N_ε(p)| ≥ MinPts
```

**Intuition**: A point in a dense region that has enough neighbors to be considered part of a cluster's "core."

**Visual**: Typically shown as large filled circles in visualizations.

**Role**: Core points initiate cluster formation and allow clusters to expand.

---

### Border Point
**Definition** [Paper §4.1, p. 227]: A point that is not a core point but lies in the ε-neighborhood of at least one core point.

**Intuition**: A point on the edge of a cluster—close enough to be included but not dense enough to be core.

**Visual**: Typically shown as medium filled circles in visualizations.

**Properties**:
- Has fewer than MinPts neighbors
- Belongs to a cluster but cannot expand it
- May be assigned arbitrarily if between two clusters

---

### Noise Point (Outlier)
**Definition** [Paper §4.1, p. 227]: A point that is neither a core point nor a border point.

**Intuition**: An isolated point in a low-density region that doesn't belong to any cluster.

**Visual**: Typically shown as 'x' markers or small dots in visualizations.

**Label**: Assigned label -1 in most implementations.

**Properties**:
- Has fewer than MinPts neighbors
- Not in any core point's ε-neighborhood
- Represents anomalies or outliers in the data

---

### Point Type
**Definition**: The classification of a point as core, border, or noise based on its local density and relationship to other points.

**Determination**:
1. If |N_ε(p)| ≥ MinPts → Core point
2. Else if p ∈ N_ε(q) for some core point q → Border point
3. Else → Noise point

---

## Density Concepts

### ε-neighborhood (Epsilon-neighborhood)
**Definition** [Paper §4.1, p. 227]: The ε-neighborhood of a point p, denoted N_ε(p), is the set of all points within distance ε from p:
```
N_ε(p) = {q ∈ D | dist(p, q) ≤ ε}
```

**Intuition**: All points within a circle (in 2D) or sphere (in higher dimensions) of radius ε centered at point p.

**Note**: The point p is included in its own ε-neighborhood.

**Visualization**: Often shown as a dashed circle around a point.

---

### Directly Density-Reachable
**Definition** [Paper §4.1, p. 227]: A point q is directly density-reachable from point p if:
1. q ∈ N_ε(p) (q is in p's ε-neighborhood)
2. |N_ε(p)| ≥ MinPts (p is a core point)

**Intuition**: You can reach q from p in one "step" if p is a core point and q is close enough.

**Properties**:
- Not symmetric: q may be directly reachable from p, but p may not be directly reachable from q
- Requires p to be a core point
- Distance between p and q must be ≤ ε

---

### Density-Reachable
**Definition** [Paper §4.1, p. 227]: A point p is density-reachable from point q if there exists a chain of points p₁, p₂, ..., pₙ where:
1. p₁ = q
2. pₙ = p
3. pᵢ₊₁ is directly density-reachable from pᵢ for all i

**Intuition**: You can reach p from q by "hopping" through a chain of core points, where each hop is within distance ε.

**Properties**:
- Transitive: if p is reachable from q, and q from r, then p is reachable from r
- Not symmetric: p may be reachable from q, but q may not be reachable from p
- Allows clusters to extend beyond single ε-neighborhoods

**Mathematical Notation**:
```
p is density-reachable from q ⟺ ∃ chain q = p₁ → p₂ → ... → pₙ = p
```

---

### Density-Connected
**Definition** [Paper §4.1, p. 227]: Two points p and q are density-connected if there exists a point o such that both p and q are density-reachable from o.

**Intuition**: Two points are density-connected if they can both be reached from some common core point.

**Properties**:
- Symmetric: if p is density-connected to q, then q is density-connected to p
- Transitive: if p is connected to q, and q to r, then p is connected to r
- Defines cluster membership (all points in a cluster are density-connected)

**Mathematical Notation**:
```
p and q are density-connected ⟺ ∃ o : (p ←― o ―→ q)
```

---

### Density
**Definition**: The number of points within a given radius (ε) of a point, or more generally, the concentration of points in a region.

**In DBSCAN**: Measured by the size of ε-neighborhoods. High density means many points within ε; low density means few points within ε.

**Importance**: DBSCAN defines clusters as dense regions separated by sparse regions.

---

## Parameters

### ε (Epsilon)
**Definition** [Paper §4.1, p. 227]: The maximum radius of the neighborhood around a point. Defines what "nearby" means for the dataset.

**Type**: Positive real number (ε > 0)

**Units**: Same as the distance metric used (e.g., meters for geographic data)

**Selection**: Use k-distance graph method [Paper §5.1, p. 229]

**Effect**:
- Too small: Many points become noise, clusters fragment
- Too large: Clusters merge together, fewer noise points
- Optimal: Separates dense regions from sparse regions

---

### MinPts
**Definition** [Paper §4.1, p. 227]: The minimum number of points required to form a dense region (i.e., to be a core point).

**Type**: Positive integer (MinPts ≥ 1)

**Heuristic** [Paper §5.2, p. 229]: MinPts ≥ dimensionality + 1

**Common Values**:
- 2D data: MinPts = 4 or 5
- Higher dimensions: MinPts = 2 × dimensionality
- Noisy data: Use higher MinPts

**Effect**:
- Too small: More noise points become core points, sensitive to noise
- Too large: Smaller clusters may be marked as noise
- Optimal: Balances cluster detection and noise robustness

---

### Distance Metric
**Definition**: A function that defines the distance between two points in the data space.

**Common Metrics**:

**Euclidean Distance** (L2 norm) [Paper §4.1, p. 227]:
```
d(p, q) = √(Σᵢ(pᵢ - qᵢ)²)
```

**Manhattan Distance** (L1 norm):
```
d(p, q) = Σᵢ|pᵢ - qᵢ|
```

**Chebyshev Distance** (L∞ norm):
```
d(p, q) = maxᵢ|pᵢ - qᵢ|
```

**Choice**: Depends on data characteristics and domain requirements.

---

## Data Structures

### Database (Dataset)
**Definition** [Paper §4.1, p. 227]: The set D of all data points to be clustered.

**Notation**: D = {p₁, p₂, ..., pₙ} where n is the number of points

**Properties**:
- Each point pᵢ is a vector in d-dimensional space
- Points may have varying local densities
- May contain noise/outliers

---

### Label Array
**Definition**: An array storing the cluster assignment for each point.

**Values**:
- 0: UNCLASSIFIED (not yet visited)
- -1: NOISE (outlier)
- 1, 2, 3, ...: Cluster IDs

**Size**: O(n) where n is the number of points

---

### Neighbor List
**Definition**: The list of point indices within the ε-neighborhood of a given point.

**Computation**: Result of RegionQuery operation

**Size**: O(k) where k is the number of neighbors (varies per point)

---

### Seed Queue
**Definition**: A queue of points to be processed during cluster expansion.

**Purpose**: Implements breadth-first search through density-reachable points

**Size**: O(n) in worst case (if all points in one cluster)

---

## Complexity Terms

### Time Complexity
**Definition**: The amount of time an algorithm takes to run as a function of input size.

**DBSCAN Complexity** [Paper §6, p. 230]:
- **Naive implementation**: O(n²) where n is the number of points
- **With spatial indexing**: O(n log n)

**Dominant Operation**: RegionQuery (finding ε-neighborhoods)

---

### Space Complexity
**Definition**: The amount of memory an algorithm uses as a function of input size.

**DBSCAN Space**: O(n) for storing labels, neighbor lists, and seed queue

---

### Big-O Notation
**Definition**: Mathematical notation describing the limiting behavior of a function (typically runtime or space usage) as the input size approaches infinity.

**Examples**:
- O(1): Constant time
- O(log n): Logarithmic time
- O(n): Linear time
- O(n log n): Linearithmic time
- O(n²): Quadratic time

---

### Spatial Indexing
**Definition**: Data structures that organize points in space to enable efficient spatial queries.

**Examples**:
- R-tree, R*-tree
- KD-tree
- Ball-tree

**Purpose**: Reduce RegionQuery complexity from O(n) to O(log n)

**Trade-off**: Additional O(n) space and O(n log n) construction time

---

## Comparison Terms

### K-Means
**Definition**: A partitioning clustering algorithm that divides data into k clusters by minimizing within-cluster variance.

**Key Differences from DBSCAN**:
- Requires k (number of clusters) as input
- Finds only spherical/convex clusters
- No noise detection
- Non-deterministic (random initialization)

---

### Hierarchical Clustering
**Definition**: A family of clustering algorithms that build a hierarchy of clusters (dendrogram).

**Types**:
- Agglomerative (bottom-up)
- Divisive (top-down)

**Key Differences from DBSCAN**:
- Produces hierarchy, not flat clustering
- No explicit noise detection
- Higher time complexity (O(n²) to O(n³))

---

### OPTICS
**Full Name**: Ordering Points To Identify the Clustering Structure

**Definition**: An extension of DBSCAN that produces a reachability plot showing cluster structure at all density levels.

**Key Differences from DBSCAN**:
- Doesn't require ε parameter (uses maximum ε)
- Produces ordering and reachability distances
- Can find clusters of varying densities
- More complex output (reachability plot)

---

### Silhouette Score
**Definition**: A metric for evaluating clustering quality, measuring how similar a point is to its own cluster compared to other clusters.

**Range**: -1 to 1
- 1: Point well matched to its cluster
- 0: Point on border between clusters
- -1: Point likely in wrong cluster

**Use in DBSCAN**: Can help evaluate parameter choices, though not perfect for density-based clustering.

---

### Adjusted Rand Index (ARI)
**Definition**: A metric for comparing two clusterings, corrected for chance.

**Range**: -1 to 1
- 1: Perfect agreement
- 0: Random labeling
- Negative: Worse than random

**Use**: Comparing DBSCAN results with ground truth or other algorithms.

---

## Additional Terms

### Curse of Dimensionality
**Definition**: The phenomenon where many algorithms become less effective as the number of dimensions increases.

**Effect on DBSCAN**:
- Distance metrics become less meaningful in high dimensions
- All points become roughly equidistant
- ε-neighborhoods become less informative
- Harder to distinguish dense from sparse regions

---

### Anomaly Detection
**Definition**: The task of identifying unusual patterns or outliers in data.

**DBSCAN Application**: Noise points identified by DBSCAN can be treated as anomalies.

---

### Spatial Data
**Definition**: Data representing objects in physical space, typically with geographic coordinates.

**Examples**: GPS locations, sensor positions, geographic features

**DBSCAN Suitability**: DBSCAN was originally designed for spatial databases and works well on spatial data.

---

### Reproducibility
**Definition**: The ability to obtain consistent results when repeating an experiment or computation.

**DBSCAN Property**: Results are reproducible because the algorithm is deterministic (no random components).

---

## Notation Guide

### Mathematical Symbols

| Symbol | Meaning | Example |
|--------|---------|---------|
| D | Database (dataset) | D = {p₁, p₂, ..., pₙ} |
| p, q, r | Points | p = (x, y) |
| ε | Epsilon (neighborhood radius) | ε = 0.5 |
| MinPts | Minimum points parameter | MinPts = 5 |
| N_ε(p) | ε-neighborhood of p | N_ε(p) = {q ∈ D \| dist(p,q) ≤ ε} |
| \|S\| | Cardinality (size) of set S | \|N_ε(p)\| = 7 |
| dist(p, q) | Distance between p and q | dist(p, q) = 2.5 |
| C | Cluster | C = {p₁, p₂, p₃} |
| n | Number of points | n = 1000 |
| d | Dimensionality | d = 2 (for 2D data) |
| k | Number of neighbors | k = 4 (for k-distance) |
| ∈ | Element of (set membership) | p ∈ C |
| ∃ | There exists | ∃ o : p is reachable from o |
| ∀ | For all | ∀ p ∈ D |
| ⟹ | Implies | p is core ⟹ \|N_ε(p)\| ≥ MinPts |
| ⟺ | If and only if | p ∈ C ⟺ p is density-connected to C |

---

## References

**Primary Reference**:
- Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In *Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96)* (Vol. 96, No. 34, pp. 226-231).

**Paper Location**: `1996-DBSCAN-KDD.pdf` in repository root

**Related Documentation**:
- [Theory and Math](01_theory_and_math.md) - Core concepts and formulas
- [Density Concepts](02_density_concepts.md) - Detailed density definitions
- [Algorithm Details](03_algorithm_details.md) - Algorithm walkthrough
- [How to Read the Paper](00_how_to_read_the_paper.md) - Paper reading guide

---

*This glossary is part of the Comprehensive DBSCAN Learning Repository. All definitions trace back to the original 1996 paper with proper citations.*
