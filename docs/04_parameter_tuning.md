# DBSCAN Parameter Tuning

## Overview

Selecting appropriate values for ε (epsilon) and MinPts is crucial for achieving good DBSCAN clustering results.

## Selecting MinPts

### General Rule

```
MinPts ≥ D + 1
```

Where D is the number of dimensions in the data

### Best Practices

- For 2D data: MinPts = 4 to 6
- For noisy data: Use higher MinPts values
- Common values: MinPts = 4 or 5

## Selecting Epsilon (ε)

### K-distance Graph Method

The most popular approach:

1. Calculate distance to k-nearest neighbor for all points (k = MinPts)
2. Sort distances in descending order
3. Plot the graph
4. Find the "elbow point" (where the curve changes direction sharply)
5. Optimal ε = distance at the elbow point

### Code Example

```python
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Find distances to k-nearest neighbors
k = 4
neighbors = NearestNeighbors(n_neighbors=k)
neighbors.fit(X)
distances, indices = neighbors.kneighbors(X)

# Sort distances
distances = np.sort(distances[:, k-1], axis=0)[::-1]

# Plot graph
plt.plot(distances)
plt.xlabel('Data Points')
plt.ylabel('k-th Nearest Neighbor Distance')
plt.title('K-distance Graph')
plt.show()
```

## Tuning Techniques

### 1. Grid Search

Try multiple values of ε and MinPts, then select the best performing combination

```python
eps_values = [0.1, 0.3, 0.5, 0.7, 1.0]
min_pts_values = [3, 4, 5, 6, 7]

for eps in eps_values:
    for min_pts in min_pts_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_pts)
        labels = dbscan.fit_predict(X)
        # Evaluate results
```

### 2. Silhouette Score

Use Silhouette Score to evaluate clustering quality

```python
from sklearn.metrics import silhouette_score

score = silhouette_score(X, labels)
```

### 3. Domain Knowledge

Use domain knowledge about the data to determine appropriate values:
- Meaningful distances in the data domain
- Expected cluster sizes

## Case Studies

### Data with Uniform Density
- ε: Medium value
- MinPts: 4-5

### Data with High Noise
- ε: Relatively small
- MinPts: Higher (7-10)

### Spatial Data
- ε: Depends on coordinate units (meters, kilometers)
- MinPts: Depends on data density

## Tips

1. Start with MinPts = 4 or 5
2. Use K-distance graph to find ε
3. Iteratively adjust both parameters
4. Verify results with visualization
5. If too much noise: Decrease ε or increase MinPts
6. If clusters too large: Increase ε or decrease MinPts
