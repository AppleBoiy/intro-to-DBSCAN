"""
DBSCAN Parameter Tuning Module

Tools for selecting optimal DBSCAN parameters (eps and min_pts).
Implements k-distance graph method from Ester et al. (1996) §5.1.

References
----------
Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996).
A density-based algorithm for discovering clusters in large spatial 
databases with noise. In KDD (Vol. 96, No. 34, pp. 226-231).
Section 5.1: Determining the Parameters Eps and MinPts
"""
import numpy as np
from typing import Tuple, List, Dict, Optional
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score


class ParameterSelector:
    """
    Tools for selecting optimal DBSCAN parameters.
    
    This class provides methods for parameter selection based on the
    k-distance graph method described in the DBSCAN paper [Paper §5.1].
    
    Methods
    -------
    compute_k_distances(X, k)
        Compute k-nearest neighbor distances for all points
    find_elbow_point(distances)
        Automatically detect elbow point in k-distance graph
    grid_search(X, eps_range, minpts_range, metric)
        Perform grid search over parameter space
    suggest_parameters(X)
        Suggest parameters based on data characteristics
    
    References
    ----------
    Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996).
    Section 5.1: Determining the Parameters Eps and MinPts
    """
    
    def __init__(self):
        """Initialize ParameterSelector."""
        pass

    def compute_k_distances(self, X: np.ndarray, k: int) -> np.ndarray:
        """
        Compute k-nearest neighbor distances for all points [Paper §5.1].
        
        This method implements the k-distance graph approach for epsilon
        selection. For each point, it computes the distance to its k-th
        nearest neighbor. The sorted k-distances can be plotted to identify
        an appropriate epsilon value using the elbow method.
        
        Parameters
        ----------
        X : np.ndarray
            Dataset of shape (n_samples, n_features)
        k : int
            Number of nearest neighbors (typically k = MinPts)
            
        Returns
        -------
        np.ndarray
            Array of k-nearest neighbor distances, sorted in descending order
            Shape: (n_samples,)
            
        Raises
        ------
        ValueError
            If k is less than 1 or greater than n_samples
        TypeError
            If X is not a numpy array
            
        Notes
        -----
        Time Complexity: O(n log n) with KD-tree
        Space Complexity: O(n)
        
        The k-distance graph method [Paper §5.1]:
        1. Compute k-distance for each point (distance to k-th neighbor)
        2. Sort k-distances in descending order
        3. Plot the sorted k-distances
        4. The "elbow" point indicates optimal epsilon
        
        The intuition is that points in clusters have small k-distances,
        while noise points have large k-distances. The elbow separates
        these two groups.
        
        Examples
        --------
        >>> selector = ParameterSelector()
        >>> k_distances = selector.compute_k_distances(X, k=4)
        >>> # Plot to find elbow
        >>> plt.plot(k_distances)
        >>> plt.xlabel('Points sorted by distance')
        >>> plt.ylabel('4-th nearest neighbor distance')
        
        References
        ----------
        Ester et al. (1996), Section 5.1: Determining the Parameters Eps and MinPts
        """
        # Input validation
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got {X.ndim}D")
        
        n_samples = len(X)
        
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        
        if k > n_samples:
            raise ValueError(f"k must be <= n_samples ({n_samples}), got {k}")
        
        # Handle edge case: single point
        if n_samples == 1:
            return np.array([0.0])
        
        # Use k+1 because kneighbors includes the point itself
        n_neighbors = min(k + 1, n_samples)
        
        # Compute k-nearest neighbors using KD-tree for efficiency
        neighbors = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
        neighbors.fit(X)
        distances, indices = neighbors.kneighbors(X)
        
        # Extract k-th nearest neighbor distance (index k, since index 0 is the point itself)
        if n_neighbors > k:
            k_distances = distances[:, k]
        else:
            # If we don't have enough neighbors, use the farthest available
            k_distances = distances[:, -1]
        
        # Sort in descending order for visualization
        k_distances_sorted = np.sort(k_distances)[::-1]
        
        return k_distances_sorted

    def find_elbow_point(self, distances: np.ndarray) -> Tuple[int, float]:
        """
        Automatically detect elbow point in k-distance graph.
        
        Uses the "maximum curvature" method to find the elbow point,
        which represents the optimal epsilon value. The elbow is the
        point where the curve changes direction most sharply.
        
        Parameters
        ----------
        distances : np.ndarray
            Sorted k-distances (typically from compute_k_distances)
            Should be sorted in descending order
            
        Returns
        -------
        Tuple[int, float]
            (index, epsilon) where:
            - index: Position of elbow point in the distances array
            - epsilon: Suggested epsilon value (distance at elbow)
            
        Raises
        ------
        ValueError
            If distances array is empty or has less than 3 points
            
        Notes
        -----
        Time Complexity: O(n)
        Space Complexity: O(n)
        
        Algorithm:
        1. Normalize the curve to [0, 1] range
        2. Create a line from first to last point
        3. Find point with maximum perpendicular distance to this line
        4. This point is the elbow
        
        This is a simplified version of the Kneedle algorithm.
        For more sophisticated elbow detection, consider using
        the kneed library.
        
        Examples
        --------
        >>> selector = ParameterSelector()
        >>> k_distances = selector.compute_k_distances(X, k=4)
        >>> elbow_idx, suggested_eps = selector.find_elbow_point(k_distances)
        >>> print(f"Suggested epsilon: {suggested_eps:.3f}")
        
        References
        ----------
        Satopaa, V., et al. (2011). "Finding a 'Kneedle' in a Haystack:
        Detecting Knee Points in System Behavior."
        """
        if len(distances) == 0:
            raise ValueError("distances array cannot be empty")
        
        if len(distances) < 3:
            raise ValueError("Need at least 3 points to find elbow")
        
        n_points = len(distances)
        
        # Normalize coordinates to [0, 1] range
        x = np.arange(n_points)
        y = distances
        
        # Normalize
        x_norm = (x - x[0]) / (x[-1] - x[0]) if x[-1] != x[0] else x
        y_norm = (y - y[-1]) / (y[0] - y[-1]) if y[0] != y[-1] else y
        
        # Create line from first to last point
        # Line equation: y = mx + b
        # For normalized coordinates from (0, 1) to (1, 0)
        # The line is simply y = 1 - x
        
        # Calculate perpendicular distance from each point to the line
        # For line ax + by + c = 0, distance = |ax + by + c| / sqrt(a^2 + b^2)
        # Our line: x + y - 1 = 0, so a=1, b=1, c=-1
        distances_to_line = np.abs(x_norm + y_norm - 1) / np.sqrt(2)
        
        # Find point with maximum distance (the elbow)
        elbow_idx = np.argmax(distances_to_line)
        suggested_eps = distances[elbow_idx]
        
        return elbow_idx, suggested_eps

    def grid_search(self, X: np.ndarray, eps_range: List[float],
                   minpts_range: List[int], metric: str = 'silhouette') -> Dict:
        """
        Perform grid search over parameter space with quality metrics.
        
        Tests all combinations of epsilon and min_pts values, evaluating
        each using the specified metric. This helps identify optimal
        parameter combinations for the given dataset.
        
        Parameters
        ----------
        X : np.ndarray
            Dataset of shape (n_samples, n_features)
        eps_range : List[float]
            List of epsilon values to test
        minpts_range : List[int]
            List of min_pts values to test
        metric : str, default='silhouette'
            Evaluation metric to use. Options:
            - 'silhouette': Silhouette score (higher is better)
            
        Returns
        -------
        Dict
            Dictionary containing:
            - 'best_eps': Best epsilon value
            - 'best_minpts': Best min_pts value
            - 'best_score': Best metric score achieved
            - 'results': 2D array of scores for all combinations
            - 'eps_range': Input epsilon range
            - 'minpts_range': Input min_pts range
            
        Raises
        ------
        ValueError
            If eps_range or minpts_range is empty
            If metric is not supported
        TypeError
            If X is not a numpy array
            
        Notes
        -----
        Time Complexity: O(|eps_range| × |minpts_range| × n²)
        Space Complexity: O(|eps_range| × |minpts_range|)
        
        The silhouette score measures how similar a point is to its own
        cluster compared to other clusters. Values range from -1 to 1:
        - 1: Point is well-matched to its cluster
        - 0: Point is on the border between clusters
        - -1: Point may be assigned to wrong cluster
        
        Note: Silhouette score requires at least 2 clusters and cannot
        be computed if all points are noise. In such cases, score is -1.
        
        Examples
        --------
        >>> selector = ParameterSelector()
        >>> results = selector.grid_search(
        ...     X, 
        ...     eps_range=[0.1, 0.3, 0.5, 0.7],
        ...     minpts_range=[3, 4, 5, 6]
        ... )
        >>> print(f"Best parameters: eps={results['best_eps']}, "
        ...       f"min_pts={results['best_minpts']}")
        >>> print(f"Best score: {results['best_score']:.3f}")
        
        References
        ----------
        Rousseeuw, P. J. (1987). "Silhouettes: A graphical aid to the
        interpretation and validation of cluster analysis."
        """
        # Input validation
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got {X.ndim}D")
        
        if len(eps_range) == 0:
            raise ValueError("eps_range cannot be empty")
        
        if len(minpts_range) == 0:
            raise ValueError("minpts_range cannot be empty")
        
        if metric not in ['silhouette']:
            raise ValueError(f"Unsupported metric: {metric}. Use 'silhouette'")
        
        # Import DBSCAN here to avoid circular dependency
        from src.dbscan_from_scratch import DBSCAN
        
        # Initialize results matrix
        results = np.zeros((len(eps_range), len(minpts_range)))
        best_score = -1
        best_eps = eps_range[0]
        best_minpts = minpts_range[0]
        
        # Grid search
        for i, eps in enumerate(eps_range):
            for j, minpts in enumerate(minpts_range):
                # Run DBSCAN
                dbscan = DBSCAN(eps=eps, min_pts=minpts)
                labels = dbscan.fit_predict(X)
                
                # Compute metric
                score = self._compute_metric(X, labels, metric)
                results[i, j] = score
                
                # Update best parameters
                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_minpts = minpts
        
        return {
            'best_eps': best_eps,
            'best_minpts': best_minpts,
            'best_score': best_score,
            'results': results,
            'eps_range': eps_range,
            'minpts_range': minpts_range
        }
    
    def _compute_metric(self, X: np.ndarray, labels: np.ndarray, 
                       metric: str) -> float:
        """
        Compute clustering quality metric.
        
        Parameters
        ----------
        X : np.ndarray
            Dataset
        labels : np.ndarray
            Cluster labels
        metric : str
            Metric name
            
        Returns
        -------
        float
            Metric score
        """
        if metric == 'silhouette':
            # Check if we have valid clusters
            unique_labels = set(labels)
            # Remove noise label (-1)
            unique_labels.discard(-1)
            
            # Need at least 2 clusters for silhouette score
            if len(unique_labels) < 2:
                return -1.0
            
            # Need at least 2 non-noise points
            non_noise_mask = labels != -1
            if np.sum(non_noise_mask) < 2:
                return -1.0
            
            try:
                # Compute silhouette score only on non-noise points
                score = silhouette_score(X[non_noise_mask], labels[non_noise_mask])
                return score
            except:
                return -1.0
        
        return -1.0

    def suggest_parameters(self, X: np.ndarray) -> Tuple[float, int]:
        """
        Suggest parameters based on data characteristics.
        
        Uses heuristics and the k-distance graph method to automatically
        suggest reasonable DBSCAN parameters for the given dataset.
        
        Heuristics [Paper §5.1]:
        1. MinPts: Use MinPts = 2*dim for dim-dimensional data
           - Minimum: 4 (for robustness)
           - For 2D data: typically 4-6
        2. Epsilon: Use k-distance graph with k = MinPts
           - Find elbow point in sorted k-distances
        
        Parameters
        ----------
        X : np.ndarray
            Dataset of shape (n_samples, n_features)
            
        Returns
        -------
        Tuple[float, int]
            (suggested_eps, suggested_minpts)
            
        Raises
        ------
        TypeError
            If X is not a numpy array
        ValueError
            If X is not 2-dimensional or is empty
            
        Notes
        -----
        Time Complexity: O(n log n)
        Space Complexity: O(n)
        
        These are starting point suggestions. Users should:
        1. Visualize results with suggested parameters
        2. Adjust based on domain knowledge
        3. Use grid_search for fine-tuning if needed
        
        The MinPts heuristic comes from the paper's recommendation
        that MinPts should be at least the dimensionality of the data
        plus one, with a minimum of 4 for robustness.
        
        Examples
        --------
        >>> selector = ParameterSelector()
        >>> suggested_eps, suggested_minpts = selector.suggest_parameters(X)
        >>> print(f"Suggested parameters: eps={suggested_eps:.3f}, "
        ...       f"min_pts={suggested_minpts}")
        >>> 
        >>> # Use suggested parameters
        >>> dbscan = DBSCAN(eps=suggested_eps, min_pts=suggested_minpts)
        >>> labels = dbscan.fit_predict(X)
        
        References
        ----------
        Ester et al. (1996), Section 5.1: Determining the Parameters Eps and MinPts
        "We recommend to choose MinPts at least as large as the dimensionality
        of the data set plus one."
        """
        # Input validation
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got {X.ndim}D")
        
        if len(X) == 0:
            raise ValueError("X cannot be empty")
        
        n_samples, n_features = X.shape
        
        # Suggest MinPts based on dimensionality [Paper §5.1]
        # Rule: MinPts >= dimensionality + 1, with minimum of 4
        suggested_minpts = max(4, 2 * n_features)
        
        # For very high-dimensional data, cap at reasonable value
        if suggested_minpts > 20:
            suggested_minpts = 20
        
        # Handle case where we don't have enough samples
        if n_samples < suggested_minpts:
            suggested_minpts = max(2, n_samples // 2)
        
        # Suggest epsilon using k-distance graph method
        try:
            k_distances = self.compute_k_distances(X, k=suggested_minpts)
            elbow_idx, suggested_eps = self.find_elbow_point(k_distances)
        except:
            # Fallback: use median of pairwise distances (simplified)
            # This is a rough heuristic if k-distance method fails
            from sklearn.metrics import pairwise_distances
            
            # Sample if dataset is large
            if n_samples > 1000:
                sample_indices = np.random.choice(n_samples, 1000, replace=False)
                X_sample = X[sample_indices]
            else:
                X_sample = X
            
            distances = pairwise_distances(X_sample)
            # Use a percentile of non-zero distances
            non_zero_distances = distances[distances > 0]
            if len(non_zero_distances) > 0:
                suggested_eps = np.percentile(non_zero_distances, 10)
            else:
                suggested_eps = 0.5  # Default fallback
        
        return suggested_eps, suggested_minpts
