"""
Visualization Functions for DBSCAN

Comprehensive visualization system for DBSCAN concepts including
epsilon-neighborhoods, point types, density-reachability, algorithm steps,
and comparison visualizations.

References
----------
Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996).
A density-based algorithm for discovering clusters in large spatial 
databases with noise. In KDD (Vol. 96, No. 34, pp. 226-231).
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class VisualizationConfig:
    """
    Configuration for consistent visualizations across all DBSCAN plots.
    
    This dataclass defines the visual styling standards for all DBSCAN
    visualizations to ensure consistency and clarity.
    
    Attributes
    ----------
    figsize : Tuple[int, int]
        Figure size in inches (width, height)
    dpi : int
        Dots per inch for figure resolution
    style : str
        Matplotlib style to use ('seaborn', 'default', etc.)
    
    Color Scheme
    ------------
    core_color : str
        Color for core points (default: blue #1f77b4)
    border_color : str
        Color for border points (default: green #2ca02c)
    noise_color : str
        Color for noise points (default: black #000000)
    neighborhood_color : str
        Color for epsilon-neighborhood circles (default: light gray #d3d3d3)
    
    Marker Styles
    -------------
    core_marker : str
        Marker style for core points (default: 'o' - circle)
    core_size : int
        Marker size for core points
    border_marker : str
        Marker style for border points (default: 'o' - circle)
    border_size : int
        Marker size for border points
    noise_marker : str
        Marker style for noise points (default: 'x' - cross)
    noise_size : int
        Marker size for noise points
    
    Line Styles
    -----------
    neighborhood_linestyle : str
        Line style for epsilon-neighborhood circles (default: '--' - dashed)
    neighborhood_linewidth : float
        Line width for epsilon-neighborhood circles
    neighborhood_alpha : float
        Transparency for epsilon-neighborhood circles
    
    Grid and Axes
    -------------
    show_grid : bool
        Whether to show grid lines
    grid_alpha : float
        Transparency for grid lines
    show_legend : bool
        Whether to show legend by default
    """
    # Figure settings
    figsize: Tuple[int, int] = (10, 6)
    dpi: int = 100
    style: str = 'seaborn-v0_8-darkgrid'
    
    # Color scheme
    core_color: str = '#1f77b4'  # Blue
    border_color: str = '#2ca02c'  # Green
    noise_color: str = '#000000'  # Black
    neighborhood_color: str = '#d3d3d3'  # Light gray
    
    # Marker styles
    core_marker: str = 'o'
    core_size: int = 100
    border_marker: str = 'o'
    border_size: int = 60
    noise_marker: str = 'x'
    noise_size: int = 50
    
    # Line styles
    neighborhood_linestyle: str = '--'
    neighborhood_linewidth: float = 1.5
    neighborhood_alpha: float = 0.3
    
    # Grid and axes
    show_grid: bool = True
    grid_alpha: float = 0.3
    show_legend: bool = True


class DBSCANVisualizer:
    """
    Comprehensive visualization system for DBSCAN concepts.
    
    This class provides a unified interface for creating all types of
    DBSCAN visualizations including basic clustering results, epsilon-
    neighborhoods, point type classifications, algorithm steps, and
    comparison plots.
    
    Parameters
    ----------
    style : str, default='seaborn-v0_8-darkgrid'
        Matplotlib style to use for all plots
    figsize : Tuple[int, int], default=(10, 6)
        Default figure size in inches (width, height)
    config : Optional[VisualizationConfig], default=None
        Custom configuration object. If None, uses default configuration
        with provided style and figsize parameters.
    
    Attributes
    ----------
    config : VisualizationConfig
        Configuration object containing all styling parameters
    
    Examples
    --------
    >>> visualizer = DBSCANVisualizer()
    >>> visualizer.plot_clusters(X, labels, title="DBSCAN Results")
    
    >>> # Custom configuration
    >>> custom_config = VisualizationConfig(
    ...     figsize=(12, 8),
    ...     core_color='red',
    ...     noise_marker='+'
    ... )
    >>> visualizer = DBSCANVisualizer(config=custom_config)
    
    Notes
    -----
    All visualization methods follow consistent styling defined in the
    VisualizationConfig dataclass. This ensures visual consistency across
    all plots and makes it easy to customize the appearance globally.
    """
    
    def __init__(
        self,
        style: str = 'seaborn-v0_8-darkgrid',
        figsize: Tuple[int, int] = (10, 6),
        config: Optional[VisualizationConfig] = None
    ):
        """
        Initialize DBSCANVisualizer with consistent styling.
        
        Parameters
        ----------
        style : str, default='seaborn-v0_8-darkgrid'
            Matplotlib style to use for all plots
        figsize : Tuple[int, int], default=(10, 6)
            Default figure size in inches (width, height)
        config : Optional[VisualizationConfig], default=None
            Custom configuration object. If None, creates default config
            with provided style and figsize.
        """
        if config is None:
            self.config = VisualizationConfig(
                figsize=figsize,
                style=style
            )
        else:
            self.config = config
        
        # Apply matplotlib style
        try:
            plt.style.use(self.config.style)
        except OSError:
            # Fallback to default if style not found
            plt.style.use('default')
    
    def plot_clusters(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        title: str = "DBSCAN Clustering",
        highlight_points: Optional[np.ndarray] = None
    ):
        """
        Plot clustering results with consistent styling.
        
        Parameters
        ----------
        X : np.ndarray
            Data array of shape (n_samples, n_features)
        labels : np.ndarray
            Cluster labels for each point (-1 = noise, 0+ = cluster ID)
        title : str, default="DBSCAN Clustering"
            Plot title
        highlight_points : Optional[np.ndarray], default=None
            Indices of points to highlight with special markers
        
        Returns
        -------
        matplotlib.pyplot
            The pyplot module for further customization or display
        """
        plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Get unique labels
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Noise points
                mask = labels == label
                plt.scatter(
                    X[mask, 0], X[mask, 1],
                    c=self.config.noise_color,
                    marker=self.config.noise_marker,
                    s=self.config.noise_size,
                    alpha=0.5,
                    label='Noise'
                )
            else:
                # Cluster points
                mask = labels == label
                plt.scatter(
                    X[mask, 0], X[mask, 1],
                    c=[color],
                    marker='o',
                    s=50,
                    alpha=0.7,
                    label=f'Cluster {label}'
                )
        
        # Highlight specific points if requested
        if highlight_points is not None:
            plt.scatter(
                X[highlight_points, 0], X[highlight_points, 1],
                facecolors='none',
                edgecolors='red',
                s=200,
                linewidths=2,
                label='Highlighted'
            )
        
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        if self.config.show_legend:
            plt.legend()
        
        if self.config.show_grid:
            plt.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        return plt

    def plot_k_distance_graph(
        self,
        X: np.ndarray,
        k: int = 4,
        show_elbow: bool = False
    ):
        """
        Plot K-distance graph for finding optimal epsilon value.
        
        The k-distance graph helps identify the optimal epsilon parameter
        by plotting the distance to the k-th nearest neighbor for each point,
        sorted in descending order. The "elbow" in this curve suggests a
        good epsilon value [Paper §5.1].
        
        Parameters
        ----------
        X : np.ndarray
            Data array of shape (n_samples, n_features)
        k : int, default=4
            Number of neighbors (should equal min_pts parameter)
        show_elbow : bool, default=False
            Whether to attempt automatic elbow detection and annotation
        
        Returns
        -------
        matplotlib.pyplot
            The pyplot module for further customization or display
        
        Notes
        -----
        The k parameter should typically match the min_pts parameter you
        plan to use for DBSCAN. A common heuristic is k = 2 * dimensions - 1.
        """
        # Find distances to k-nearest neighbors
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(X)
        distances, indices = neighbors.kneighbors(X)
        
        # Sort distances in descending order
        distances = np.sort(distances[:, k-1], axis=0)[::-1]
        
        plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        plt.plot(distances, linewidth=2, label='k-distance')
        
        # Elbow detection if requested
        if show_elbow:
            elbow_idx = self._detect_elbow(distances)
            elbow_eps = distances[elbow_idx]
            
            # Mark the elbow point
            plt.scatter(
                [elbow_idx], [elbow_eps],
                c='red',
                s=200,
                marker='o',
                edgecolors='black',
                linewidths=2,
                label=f'Elbow (ε≈{elbow_eps:.3f})',
                zorder=5
            )
            
            # Draw horizontal line at elbow
            plt.axhline(
                y=elbow_eps,
                color='red',
                linestyle='--',
                alpha=0.5,
                linewidth=1.5
            )
            
            # Annotate the elbow point
            plt.annotate(
                f'Suggested ε = {elbow_eps:.3f}',
                xy=(elbow_idx, elbow_eps),
                xytext=(elbow_idx + len(distances) * 0.1, elbow_eps + np.max(distances) * 0.05),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10,
                color='red',
                fontweight='bold'
            )
        
        plt.xlabel('Data Points (sorted by distance)')
        plt.ylabel(f'{k}-th Nearest Neighbor Distance')
        plt.title(f'K-distance Graph (k={k})')
        
        if self.config.show_legend:
            plt.legend()
        
        if self.config.show_grid:
            plt.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        
        return plt
    
    def _detect_elbow(self, distances: np.ndarray) -> int:
        """
        Detect elbow point in k-distance graph using the maximum curvature method.
        
        The elbow is the point of maximum curvature in the sorted distance curve,
        which represents the transition from dense to sparse regions.
        
        Parameters
        ----------
        distances : np.ndarray
            Sorted k-distances in descending order
        
        Returns
        -------
        int
            Index of the elbow point
        
        Notes
        -----
        Uses the "knee/elbow detection" algorithm based on the maximum distance
        from the line connecting the first and last points to the curve.
        """
        n_points = len(distances)
        
        # Handle edge cases
        if n_points < 3:
            return 0
        
        # Create line from first to last point
        first_point = np.array([0, distances[0]])
        last_point = np.array([n_points - 1, distances[-1]])
        
        # Calculate perpendicular distance from each point to the line
        line_vec = last_point - first_point
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            return n_points // 2
        
        line_unitvec = line_vec / line_len
        
        max_dist = 0
        elbow_idx = 0
        
        for i in range(1, n_points - 1):
            point = np.array([i, distances[i]])
            vec_to_point = point - first_point
            
            # Calculate perpendicular distance
            projection = np.dot(vec_to_point, line_unitvec)
            closest_point = first_point + projection * line_unitvec
            dist = np.linalg.norm(point - closest_point)
            
            if dist > max_dist:
                max_dist = dist
                elbow_idx = i
        
        return elbow_idx
    
    def plot_epsilon_neighborhood(
        self,
        X: np.ndarray,
        point_idx: int,
        eps: float,
        labels: Optional[np.ndarray] = None,
        title: Optional[str] = None
    ):
        """
        Visualize epsilon-neighborhood of a specific point [Paper §4.1].
        
        Shows the ε-neighborhood N_ε(p) = {q ∈ D | dist(p, q) ≤ ε} as a circle
        around the selected point, highlighting all neighbors within the radius.
        
        Parameters
        ----------
        X : np.ndarray
            Data array of shape (n_samples, n_features)
        point_idx : int
            Index of the point to show neighborhood for
        eps : float
            Epsilon radius for the neighborhood
        labels : Optional[np.ndarray], default=None
            Cluster labels for coloring points (if available)
        title : Optional[str], default=None
            Plot title (auto-generated if None)
        
        Returns
        -------
        matplotlib.pyplot
            The pyplot module for further customization or display
        
        Notes
        -----
        The epsilon-neighborhood is a fundamental concept in DBSCAN.
        A point is a core point if |N_ε(p)| >= MinPts.
        
        Examples
        --------
        >>> visualizer = DBSCANVisualizer()
        >>> visualizer.plot_epsilon_neighborhood(X, point_idx=0, eps=0.5)
        """
        plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Plot all points
        if labels is not None:
            unique_labels = set(labels)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
            for label, color in zip(unique_labels, colors):
                if label == -1:
                    mask = labels == label
                    plt.scatter(
                        X[mask, 0], X[mask, 1],
                        c=self.config.noise_color,
                        marker=self.config.noise_marker,
                        s=self.config.noise_size,
                        alpha=0.3,
                        label='Noise'
                    )
                else:
                    mask = labels == label
                    plt.scatter(
                        X[mask, 0], X[mask, 1],
                        c=[color],
                        marker='o',
                        s=30,
                        alpha=0.3,
                        label=f'Cluster {label}'
                    )
        else:
            plt.scatter(X[:, 0], X[:, 1], c='lightgray', s=30, alpha=0.5, label='All points')
        
        # Highlight the selected point
        plt.scatter(
            X[point_idx, 0], X[point_idx, 1],
            c='red',
            marker='*',
            s=300,
            edgecolors='black',
            linewidths=2,
            label=f'Point {point_idx}',
            zorder=5
        )
        
        # Find and highlight neighbors within eps
        distances = np.sqrt(np.sum((X - X[point_idx])**2, axis=1))
        neighbors = np.where(distances <= eps)[0]
        neighbors = neighbors[neighbors != point_idx]  # Exclude the point itself
        
        if len(neighbors) > 0:
            plt.scatter(
                X[neighbors, 0], X[neighbors, 1],
                facecolors='none',
                edgecolors='orange',
                s=150,
                linewidths=2,
                label=f'Neighbors ({len(neighbors)})',
                zorder=4
            )
        
        # Draw epsilon-neighborhood circle
        circle = plt.Circle(
            (X[point_idx, 0], X[point_idx, 1]),
            eps,
            color=self.config.neighborhood_color,
            fill=False,
            linestyle=self.config.neighborhood_linestyle,
            linewidth=self.config.neighborhood_linewidth,
            alpha=self.config.neighborhood_alpha + 0.4,
            label=f'ε-neighborhood (ε={eps:.2f})'
        )
        plt.gca().add_patch(circle)
        
        if title is None:
            title = f'ε-neighborhood of Point {point_idx} (ε={eps:.2f}, |N_ε(p)|={len(neighbors)+1})'
        
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.axis('equal')
        
        if self.config.show_legend:
            plt.legend(loc='best')
        
        if self.config.show_grid:
            plt.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        return plt
    
    def plot_point_types(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        core_sample_indices: np.ndarray,
        eps: float,
        title: str = "DBSCAN Point Types"
    ):
        """
        Visualize point type classification: core, border, and noise [Paper §4.1].
        
        Distinguishes between:
        - Core points: Have at least MinPts neighbors within ε
        - Border points: Within ε of a core point but not core themselves
        - Noise points: Neither core nor border (outliers)
        
        Parameters
        ----------
        X : np.ndarray
            Data array of shape (n_samples, n_features)
        labels : np.ndarray
            Cluster labels for each point (-1 = noise, 0+ = cluster ID)
        core_sample_indices : np.ndarray
            Indices of core points
        eps : float
            Epsilon radius (used for visualization context)
        title : str, default="DBSCAN Point Types"
            Plot title
        
        Returns
        -------
        matplotlib.pyplot
            The pyplot module for further customization or display
        
        Notes
        -----
        Point type classification is fundamental to understanding DBSCAN:
        - Core points form the "skeleton" of clusters
        - Border points extend clusters but cannot form new ones
        - Noise points are outliers that don't belong to any cluster
        
        Examples
        --------
        >>> from src.dbscan_from_scratch import DBSCAN
        >>> dbscan = DBSCAN(eps=0.5, min_pts=5)
        >>> labels = dbscan.fit_predict(X)
        >>> visualizer = DBSCANVisualizer()
        >>> visualizer.plot_point_types(X, labels, dbscan.core_sample_indices_, eps=0.5)
        """
        plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Identify point types
        core_mask = np.zeros(len(X), dtype=bool)
        core_mask[core_sample_indices] = True
        
        noise_mask = labels == -1
        border_mask = ~core_mask & ~noise_mask
        
        # Plot noise points
        if np.any(noise_mask):
            plt.scatter(
                X[noise_mask, 0], X[noise_mask, 1],
                c=self.config.noise_color,
                marker=self.config.noise_marker,
                s=self.config.noise_size,
                alpha=0.6,
                label=f'Noise ({np.sum(noise_mask)})',
                zorder=1
            )
        
        # Plot border points
        if np.any(border_mask):
            plt.scatter(
                X[border_mask, 0], X[border_mask, 1],
                c=self.config.border_color,
                marker=self.config.border_marker,
                s=self.config.border_size,
                alpha=0.7,
                label=f'Border ({np.sum(border_mask)})',
                zorder=2
            )
        
        # Plot core points
        if np.any(core_mask):
            plt.scatter(
                X[core_mask, 0], X[core_mask, 1],
                c=self.config.core_color,
                marker=self.config.core_marker,
                s=self.config.core_size,
                alpha=0.8,
                edgecolors='black',
                linewidths=0.5,
                label=f'Core ({np.sum(core_mask)})',
                zorder=3
            )
        
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        if self.config.show_legend:
            plt.legend(loc='best')
        
        if self.config.show_grid:
            plt.grid(True, alpha=self.config.grid_alpha)
        
        # Add annotation explaining point types
        textstr = (
            'Point Types:\n'
            '• Core: |N_ε(p)| ≥ MinPts\n'
            '• Border: In N_ε(core) but not core\n'
            '• Noise: Neither core nor border'
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(
            0.02, 0.98, textstr,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=props
        )
        
        plt.tight_layout()
        return plt
    
    def plot_density_reachability(
        self,
        X: np.ndarray,
        point_chain: list,
        eps: float,
        labels: Optional[np.ndarray] = None,
        title: Optional[str] = None
    ):
        """
        Visualize density-reachability chain between points [Paper §4.1].
        
        Shows a chain of points where each point is directly density-reachable
        from the previous one. A point q is density-reachable from p if there
        exists a chain p = p1, p2, ..., pn = q where each pi+1 is directly
        density-reachable from pi.
        
        Parameters
        ----------
        X : np.ndarray
            Data array of shape (n_samples, n_features)
        point_chain : list
            List of point indices forming the density-reachability chain
            (e.g., [0, 5, 12, 18] means 0→5→12→18)
        eps : float
            Epsilon radius for neighborhood visualization
        labels : Optional[np.ndarray], default=None
            Cluster labels for coloring points (if available)
        title : Optional[str], default=None
            Plot title (auto-generated if None)
        
        Returns
        -------
        matplotlib.pyplot
            The pyplot module for further customization or display
        
        Notes
        -----
        Density-reachability is a transitive relation that forms the basis
        of cluster expansion in DBSCAN. This visualization helps understand
        how clusters grow from core points.
        
        Examples
        --------
        >>> visualizer = DBSCANVisualizer()
        >>> chain = [0, 5, 12, 18]  # Point 18 is density-reachable from 0
        >>> visualizer.plot_density_reachability(X, chain, eps=0.5)
        """
        plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Plot all points in background
        if labels is not None:
            unique_labels = set(labels)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
            for label, color in zip(unique_labels, colors):
                if label == -1:
                    mask = labels == label
                    plt.scatter(
                        X[mask, 0], X[mask, 1],
                        c=self.config.noise_color,
                        marker=self.config.noise_marker,
                        s=self.config.noise_size,
                        alpha=0.2
                    )
                else:
                    mask = labels == label
                    plt.scatter(
                        X[mask, 0], X[mask, 1],
                        c=[color],
                        marker='o',
                        s=30,
                        alpha=0.2
                    )
        else:
            plt.scatter(X[:, 0], X[:, 1], c='lightgray', s=30, alpha=0.3)
        
        # Draw epsilon-neighborhoods for chain points
        for idx in point_chain:
            circle = plt.Circle(
                (X[idx, 0], X[idx, 1]),
                eps,
                color=self.config.neighborhood_color,
                fill=False,
                linestyle=self.config.neighborhood_linestyle,
                linewidth=1.0,
                alpha=0.4
            )
            plt.gca().add_patch(circle)
        
        # Draw arrows showing the reachability chain
        for i in range(len(point_chain) - 1):
            start_idx = point_chain[i]
            end_idx = point_chain[i + 1]
            plt.annotate(
                '',
                xy=(X[end_idx, 0], X[end_idx, 1]),
                xytext=(X[start_idx, 0], X[start_idx, 1]),
                arrowprops=dict(
                    arrowstyle='->',
                    color='red',
                    lw=2,
                    alpha=0.7
                ),
                zorder=4
            )
        
        # Highlight chain points with numbers
        for i, idx in enumerate(point_chain):
            plt.scatter(
                X[idx, 0], X[idx, 1],
                c='red' if i == 0 else ('orange' if i == len(point_chain) - 1 else 'yellow'),
                marker='o',
                s=200,
                edgecolors='black',
                linewidths=2,
                zorder=5
            )
            plt.text(
                X[idx, 0], X[idx, 1],
                str(i),
                fontsize=12,
                fontweight='bold',
                ha='center',
                va='center',
                zorder=6
            )
        
        if title is None:
            title = f'Density-Reachability Chain (ε={eps:.2f})'
        
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.axis('equal')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=10, label='Start point', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow',
                   markersize=10, label='Intermediate', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                   markersize=10, label='End point', markeredgecolor='black'),
            Line2D([0], [0], color='red', lw=2, label='Reachability path')
        ]
        plt.legend(handles=legend_elements, loc='best')
        
        if self.config.show_grid:
            plt.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        return plt
    
    def plot_density_connectivity(
        self,
        X: np.ndarray,
        point_a: int,
        point_b: int,
        connecting_point: int,
        eps: float,
        labels: Optional[np.ndarray] = None,
        title: Optional[str] = None
    ):
        """
        Visualize density-connectivity between two points [Paper §4.1].
        
        Two points p and q are density-connected if there exists a point o
        such that both p and q are density-reachable from o. This visualization
        shows the connecting point and the paths to both target points.
        
        Parameters
        ----------
        X : np.ndarray
            Data array of shape (n_samples, n_features)
        point_a : int
            Index of first point
        point_b : int
            Index of second point
        connecting_point : int
            Index of the point from which both a and b are density-reachable
        eps : float
            Epsilon radius for neighborhood visualization
        labels : Optional[np.ndarray], default=None
            Cluster labels for coloring points (if available)
        title : Optional[str], default=None
            Plot title (auto-generated if None)
        
        Returns
        -------
        matplotlib.pyplot
            The pyplot module for further customization or display
        
        Notes
        -----
        Density-connectivity is a symmetric relation that defines cluster
        membership. All points in a cluster are density-connected to each other.
        
        Examples
        --------
        >>> visualizer = DBSCANVisualizer()
        >>> # Points 5 and 12 are density-connected via point 8
        >>> visualizer.plot_density_connectivity(X, point_a=5, point_b=12,
        ...                                      connecting_point=8, eps=0.5)
        """
        plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Plot all points in background
        if labels is not None:
            unique_labels = set(labels)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
            for label, color in zip(unique_labels, colors):
                if label == -1:
                    mask = labels == label
                    plt.scatter(
                        X[mask, 0], X[mask, 1],
                        c=self.config.noise_color,
                        marker=self.config.noise_marker,
                        s=self.config.noise_size,
                        alpha=0.2
                    )
                else:
                    mask = labels == label
                    plt.scatter(
                        X[mask, 0], X[mask, 1],
                        c=[color],
                        marker='o',
                        s=30,
                        alpha=0.2
                    )
        else:
            plt.scatter(X[:, 0], X[:, 1], c='lightgray', s=30, alpha=0.3)
        
        # Draw epsilon-neighborhoods for the three key points
        for idx, color_alpha in [(point_a, 0.3), (point_b, 0.3), (connecting_point, 0.5)]:
            circle = plt.Circle(
                (X[idx, 0], X[idx, 1]),
                eps,
                color=self.config.neighborhood_color,
                fill=False,
                linestyle=self.config.neighborhood_linestyle,
                linewidth=1.5,
                alpha=color_alpha
            )
            plt.gca().add_patch(circle)
        
        # Draw lines showing connectivity
        plt.plot(
            [X[connecting_point, 0], X[point_a, 0]],
            [X[connecting_point, 1], X[point_a, 1]],
            'b--',
            linewidth=2,
            alpha=0.6,
            label='Path to point A'
        )
        plt.plot(
            [X[connecting_point, 0], X[point_b, 0]],
            [X[connecting_point, 1], X[point_b, 1]],
            'g--',
            linewidth=2,
            alpha=0.6,
            label='Path to point B'
        )
        
        # Highlight the three key points
        plt.scatter(
            X[point_a, 0], X[point_a, 1],
            c='blue',
            marker='o',
            s=250,
            edgecolors='black',
            linewidths=2,
            label='Point A',
            zorder=5
        )
        plt.scatter(
            X[point_b, 0], X[point_b, 1],
            c='green',
            marker='o',
            s=250,
            edgecolors='black',
            linewidths=2,
            label='Point B',
            zorder=5
        )
        plt.scatter(
            X[connecting_point, 0], X[connecting_point, 1],
            c='red',
            marker='*',
            s=400,
            edgecolors='black',
            linewidths=2,
            label='Connecting point',
            zorder=6
        )
        
        if title is None:
            title = f'Density-Connectivity (ε={eps:.2f})'
        
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.axis('equal')
        
        if self.config.show_legend:
            plt.legend(loc='best')
        
        if self.config.show_grid:
            plt.grid(True, alpha=self.config.grid_alpha)
        
        # Add annotation
        textstr = (
            'Density-Connected:\n'
            'Points A and B are density-connected\n'
            'via the connecting point (both are\n'
            'density-reachable from it)'
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(
            0.02, 0.98, textstr,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=props
        )
        
        plt.tight_layout()
        return plt

    def plot_algorithm_step(
        self,
        X: np.ndarray,
        current_point: int,
        visited: np.ndarray,
        labels: np.ndarray,
        eps: float,
        step_num: int,
        current_neighbors: Optional[np.ndarray] = None,
        title: Optional[str] = None
    ):
        """
        Visualize a single step of the DBSCAN algorithm execution.
        
        Shows the current state of the algorithm including which points have
        been visited, current cluster assignments, the point being processed,
        and its epsilon-neighborhood.
        
        Parameters
        ----------
        X : np.ndarray
            Data array of shape (n_samples, n_features)
        current_point : int
            Index of the point currently being processed
        visited : np.ndarray
            Boolean array indicating which points have been visited
        labels : np.ndarray
            Current cluster labels (-1 = noise/unvisited, 0+ = cluster ID)
        eps : float
            Epsilon radius for neighborhood visualization
        step_num : int
            Current step number in the algorithm
        current_neighbors : Optional[np.ndarray], default=None
            Indices of neighbors of the current point (if available)
        title : Optional[str], default=None
            Plot title (auto-generated if None)
        
        Returns
        -------
        matplotlib.pyplot
            The pyplot module for further customization or display
        
        Notes
        -----
        This visualization helps understand how DBSCAN processes points
        sequentially and expands clusters through density-reachability.
        
        Examples
        --------
        >>> visualizer = DBSCANVisualizer()
        >>> # During algorithm execution
        >>> visualizer.plot_algorithm_step(X, current_point=5, visited=visited_mask,
        ...                               labels=current_labels, eps=0.5, step_num=10)
        """
        plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Plot unvisited points
        unvisited_mask = ~visited
        if np.any(unvisited_mask):
            plt.scatter(
                X[unvisited_mask, 0], X[unvisited_mask, 1],
                c='lightgray',
                marker='o',
                s=50,
                alpha=0.3,
                label='Unvisited',
                zorder=1
            )
        
        # Plot visited points by cluster
        unique_labels = set(labels[visited])
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Noise points
                mask = visited & (labels == label)
                if np.any(mask):
                    plt.scatter(
                        X[mask, 0], X[mask, 1],
                        c=self.config.noise_color,
                        marker=self.config.noise_marker,
                        s=self.config.noise_size,
                        alpha=0.6,
                        label='Noise',
                        zorder=2
                    )
            elif label == 0:
                # Unassigned but visited
                mask = visited & (labels == label)
                if np.any(mask):
                    plt.scatter(
                        X[mask, 0], X[mask, 1],
                        c='gray',
                        marker='o',
                        s=50,
                        alpha=0.5,
                        label='Visited (unassigned)',
                        zorder=2
                    )
            else:
                # Cluster points
                mask = visited & (labels == label)
                if np.any(mask):
                    plt.scatter(
                        X[mask, 0], X[mask, 1],
                        c=[color],
                        marker='o',
                        s=60,
                        alpha=0.7,
                        label=f'Cluster {label}',
                        zorder=2
                    )
        
        # Draw epsilon-neighborhood circle around current point
        circle = plt.Circle(
            (X[current_point, 0], X[current_point, 1]),
            eps,
            color='red',
            fill=False,
            linestyle=self.config.neighborhood_linestyle,
            linewidth=2.5,
            alpha=0.6,
            label=f'ε-neighborhood',
            zorder=3
        )
        plt.gca().add_patch(circle)
        
        # Highlight current neighbors if provided
        if current_neighbors is not None and len(current_neighbors) > 0:
            neighbors_to_show = current_neighbors[current_neighbors != current_point]
            if len(neighbors_to_show) > 0:
                plt.scatter(
                    X[neighbors_to_show, 0], X[neighbors_to_show, 1],
                    facecolors='none',
                    edgecolors='orange',
                    s=150,
                    linewidths=2,
                    label=f'Neighbors ({len(neighbors_to_show)})',
                    zorder=4
                )
        
        # Highlight the current point being processed
        plt.scatter(
            X[current_point, 0], X[current_point, 1],
            c='red',
            marker='*',
            s=400,
            edgecolors='black',
            linewidths=2,
            label=f'Current point ({current_point})',
            zorder=5
        )
        
        # Add step number annotation
        plt.text(
            0.02, 0.02,
            f'Step {step_num}',
            transform=plt.gca().transAxes,
            fontsize=14,
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
            zorder=6
        )
        
        if title is None:
            title = f'DBSCAN Algorithm - Step {step_num}'
        
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        if self.config.show_legend:
            plt.legend(loc='best', fontsize=8)
        
        if self.config.show_grid:
            plt.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        return plt
    
    def animate_algorithm_steps(
        self,
        X: np.ndarray,
        eps: float,
        min_pts: int,
        save_path: Optional[str] = None,
        interval: int = 1000
    ):
        """
        Create an animation showing the full DBSCAN algorithm execution.
        
        Generates a step-by-step animation of how DBSCAN processes each point,
        identifies clusters, and classifies points as core, border, or noise.
        
        Parameters
        ----------
        X : np.ndarray
            Data array of shape (n_samples, n_features)
        eps : float
            Epsilon radius for DBSCAN
        min_pts : int
            Minimum points for core point classification
        save_path : Optional[str], default=None
            If provided, saves animation to this file path (e.g., 'animation.gif')
            Requires imagemagick or pillow for GIF, ffmpeg for MP4
        interval : int, default=1000
            Delay between frames in milliseconds
        
        Returns
        -------
        matplotlib.animation.FuncAnimation
            Animation object that can be displayed or saved
        
        Notes
        -----
        This animation provides a visual understanding of how DBSCAN
        sequentially processes points and expands clusters. It's particularly
        useful for educational purposes to see the algorithm in action.
        
        The animation shows:
        - Current point being processed (red star)
        - Epsilon-neighborhood (red circle)
        - Visited vs unvisited points
        - Current cluster assignments
        - Step number
        
        Examples
        --------
        >>> from sklearn.datasets import make_moons
        >>> X, _ = make_moons(n_samples=100, noise=0.05, random_state=42)
        >>> visualizer = DBSCANVisualizer()
        >>> anim = visualizer.animate_algorithm_steps(X, eps=0.3, min_pts=5)
        >>> plt.show()  # Display animation
        
        >>> # Save animation
        >>> anim = visualizer.animate_algorithm_steps(X, eps=0.3, min_pts=5,
        ...                                           save_path='dbscan.gif')
        """
        from matplotlib.animation import FuncAnimation
        
        # Import DBSCAN to run the algorithm step by step
        # We'll need to modify this to capture intermediate states
        from src.dbscan_from_scratch import DBSCAN
        
        # Run DBSCAN and capture states at each step
        states = self._capture_algorithm_states(X, eps, min_pts)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        def update(frame):
            """Update function for animation"""
            ax.clear()
            
            state = states[frame]
            current_point = state['current_point']
            visited = state['visited']
            labels = state['labels']
            step_num = state['step_num']
            current_neighbors = state.get('neighbors', None)
            
            # Plot unvisited points
            unvisited_mask = ~visited
            if np.any(unvisited_mask):
                ax.scatter(
                    X[unvisited_mask, 0], X[unvisited_mask, 1],
                    c='lightgray',
                    marker='o',
                    s=50,
                    alpha=0.3,
                    label='Unvisited',
                    zorder=1
                )
            
            # Plot visited points by cluster
            unique_labels = set(labels[visited])
            colors = plt.cm.Spectral(np.linspace(0, 1, max(len(unique_labels), 1)))
            
            for label, color in zip(unique_labels, colors):
                if label == -1:
                    mask = visited & (labels == label)
                    if np.any(mask):
                        ax.scatter(
                            X[mask, 0], X[mask, 1],
                            c=self.config.noise_color,
                            marker=self.config.noise_marker,
                            s=self.config.noise_size,
                            alpha=0.6,
                            label='Noise',
                            zorder=2
                        )
                elif label == 0:
                    mask = visited & (labels == label)
                    if np.any(mask):
                        ax.scatter(
                            X[mask, 0], X[mask, 1],
                            c='gray',
                            marker='o',
                            s=50,
                            alpha=0.5,
                            label='Visited (unassigned)',
                            zorder=2
                        )
                else:
                    mask = visited & (labels == label)
                    if np.any(mask):
                        ax.scatter(
                            X[mask, 0], X[mask, 1],
                            c=[color],
                            marker='o',
                            s=60,
                            alpha=0.7,
                            label=f'Cluster {label}',
                            zorder=2
                        )
            
            # Draw epsilon-neighborhood
            if current_point is not None:
                circle = plt.Circle(
                    (X[current_point, 0], X[current_point, 1]),
                    eps,
                    color='red',
                    fill=False,
                    linestyle='--',
                    linewidth=2.5,
                    alpha=0.6,
                    zorder=3
                )
                ax.add_patch(circle)
                
                # Highlight neighbors
                if current_neighbors is not None and len(current_neighbors) > 0:
                    neighbors_to_show = current_neighbors[current_neighbors != current_point]
                    if len(neighbors_to_show) > 0:
                        ax.scatter(
                            X[neighbors_to_show, 0], X[neighbors_to_show, 1],
                            facecolors='none',
                            edgecolors='orange',
                            s=150,
                            linewidths=2,
                            zorder=4
                        )
                
                # Highlight current point
                ax.scatter(
                    X[current_point, 0], X[current_point, 1],
                    c='red',
                    marker='*',
                    s=400,
                    edgecolors='black',
                    linewidths=2,
                    zorder=5
                )
            
            # Add step number
            ax.text(
                0.02, 0.02,
                f'Step {step_num}',
                transform=ax.transAxes,
                fontsize=14,
                fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                zorder=6
            )
            
            ax.set_title(f'DBSCAN Algorithm Animation (ε={eps:.2f}, MinPts={min_pts})')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            
            if self.config.show_grid:
                ax.grid(True, alpha=self.config.grid_alpha)
            
            # Only show legend for first few unique labels to avoid clutter
            if len(unique_labels) <= 5:
                ax.legend(loc='best', fontsize=8)
        
        # Create animation
        anim = FuncAnimation(
            fig,
            update,
            frames=len(states),
            interval=interval,
            repeat=True,
            blit=False
        )
        
        # Save if path provided
        if save_path is not None:
            anim.save(save_path, writer='pillow' if save_path.endswith('.gif') else 'ffmpeg')
        
        plt.tight_layout()
        return anim
    
    def _capture_algorithm_states(
        self,
        X: np.ndarray,
        eps: float,
        min_pts: int
    ) -> list:
        """
        Run DBSCAN and capture the state at each step for animation.
        
        Parameters
        ----------
        X : np.ndarray
            Data array of shape (n_samples, n_features)
        eps : float
            Epsilon radius
        min_pts : int
            Minimum points for core point
        
        Returns
        -------
        list
            List of state dictionaries, each containing:
            - current_point: Index of point being processed
            - visited: Boolean array of visited points
            - labels: Current cluster labels
            - step_num: Step number
            - neighbors: Indices of current point's neighbors
        """
        n_points = len(X)
        labels = np.zeros(n_points, dtype=int)
        visited = np.zeros(n_points, dtype=bool)
        cluster_id = 0
        states = []
        step_num = 0
        
        # Helper function to compute distance
        def compute_distance(p1, p2):
            return np.sqrt(np.sum((p1 - p2) ** 2))
        
        # Helper function to find neighbors
        def get_neighbors(point_idx):
            neighbors = []
            for idx in range(n_points):
                if compute_distance(X[point_idx], X[idx]) <= eps:
                    neighbors.append(idx)
            return np.array(neighbors)
        
        # Initial state
        states.append({
            'current_point': None,
            'visited': visited.copy(),
            'labels': labels.copy(),
            'step_num': step_num,
            'neighbors': None
        })
        
        # Process each point
        for point_idx in range(n_points):
            if visited[point_idx]:
                continue
            
            step_num += 1
            visited[point_idx] = True
            neighbors = get_neighbors(point_idx)
            
            # Capture state when visiting point
            states.append({
                'current_point': point_idx,
                'visited': visited.copy(),
                'labels': labels.copy(),
                'step_num': step_num,
                'neighbors': neighbors.copy()
            })
            
            # Check if core point
            if len(neighbors) < min_pts:
                labels[point_idx] = -1
                step_num += 1
                states.append({
                    'current_point': point_idx,
                    'visited': visited.copy(),
                    'labels': labels.copy(),
                    'step_num': step_num,
                    'neighbors': neighbors.copy()
                })
            else:
                # Create new cluster
                cluster_id += 1
                labels[point_idx] = cluster_id
                
                # Expand cluster
                i = 0
                neighbor_list = list(neighbors)
                
                while i < len(neighbor_list):
                    neighbor_idx = neighbor_list[i]
                    
                    if labels[neighbor_idx] == -1:
                        labels[neighbor_idx] = cluster_id
                    
                    if not visited[neighbor_idx]:
                        visited[neighbor_idx] = True
                        neighbor_neighbors = get_neighbors(neighbor_idx)
                        
                        step_num += 1
                        states.append({
                            'current_point': neighbor_idx,
                            'visited': visited.copy(),
                            'labels': labels.copy(),
                            'step_num': step_num,
                            'neighbors': neighbor_neighbors.copy()
                        })
                        
                        if len(neighbor_neighbors) >= min_pts:
                            neighbor_list.extend(neighbor_neighbors)
                        
                        labels[neighbor_idx] = cluster_id
                    
                    i += 1
        
        # Final state
        step_num += 1
        states.append({
            'current_point': None,
            'visited': visited.copy(),
            'labels': labels.copy(),
            'step_num': step_num,
            'neighbors': None
        })
        
        return states

    def plot_algorithm_comparison(
        self,
        X: np.ndarray,
        algorithms_dict: dict,
        title: str = "Algorithm Comparison"
    ):
        """
        Create side-by-side comparison of different clustering algorithms.
        
        Visualizes results from multiple clustering algorithms on the same dataset
        to compare their behavior, cluster shapes, and noise handling [Requirement 10.3].
        
        Parameters
        ----------
        X : np.ndarray
            Data array of shape (n_samples, n_features)
        algorithms_dict : dict
            Dictionary mapping algorithm names to their cluster labels
            Example: {'DBSCAN': labels1, 'K-Means': labels2, 'OPTICS': labels3}
        title : str, default="Algorithm Comparison"
            Overall title for the comparison plot
        
        Returns
        -------
        matplotlib.pyplot
            The pyplot module for further customization or display
        
        Notes
        -----
        This visualization helps understand the strengths and weaknesses of
        different clustering approaches. DBSCAN excels at arbitrary shapes and
        noise detection, while K-Means assumes spherical clusters.
        
        Examples
        --------
        >>> from sklearn.cluster import KMeans, OPTICS
        >>> from src.dbscan_from_scratch import DBSCAN
        >>> 
        >>> dbscan = DBSCAN(eps=0.3, min_pts=5)
        >>> kmeans = KMeans(n_clusters=2)
        >>> optics = OPTICS(min_samples=5)
        >>> 
        >>> algorithms = {
        ...     'DBSCAN': dbscan.fit_predict(X),
        ...     'K-Means': kmeans.fit_predict(X),
        ...     'OPTICS': optics.fit_predict(X)
        ... }
        >>> 
        >>> visualizer = DBSCANVisualizer()
        >>> visualizer.plot_algorithm_comparison(X, algorithms)
        """
        n_algorithms = len(algorithms_dict)
        
        # Create subplots
        fig, axes = plt.subplots(1, n_algorithms, figsize=(6 * n_algorithms, 5), dpi=self.config.dpi)
        
        # Handle single algorithm case
        if n_algorithms == 1:
            axes = [axes]
        
        # Plot each algorithm's results
        for idx, (alg_name, labels) in enumerate(algorithms_dict.items()):
            ax = axes[idx]
            
            # Get unique labels
            unique_labels = set(labels)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
            
            # Count clusters and noise
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = np.sum(labels == -1) if -1 in labels else 0
            
            # Plot each cluster
            for label, color in zip(unique_labels, colors):
                if label == -1:
                    # Noise points
                    mask = labels == label
                    ax.scatter(
                        X[mask, 0], X[mask, 1],
                        c=self.config.noise_color,
                        marker=self.config.noise_marker,
                        s=self.config.noise_size,
                        alpha=0.6,
                        label='Noise'
                    )
                else:
                    # Cluster points
                    mask = labels == label
                    ax.scatter(
                        X[mask, 0], X[mask, 1],
                        c=[color],
                        marker='o',
                        s=60,
                        alpha=0.7,
                        label=f'Cluster {label}'
                    )
            
            # Set subplot title with statistics
            ax.set_title(f'{alg_name}\n({n_clusters} clusters, {n_noise} noise)')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            
            if self.config.show_grid:
                ax.grid(True, alpha=self.config.grid_alpha)
            
            # Only show legend if not too many clusters
            if len(unique_labels) <= 6:
                ax.legend(fontsize=8, loc='best')
        
        # Overall title
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return plt
    
    def plot_parameter_sensitivity(
        self,
        X: np.ndarray,
        eps_range: list,
        minpts_range: list,
        title: str = "Parameter Sensitivity Analysis"
    ):
        """
        Visualize how eps and min_pts parameters affect clustering results.
        
        Creates a grid of subplots showing DBSCAN results for different
        parameter combinations, helping users understand parameter sensitivity
        [Requirements 3.9, 4.3].
        
        Parameters
        ----------
        X : np.ndarray
            Data array of shape (n_samples, n_features)
        eps_range : list
            List of epsilon values to test (e.g., [0.1, 0.3, 0.5])
        minpts_range : list
            List of min_pts values to test (e.g., [3, 5, 7])
        title : str, default="Parameter Sensitivity Analysis"
            Overall title for the plot
        
        Returns
        -------
        matplotlib.pyplot
            The pyplot module for further customization or display
        
        Notes
        -----
        This visualization demonstrates:
        - Small eps: More noise, fragmented clusters
        - Large eps: Merged clusters, less noise
        - Small min_pts: More clusters, less noise
        - Large min_pts: Fewer clusters, more noise
        
        Examples
        --------
        >>> visualizer = DBSCANVisualizer()
        >>> eps_values = [0.2, 0.3, 0.4]
        >>> minpts_values = [3, 5, 7]
        >>> visualizer.plot_parameter_sensitivity(X, eps_values, minpts_values)
        """
        from src.dbscan_from_scratch import DBSCAN
        
        n_eps = len(eps_range)
        n_minpts = len(minpts_range)
        
        # Create grid of subplots
        fig, axes = plt.subplots(
            n_minpts, n_eps,
            figsize=(5 * n_eps, 4 * n_minpts),
            dpi=self.config.dpi
        )
        
        # Handle single row/column cases
        if n_minpts == 1 and n_eps == 1:
            axes = np.array([[axes]])
        elif n_minpts == 1:
            axes = axes.reshape(1, -1)
        elif n_eps == 1:
            axes = axes.reshape(-1, 1)
        
        # Test each parameter combination
        for i, min_pts in enumerate(minpts_range):
            for j, eps in enumerate(eps_range):
                ax = axes[i, j]
                
                # Run DBSCAN with current parameters
                dbscan = DBSCAN(eps=eps, min_pts=min_pts)
                labels = dbscan.fit_predict(X)
                
                # Get unique labels
                unique_labels = set(labels)
                colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
                
                # Count clusters and noise
                n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                n_noise = np.sum(labels == -1)
                
                # Plot results
                for label, color in zip(unique_labels, colors):
                    if label == -1:
                        mask = labels == label
                        ax.scatter(
                            X[mask, 0], X[mask, 1],
                            c=self.config.noise_color,
                            marker=self.config.noise_marker,
                            s=30,
                            alpha=0.5
                        )
                    else:
                        mask = labels == label
                        ax.scatter(
                            X[mask, 0], X[mask, 1],
                            c=[color],
                            marker='o',
                            s=40,
                            alpha=0.7
                        )
                
                # Set subplot title
                ax.set_title(
                    f'ε={eps:.2f}, MinPts={min_pts}\n'
                    f'{n_clusters} clusters, {n_noise} noise',
                    fontsize=9
                )
                ax.set_xlabel('Feature 1', fontsize=8)
                ax.set_ylabel('Feature 2', fontsize=8)
                
                if self.config.show_grid:
                    ax.grid(True, alpha=self.config.grid_alpha)
        
        # Overall title
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return plt
    
    def plot_cluster_shapes(
        self,
        datasets_dict: dict,
        eps: float = 0.3,
        min_pts: int = 5,
        title: str = "DBSCAN on Various Cluster Shapes"
    ):
        """
        Demonstrate DBSCAN performance on datasets with different cluster geometries.
        
        Shows how DBSCAN handles various cluster shapes (moons, circles, blobs,
        elongated clusters) compared to algorithms that assume spherical clusters
        [Requirements 3.9, 10.8].
        
        Parameters
        ----------
        datasets_dict : dict
            Dictionary mapping dataset names to data arrays
            Example: {'Moons': X1, 'Circles': X2, 'Blobs': X3}
        eps : float, default=0.3
            Epsilon parameter for DBSCAN
        min_pts : int, default=5
            Minimum points parameter for DBSCAN
        title : str, default="DBSCAN on Various Cluster Shapes"
            Overall title for the plot
        
        Returns
        -------
        matplotlib.pyplot
            The pyplot module for further customization or display
        
        Notes
        -----
        DBSCAN's strength is handling arbitrary cluster shapes. This visualization
        demonstrates its ability to find non-convex clusters, which algorithms
        like K-Means struggle with.
        
        Examples
        --------
        >>> from sklearn.datasets import make_moons, make_circles, make_blobs
        >>> 
        >>> datasets = {
        ...     'Moons': make_moons(n_samples=200, noise=0.05)[0],
        ...     'Circles': make_circles(n_samples=200, noise=0.05, factor=0.5)[0],
        ...     'Blobs': make_blobs(n_samples=200, centers=3)[0]
        ... }
        >>> 
        >>> visualizer = DBSCANVisualizer()
        >>> visualizer.plot_cluster_shapes(datasets, eps=0.3, min_pts=5)
        """
        from src.dbscan_from_scratch import DBSCAN
        
        n_datasets = len(datasets_dict)
        
        # Create subplots
        fig, axes = plt.subplots(
            1, n_datasets,
            figsize=(6 * n_datasets, 5),
            dpi=self.config.dpi
        )
        
        # Handle single dataset case
        if n_datasets == 1:
            axes = [axes]
        
        # Process each dataset
        for idx, (dataset_name, X) in enumerate(datasets_dict.items()):
            ax = axes[idx]
            
            # Run DBSCAN
            dbscan = DBSCAN(eps=eps, min_pts=min_pts)
            labels = dbscan.fit_predict(X)
            
            # Get unique labels
            unique_labels = set(labels)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
            
            # Count clusters and noise
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = np.sum(labels == -1)
            
            # Plot results
            for label, color in zip(unique_labels, colors):
                if label == -1:
                    mask = labels == label
                    ax.scatter(
                        X[mask, 0], X[mask, 1],
                        c=self.config.noise_color,
                        marker=self.config.noise_marker,
                        s=self.config.noise_size,
                        alpha=0.6,
                        label='Noise'
                    )
                else:
                    mask = labels == label
                    ax.scatter(
                        X[mask, 0], X[mask, 1],
                        c=[color],
                        marker='o',
                        s=60,
                        alpha=0.7,
                        label=f'Cluster {label}'
                    )
            
            # Set subplot title
            ax.set_title(
                f'{dataset_name}\n'
                f'{n_clusters} clusters, {n_noise} noise',
                fontsize=11
            )
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            
            if self.config.show_grid:
                ax.grid(True, alpha=self.config.grid_alpha)
            
            # Show legend if not too many clusters
            if len(unique_labels) <= 5:
                ax.legend(fontsize=8, loc='best')
        
        # Overall title
        fig.suptitle(
            f'{title}\n(ε={eps:.2f}, MinPts={min_pts})',
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()
        
        return plt
    
    def plot_density_variations(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        eps: float,
        title: str = "Density Variations with Heatmap"
    ):
        """
        Visualize density variations in the dataset using a density heatmap.
        
        Creates a visualization showing both the clustering results and the
        underlying density distribution of the data, helping understand how
        DBSCAN identifies dense regions [Requirements 3.10, 10.8].
        
        Parameters
        ----------
        X : np.ndarray
            Data array of shape (n_samples, n_features)
        labels : np.ndarray
            Cluster labels from DBSCAN
        eps : float
            Epsilon parameter used for clustering
        title : str, default="Density Variations with Heatmap"
            Plot title
        
        Returns
        -------
        matplotlib.pyplot
            The pyplot module for further customization or display
        
        Notes
        -----
        The density heatmap is computed using kernel density estimation (KDE)
        or by counting points in a grid. This visualization helps understand:
        - Why certain regions form clusters (high density)
        - Why certain points are noise (low density)
        - How density varies across the dataset
        
        Examples
        --------
        >>> from src.dbscan_from_scratch import DBSCAN
        >>> 
        >>> dbscan = DBSCAN(eps=0.3, min_pts=5)
        >>> labels = dbscan.fit_predict(X)
        >>> 
        >>> visualizer = DBSCANVisualizer()
        >>> visualizer.plot_density_variations(X, labels, eps=0.3)
        """
        from scipy.stats import gaussian_kde
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(14, 6),
            dpi=self.config.dpi
        )
        
        # Left plot: Clustering results
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                mask = labels == label
                ax1.scatter(
                    X[mask, 0], X[mask, 1],
                    c=self.config.noise_color,
                    marker=self.config.noise_marker,
                    s=self.config.noise_size,
                    alpha=0.6,
                    label='Noise',
                    zorder=2
                )
            else:
                mask = labels == label
                ax1.scatter(
                    X[mask, 0], X[mask, 1],
                    c=[color],
                    marker='o',
                    s=60,
                    alpha=0.7,
                    label=f'Cluster {label}',
                    zorder=2
                )
        
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(labels == -1)
        
        ax1.set_title(
            f'DBSCAN Results\n{n_clusters} clusters, {n_noise} noise',
            fontsize=11
        )
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        
        if self.config.show_grid:
            ax1.grid(True, alpha=self.config.grid_alpha)
        
        if len(unique_labels) <= 6:
            ax1.legend(fontsize=8, loc='best')
        
        # Right plot: Density heatmap
        try:
            # Use KDE for smooth density estimation
            kde = gaussian_kde(X.T)
            
            # Create grid for density evaluation
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 100),
                np.linspace(y_min, y_max, 100)
            )
            
            # Evaluate density
            positions = np.vstack([xx.ravel(), yy.ravel()])
            density = kde(positions).reshape(xx.shape)
            
            # Plot density heatmap
            im = ax2.contourf(xx, yy, density, levels=20, cmap='YlOrRd', alpha=0.8)
            
            # Overlay data points
            ax2.scatter(
                X[:, 0], X[:, 1],
                c='black',
                s=10,
                alpha=0.3,
                zorder=2
            )
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label('Density', rotation=270, labelpad=15)
            
        except Exception as e:
            # Fallback to simple histogram-based density
            ax2.hist2d(
                X[:, 0], X[:, 1],
                bins=30,
                cmap='YlOrRd',
                alpha=0.8
            )
            
            # Overlay data points
            ax2.scatter(
                X[:, 0], X[:, 1],
                c='black',
                s=10,
                alpha=0.3,
                zorder=2
            )
            
            cbar = plt.colorbar(ax=ax2)
            cbar.set_label('Point Count', rotation=270, labelpad=15)
        
        ax2.set_title(
            f'Density Distribution\n(ε={eps:.2f})',
            fontsize=11
        )
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')
        
        if self.config.show_grid:
            ax2.grid(True, alpha=self.config.grid_alpha)
        
        # Overall title
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return plt

    def plot_distance_metrics(
        self,
        point_a: np.ndarray,
        point_b: np.ndarray,
        metrics: list = None,
        title: str = "Distance Metrics Comparison"
    ):
        """
        Visualize different distance metrics geometrically.
        
        Shows how different distance metrics (Euclidean, Manhattan, Chebyshev)
        measure distance between two points, illustrating the geometric
        interpretation of each metric [Requirement 8.6].
        
        Parameters
        ----------
        point_a : np.ndarray
            First point coordinates (2D)
        point_b : np.ndarray
            Second point coordinates (2D)
        metrics : list, default=None
            List of metrics to visualize. If None, uses ['euclidean', 'manhattan', 'chebyshev']
        title : str, default="Distance Metrics Comparison"
            Plot title
        
        Returns
        -------
        matplotlib.pyplot
            The pyplot module for further customization or display
        
        Notes
        -----
        Distance metrics define how DBSCAN measures similarity between points:
        - Euclidean: Straight-line distance (L2 norm)
        - Manhattan: Sum of absolute differences (L1 norm)
        - Chebyshev: Maximum absolute difference (L∞ norm)
        
        Examples
        --------
        >>> visualizer = DBSCANVisualizer()
        >>> p1 = np.array([0, 0])
        >>> p2 = np.array([3, 4])
        >>> visualizer.plot_distance_metrics(p1, p2)
        """
        if metrics is None:
            metrics = ['euclidean', 'manhattan', 'chebyshev']
        
        n_metrics = len(metrics)
        
        # Create subplots
        fig, axes = plt.subplots(
            1, n_metrics,
            figsize=(6 * n_metrics, 5),
            dpi=self.config.dpi
        )
        
        # Handle single metric case
        if n_metrics == 1:
            axes = [axes]
        
        # Compute distances
        distances = {}
        if 'euclidean' in metrics:
            distances['euclidean'] = np.sqrt(np.sum((point_b - point_a) ** 2))
        if 'manhattan' in metrics:
            distances['manhattan'] = np.sum(np.abs(point_b - point_a))
        if 'chebyshev' in metrics:
            distances['chebyshev'] = np.max(np.abs(point_b - point_a))
        
        # Plot each metric
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Plot points
            ax.scatter(*point_a, c='blue', s=200, marker='o', 
                      edgecolors='black', linewidths=2, label='Point A', zorder=5)
            ax.scatter(*point_b, c='red', s=200, marker='o', 
                      edgecolors='black', linewidths=2, label='Point B', zorder=5)
            
            # Add point labels
            ax.text(point_a[0], point_a[1] - 0.3, 'A', 
                   fontsize=12, ha='center', fontweight='bold')
            ax.text(point_b[0], point_b[1] - 0.3, 'B', 
                   fontsize=12, ha='center', fontweight='bold')
            
            # Visualize distance based on metric
            if metric == 'euclidean':
                # Draw straight line
                ax.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], 
                       'g-', linewidth=3, alpha=0.7, label=f'd = {distances[metric]:.2f}')
                
                # Draw right triangle to show Pythagorean theorem
                ax.plot([point_a[0], point_b[0]], [point_a[1], point_a[1]], 
                       'gray', linestyle='--', linewidth=1, alpha=0.5)
                ax.plot([point_b[0], point_b[0]], [point_a[1], point_b[1]], 
                       'gray', linestyle='--', linewidth=1, alpha=0.5)
                
                # Add formula
                formula = r'$d = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$'
                ax.text(0.5, 0.95, formula, transform=ax.transAxes, 
                       fontsize=10, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
            elif metric == 'manhattan':
                # Draw L-shaped path
                ax.plot([point_a[0], point_b[0]], [point_a[1], point_a[1]], 
                       'g-', linewidth=3, alpha=0.7)
                ax.plot([point_b[0], point_b[0]], [point_a[1], point_b[1]], 
                       'g-', linewidth=3, alpha=0.7, label=f'd = {distances[metric]:.2f}')
                
                # Add arrows to show direction
                mid_x = (point_a[0] + point_b[0]) / 2
                mid_y = (point_a[1] + point_b[1]) / 2
                ax.annotate('', xy=(point_b[0], point_a[1]), xytext=(point_a[0], point_a[1]),
                           arrowprops=dict(arrowstyle='->', color='green', lw=2))
                ax.annotate('', xy=(point_b[0], point_b[1]), xytext=(point_b[0], point_a[1]),
                           arrowprops=dict(arrowstyle='->', color='green', lw=2))
                
                # Add formula
                formula = r'$d = |x_2-x_1| + |y_2-y_1|$'
                ax.text(0.5, 0.95, formula, transform=ax.transAxes, 
                       fontsize=10, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
            elif metric == 'chebyshev':
                # Draw square showing maximum coordinate difference
                max_diff = np.max(np.abs(point_b - point_a))
                
                # Draw the square
                square_x = [point_a[0], point_a[0] + max_diff, point_a[0] + max_diff, point_a[0], point_a[0]]
                square_y = [point_a[1], point_a[1], point_a[1] + max_diff, point_a[1] + max_diff, point_a[1]]
                ax.plot(square_x, square_y, 'g-', linewidth=2, alpha=0.5)
                ax.fill(square_x, square_y, color='green', alpha=0.1)
                
                # Draw line to point B
                ax.plot([point_a[0], point_b[0]], [point_a[1], point_b[1]], 
                       'g--', linewidth=2, alpha=0.7, label=f'd = {distances[metric]:.2f}')
                
                # Add formula
                formula = r'$d = \max(|x_2-x_1|, |y_2-y_1|)$'
                ax.text(0.5, 0.95, formula, transform=ax.transAxes, 
                       fontsize=10, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Set subplot properties
            ax.set_title(f'{metric.capitalize()} Distance', fontsize=12, fontweight='bold')
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            ax.set_aspect('equal', adjustable='box')
            
            if self.config.show_grid:
                ax.grid(True, alpha=self.config.grid_alpha)
            
            ax.legend(loc='best', fontsize=9)
            
            # Set axis limits with padding
            all_points = np.vstack([point_a, point_b])
            x_min, x_max = all_points[:, 0].min() - 1, all_points[:, 0].max() + 1
            y_min, y_max = all_points[:, 1].min() - 1, all_points[:, 1].max() + 1
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        
        # Overall title
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return plt
    
    def plot_complexity_analysis(
        self,
        sizes: np.ndarray,
        times: np.ndarray,
        algorithm_name: str = "DBSCAN",
        theoretical_complexity: str = "O(n²)",
        title: str = "Algorithm Complexity Analysis"
    ):
        """
        Visualize algorithm complexity with empirical and theoretical curves.
        
        Plots measured runtime against dataset size along with theoretical
        complexity curves to demonstrate algorithmic performance characteristics
        [Requirements 8.7, 15.1].
        
        Parameters
        ----------
        sizes : np.ndarray
            Array of dataset sizes tested
        times : np.ndarray
            Array of measured execution times (in seconds)
        algorithm_name : str, default="DBSCAN"
            Name of the algorithm being analyzed
        theoretical_complexity : str, default="O(n²)"
            Theoretical complexity notation (e.g., "O(n²)", "O(n log n)")
        title : str, default="Algorithm Complexity Analysis"
            Plot title
        
        Returns
        -------
        matplotlib.pyplot
            The pyplot module for further customization or display
        
        Notes
        -----
        DBSCAN complexity:
        - Naive implementation: O(n²) - checks all point pairs
        - With spatial indexing: O(n log n) - uses R-tree or KD-tree
        
        This visualization helps understand:
        - How runtime scales with dataset size
        - Whether implementation matches theoretical complexity
        - When spatial indexing becomes beneficial
        
        Examples
        --------
        >>> sizes = np.array([100, 500, 1000, 2000, 5000])
        >>> times = np.array([0.01, 0.25, 1.0, 4.0, 25.0])
        >>> visualizer = DBSCANVisualizer()
        >>> visualizer.plot_complexity_analysis(sizes, times)
        """
        plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Plot empirical measurements
        plt.plot(sizes, times, 'bo-', linewidth=2, markersize=8, 
                label='Measured Runtime', zorder=3)
        
        # Generate theoretical curves
        # Normalize to match the scale of measured data
        if len(times) > 0 and len(sizes) > 0:
            # Use first data point for normalization
            scale_factor = times[0] / (sizes[0] ** 2) if theoretical_complexity == "O(n²)" else times[0] / (sizes[0] * np.log(sizes[0]))
            
            # Generate smooth curve
            smooth_sizes = np.linspace(sizes.min(), sizes.max(), 100)
            
            if theoretical_complexity == "O(n²)":
                theoretical_times = scale_factor * smooth_sizes ** 2
                plt.plot(smooth_sizes, theoretical_times, 'r--', linewidth=2, 
                        alpha=0.7, label=f'Theoretical {theoretical_complexity}', zorder=2)
            elif theoretical_complexity == "O(n log n)":
                theoretical_times = scale_factor * smooth_sizes * np.log(smooth_sizes)
                plt.plot(smooth_sizes, theoretical_times, 'r--', linewidth=2, 
                        alpha=0.7, label=f'Theoretical {theoretical_complexity}', zorder=2)
            elif theoretical_complexity == "O(n)":
                theoretical_times = scale_factor * smooth_sizes
                plt.plot(smooth_sizes, theoretical_times, 'r--', linewidth=2, 
                        alpha=0.7, label=f'Theoretical {theoretical_complexity}', zorder=2)
            
            # Add reference curves for comparison
            if theoretical_complexity == "O(n²)":
                # Add O(n log n) for comparison
                scale_nlogn = times[0] / (sizes[0] * np.log(sizes[0]))
                comparison_times = scale_nlogn * smooth_sizes * np.log(smooth_sizes)
                plt.plot(smooth_sizes, comparison_times, 'g:', linewidth=2, 
                        alpha=0.5, label='O(n log n) (with indexing)', zorder=1)
        
        plt.xlabel('Dataset Size (n)', fontsize=11)
        plt.ylabel('Runtime (seconds)', fontsize=11)
        plt.title(f'{title}\n{algorithm_name} - {theoretical_complexity}', 
                 fontsize=12, fontweight='bold')
        
        if self.config.show_legend:
            plt.legend(loc='best', fontsize=10)
        
        if self.config.show_grid:
            plt.grid(True, alpha=self.config.grid_alpha)
        
        # Add annotation explaining complexity
        textstr = (
            f'Complexity: {theoretical_complexity}\n'
            f'Naive DBSCAN: O(n²)\n'
            f'With spatial index: O(n log n)'
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(
            0.98, 0.02, textstr,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=props
        )
        
        plt.tight_layout()
        return plt
    
    def plot_scalability_benchmark(
        self,
        results_df,
        title: str = "DBSCAN Scalability Benchmark"
    ):
        """
        Visualize scalability benchmark results from a results dataframe.
        
        Creates comprehensive visualization of benchmark results showing how
        DBSCAN performance scales with dataset size, including runtime,
        memory usage, and cluster quality metrics [Requirements 8.8, 15.6].
        
        Parameters
        ----------
        results_df : pandas.DataFrame
            Benchmark results with columns:
            - 'n_samples': Dataset size
            - 'runtime': Execution time in seconds
            - 'memory_mb': Memory usage in megabytes (optional)
            - 'n_clusters': Number of clusters found (optional)
            - 'n_noise': Number of noise points (optional)
        title : str, default="DBSCAN Scalability Benchmark"
            Plot title
        
        Returns
        -------
        matplotlib.pyplot
            The pyplot module for further customization or display
        
        Notes
        -----
        This visualization provides a comprehensive view of DBSCAN performance:
        - Runtime scaling: How execution time grows with data size
        - Memory usage: RAM requirements for different dataset sizes
        - Cluster statistics: How clustering results vary with scale
        
        Examples
        --------
        >>> import pandas as pd
        >>> results = pd.DataFrame({
        ...     'n_samples': [100, 500, 1000, 5000, 10000],
        ...     'runtime': [0.01, 0.25, 1.0, 25.0, 100.0],
        ...     'memory_mb': [10, 15, 25, 80, 200],
        ...     'n_clusters': [3, 3, 3, 3, 3],
        ...     'n_noise': [5, 20, 40, 200, 400]
        ... })
        >>> visualizer = DBSCANVisualizer()
        >>> visualizer.plot_scalability_benchmark(results)
        """
        import pandas as pd
        
        # Ensure we have a DataFrame
        if not isinstance(results_df, pd.DataFrame):
            raise TypeError("results_df must be a pandas DataFrame")
        
        # Check required columns
        if 'n_samples' not in results_df.columns or 'runtime' not in results_df.columns:
            raise ValueError("results_df must contain 'n_samples' and 'runtime' columns")
        
        # Determine number of subplots based on available data
        has_memory = 'memory_mb' in results_df.columns
        has_clusters = 'n_clusters' in results_df.columns and 'n_noise' in results_df.columns
        
        n_plots = 1 + (1 if has_memory else 0) + (1 if has_clusters else 0)
        
        # Create subplots
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), dpi=self.config.dpi)
        
        # Handle single plot case
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Plot 1: Runtime scaling
        ax = axes[plot_idx]
        ax.plot(results_df['n_samples'], results_df['runtime'], 
               'bo-', linewidth=2, markersize=8, label='Measured Runtime')
        
        # Add theoretical O(n²) curve
        sizes = results_df['n_samples'].values
        times = results_df['runtime'].values
        if len(times) > 0 and len(sizes) > 0:
            scale_factor = times[0] / (sizes[0] ** 2)
            theoretical_times = scale_factor * sizes ** 2
            ax.plot(sizes, theoretical_times, 'r--', linewidth=2, 
                   alpha=0.7, label='Theoretical O(n²)')
        
        ax.set_xlabel('Dataset Size (n)', fontsize=11)
        ax.set_ylabel('Runtime (seconds)', fontsize=11)
        ax.set_title('Runtime Scaling', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        
        if self.config.show_grid:
            ax.grid(True, alpha=self.config.grid_alpha)
        
        plot_idx += 1
        
        # Plot 2: Memory usage (if available)
        if has_memory:
            ax = axes[plot_idx]
            ax.plot(results_df['n_samples'], results_df['memory_mb'], 
                   'go-', linewidth=2, markersize=8, label='Memory Usage')
            
            # Add theoretical O(n) curve for memory
            if len(sizes) > 0:
                scale_factor = results_df['memory_mb'].iloc[0] / sizes[0]
                theoretical_memory = scale_factor * sizes
                ax.plot(sizes, theoretical_memory, 'r--', linewidth=2, 
                       alpha=0.7, label='Theoretical O(n)')
            
            ax.set_xlabel('Dataset Size (n)', fontsize=11)
            ax.set_ylabel('Memory Usage (MB)', fontsize=11)
            ax.set_title('Memory Scaling', fontsize=11, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            
            if self.config.show_grid:
                ax.grid(True, alpha=self.config.grid_alpha)
            
            plot_idx += 1
        
        # Plot 3: Cluster statistics (if available)
        if has_clusters:
            ax = axes[plot_idx]
            
            # Plot clusters and noise on same axis with different y-axes
            ax.plot(results_df['n_samples'], results_df['n_clusters'], 
                   'bs-', linewidth=2, markersize=8, label='Clusters')
            ax.set_xlabel('Dataset Size (n)', fontsize=11)
            ax.set_ylabel('Number of Clusters', fontsize=11, color='blue')
            ax.tick_params(axis='y', labelcolor='blue')
            
            # Create second y-axis for noise points
            ax2 = ax.twinx()
            ax2.plot(results_df['n_samples'], results_df['n_noise'], 
                    'ro-', linewidth=2, markersize=8, label='Noise Points')
            ax2.set_ylabel('Number of Noise Points', fontsize=11, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            ax.set_title('Clustering Statistics', fontsize=11, fontweight='bold')
            
            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
            
            if self.config.show_grid:
                ax.grid(True, alpha=self.config.grid_alpha)
        
        # Overall title
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return plt
