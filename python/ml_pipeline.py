"""
OptiGraph Pipeline - Orchestrates PCA + K-Means
"""
import numpy as np
import sys
sys.path.append('/python/engine')

from engine.pca import PCAFromScratch
from engine.kmeans import KMeansFromScratch


def run_pipeline(X: np.ndarray, n_components=2, n_clusters=8, target_variance=0.85):
    """
    Run complete ML pipeline: PCA dimensionality reduction + K-Means clustering.
    
    Args:
        X: Input data, shape (n_samples, n_features)
        n_components: Number of PCA components (can be auto-adjusted)
        n_clusters: Number of K-Means clusters
        target_variance: Minimum cumulative variance to retain (0.85 = 85%)
    
    Returns:
        dict with:
            - points: 2D/3D coordinates after PCA
            - labels: Cluster assignments
            - centroids: Cluster centers in reduced space
            - inertia: Final inertia value
            - inertia_history: Convergence curve
            - explained_variance_ratio: PCA variance retention per component
            - cumulative_variance: Cumulative variance explained
    """
    # Step 1: PCA dimensionality reduction with automatic component selection
    pca = PCAFromScratch(n_components=min(n_components, X.shape[1] - 1))
    
    # IMPROVEMENT: Auto-adjust components to meet target variance
    initial_fit = PCAFromScratch(n_components=min(50, X.shape[1] - 1))
    initial_fit.fit(X)
    
    # Find minimum components needed for target variance
    cumulative_var = np.cumsum(initial_fit.explained_variance_ratio_)
    components_needed = np.argmax(cumulative_var >= target_variance) + 1
    
    # Use max of requested components and variance-based requirement (for visualization)
    final_n_components = max(n_components, min(components_needed, n_components + 2))
    
    if final_n_components != n_components:
        print(f"Info: Adjusted n_components from {n_components} to {final_n_components} "
              f"(target variance: {target_variance:.1%}, achieved: {cumulative_var[final_n_components-1]:.1%})")
    
    pca = PCAFromScratch(n_components=final_n_components)
    Z = pca.fit_transform(X)
    
    # For visualization, use only first 2-3 components
    Z_vis = Z[:, :n_components]
    
    # Step 2: K-Means clustering in reduced space (use all components for better clustering)
    km = KMeansFromScratch(n_clusters=n_clusters, max_iter=100, tol=1e-4)
    labels = km.fit_predict(Z)  # Use all components for clustering
    
    # Project centroids to visualization space
    centroids_vis = km.centroids_[:, :n_components]
    
    # Calculate cumulative variance for interpretability
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    return {
        "points": Z_vis.tolist(),  # Visualization space (2D/3D)
        "labels": labels.tolist(),
        "centroids": centroids_vis.tolist(),  # Centroids in visualization space
        "inertia": km.inertia_,
        "inertia_history": km.inertia_history_,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": cumulative_variance.tolist(),
        "n_samples": X.shape[0],
        "n_features_original": X.shape[1],
        "n_components": final_n_components,  # Actual components used
        "n_components_vis": n_components,    # Components for visualization
        "n_clusters": n_clusters,
        "variance_retained": float(cumulative_variance[final_n_components-1]),
    }


# Expose to JavaScript
class OptiGraphEngine:
    """Main interface for JS to call Python ML engine."""
    
    @staticmethod
    def process(data_array, n_components=2, n_clusters=8, target_variance=0.85):
        """
        Process embeddings through improved PCA + K-Means pipeline.
        
        Args:
            data_array: 2D list or numpy array
            n_components: PCA components for visualization (2D/3D)
            n_clusters: K-Means clusters
            target_variance: Minimum variance to retain (default 85%)
        
        Returns:
            Results dict with enhanced metrics
        """
        X = np.array(data_array, dtype=np.float64)
        return run_pipeline(X, n_components, n_clusters, target_variance)
