"""
OptiGraph Pipeline - Orchestrates PCA + K-Means
"""
import numpy as np
import sys
sys.path.append('/python/engine')

from engine.pca import PCAFromScratch
from engine.kmeans import KMeansFromScratch


def run_pipeline(X: np.ndarray, n_components=2, n_clusters=8):
    """
    Run complete ML pipeline: PCA dimensionality reduction + K-Means clustering.
    
    Args:
        X: Input data, shape (n_samples, n_features)
        n_components: Number of PCA components (typically 2 or 3 for visualization)
        n_clusters: Number of K-Means clusters
    
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
    # Step 1: PCA dimensionality reduction
    pca = PCAFromScratch(n_components=n_components)
    Z = pca.fit_transform(X)
    
    # Step 2: K-Means clustering in reduced space
    km = KMeansFromScratch(n_clusters=n_clusters, max_iter=100, tol=1e-4)
    labels = km.fit_predict(Z)
    
    # Calculate cumulative variance for interpretability
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    return {
        "points": Z.tolist(),
        "labels": labels.tolist(),
        "centroids": km.centroids_.tolist(),
        "inertia": km.inertia_,
        "inertia_history": km.inertia_history_,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": cumulative_variance.tolist(),
        "n_samples": X.shape[0],
        "n_features_original": X.shape[1],
        "n_components": n_components,
        "n_clusters": n_clusters,
    }


# Expose to JavaScript
class OptiGraphEngine:
    """Main interface for JS to call Python ML engine."""
    
    @staticmethod
    def process(data_array, n_components=2, n_clusters=8):
        """
        Process embeddings through PCA + K-Means pipeline.
        
        Args:
            data_array: 2D list or numpy array
            n_components: PCA components
            n_clusters: K-Means clusters
        
        Returns:
            Results dict
        """
        X = np.array(data_array, dtype=np.float64)
        return run_pipeline(X, n_components, n_clusters)
