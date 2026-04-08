"""
Unit tests for K-Means implementation.
Validates algorithmic correctness and convergence properties.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.kmeans import KMeansFromScratch


class TestKMeansCorrectness:
    """Test algorithmic properties of K-Means."""
    
    def test_inertia_monotonic_decrease(self):
        """Inertia should decrease or stay constant across iterations."""
        X = np.random.randn(200, 10)
        kmeans = KMeansFromScratch(n_clusters=5, n_init=1, random_state=42)
        kmeans.fit(X)
        
        history = kmeans.inertia_history_
        for i in range(1, len(history)):
            # Inertia should never increase (Lloyd's guarantee)
            assert history[i] <= history[i-1] + 1e-9
    
    def test_reproducibility(self):
        """Same random_state should give identical results."""
        X = np.random.randn(100, 5)
        
        kmeans1 = KMeansFromScratch(n_clusters=3, random_state=42)
        kmeans2 = KMeansFromScratch(n_clusters=3, random_state=42)
        
        labels1 = kmeans1.fit_predict(X)
        labels2 = kmeans2.fit_predict(X)
        
        np.testing.assert_array_equal(labels1, labels2)
        np.testing.assert_allclose(kmeans1.centroids_, kmeans2.centroids_)
    
    def test_cluster_assignment_consistency(self):
        """Each point should be assigned to its nearest centroid."""
        X = np.random.randn(100, 5)
        kmeans = KMeansFromScratch(n_clusters=4, random_state=42)
        kmeans.fit(X)
        
        # Manually compute nearest centroid for each point
        from engine.kmeans import _pairwise_sq_dists
        d2 = _pairwise_sq_dists(X, kmeans.centroids_)
        expected_labels = np.argmin(d2, axis=1)
        
        np.testing.assert_array_equal(kmeans.labels_, expected_labels)
    
    def test_n_init_improves_results(self):
        """Multiple initializations should give same or better results."""
        X = np.random.randn(200, 10)
        
        # Single run
        kmeans_single = KMeansFromScratch(n_clusters=5, n_init=1, random_state=42)
        kmeans_single.fit(X)
        
        # Multiple runs
        kmeans_multi = KMeansFromScratch(n_clusters=5, n_init=10, random_state=42)
        kmeans_multi.fit(X)
        
        # Multi-run should have inertia <= single run
        assert kmeans_multi.inertia_ <= kmeans_single.inertia_ + 1e-6
    
    def test_predict_consistency(self):
        """Predict on training data should match labels_."""
        X = np.random.randn(100, 5)
        kmeans = KMeansFromScratch(n_clusters=4, random_state=42)
        kmeans.fit(X)
        
        predicted = kmeans.predict(X)
        np.testing.assert_array_equal(predicted, kmeans.labels_)


class TestKMeansEdgeCases:
    """Test edge cases and error handling."""
    
    def test_n_clusters_validation(self):
        """Should raise error if n_clusters > n_samples."""
        X = np.random.randn(10, 5)
        kmeans = KMeansFromScratch(n_clusters=15, random_state=42)
        
        with pytest.raises(ValueError, match="must be <="):
            kmeans.fit(X)
    
    def test_nan_input(self):
        """Should raise error on NaN input."""
        X = np.random.randn(50, 10)
        X[5, 3] = np.nan
        kmeans = KMeansFromScratch(n_clusters=3, random_state=42)
        
        with pytest.raises(ValueError, match="NaN"):
            kmeans.fit(X)
    
    def test_predict_before_fit(self):
        """Should raise error if predict called before fit."""
        X = np.random.randn(50, 10)
        kmeans = KMeansFromScratch(n_clusters=3, random_state=42)
        
        with pytest.raises(RuntimeError, match="Call fit"):
            kmeans.predict(X)
    
    def test_single_cluster(self):
        """Should work with n_clusters=1."""
        X = np.random.randn(50, 5)
        kmeans = KMeansFromScratch(n_clusters=1, random_state=42)
        kmeans.fit(X)
        
        # All points should be in cluster 0
        assert np.all(kmeans.labels_ == 0)
        
        # Centroid should be mean of all points
        expected_centroid = X.mean(axis=0)
        np.testing.assert_allclose(kmeans.centroids_[0], expected_centroid, rtol=1e-6)


class TestKMeansPlusPlusInitialization:
    """Test K-Means++ initialization quality."""
    
    def test_spread_initialization(self):
        """K-Means++ should spread initial centroids far apart."""
        # Create well-separated clusters
        np.random.seed(42)
        cluster1 = np.random.randn(50, 2) + np.array([0, 0])
        cluster2 = np.random.randn(50, 2) + np.array([10, 0])
        cluster3 = np.random.randn(50, 2) + np.array([5, 10])
        X = np.vstack([cluster1, cluster2, cluster3])
        
        kmeans = KMeansFromScratch(n_clusters=3, max_iter=0, random_state=42)
        # Run fit with max_iter=0 to test just initialization
        kmeans._rng = np.random.default_rng(42)
        init_centroids = kmeans._init_kmeanspp(X, kmeans._rng)
        
        # Initial centroids should be reasonably far apart
        from engine.kmeans import _pairwise_sq_dists
        d2 = _pairwise_sq_dists(init_centroids, init_centroids)
        
        # Set diagonal to inf to ignore self-distances
        d2[np.diag_indices_from(d2)] = np.inf
        min_dist = np.sqrt(np.min(d2))
        
        # Minimum distance should be > 5 (clusters are 10 units apart)
        assert min_dist > 5.0


class TestKMeansConvergence:
    """Test convergence behavior."""
    
    def test_converges_on_well_separated_data(self):
        """Should converge quickly on well-separated clusters."""
        # Create 4 well-separated Gaussian blobs
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(50, 2) + [0, 0],
            np.random.randn(50, 2) + [10, 0],
            np.random.randn(50, 2) + [0, 10],
            np.random.randn(50, 2) + [10, 10]
        ])
        
        kmeans = KMeansFromScratch(n_clusters=4, n_init=1, random_state=42)
        kmeans.fit(X)
        
        # Should converge in < 20 iterations for well-separated data
        assert kmeans.n_iter_ < 20
    
    def test_max_iter_respected(self):
        """Should stop at max_iter even if not converged."""
        X = np.random.randn(100, 50)  # High-dimensional noisy data
        max_iter = 5
        
        kmeans = KMeansFromScratch(n_clusters=10, max_iter=max_iter, n_init=1, random_state=42)
        kmeans.fit(X)
        
        # Number of iterations should be <= max_iter
        assert kmeans.n_iter_ <= max_iter
