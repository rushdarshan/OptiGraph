"""
K-Means from Scratch - Algorithmic Implementation
K-Means++ initialization + Lloyd's algorithm
No sklearn allowed - pure NumPy optimization
"""
import numpy as np


def _pairwise_sq_dists(A, B):
    """
    Compute pairwise squared Euclidean distances efficiently.
    
    Uses the identity: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    
    Args:
        A: shape (n, d)
        B: shape (k, d)
    
    Returns:
        D: shape (n, k) - squared distances
    """
    A2 = np.sum(A * A, axis=1, keepdims=True)  # (n, 1)
    B2 = np.sum(B * B, axis=1)  # (k,)
    return A2 + B2 - 2.0 * (A @ B.T)


class KMeansFromScratch:
    """
    K-Means clustering implemented from first principles.
    
    Implements Lloyd's algorithm with K-Means++ initialization
    for robust cluster detection.
    """
    
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, n_init=10, random_state=42):
        """
        Args:
            n_clusters: Number of clusters
            max_iter: Maximum iterations per run
            tol: Convergence tolerance (centroid shift)
            n_init: Number of times to run with different initializations
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.inertia_history_ = []  # Track convergence for best run
        self.n_iter_ = None  # Number of iterations for best run
        
        # Initialize RNG for reproducibility
        self._rng = np.random.default_rng(self.random_state)

    def _init_kmeanspp(self, X, rng):
        """
        K-Means++ initialization for better convergence.
        
        Algorithm:
        1. Choose first centroid uniformly at random
        2. For each remaining centroid:
           - Compute D(x)² = squared min distance to existing centroids
           - Choose next centroid with probability ∝ D(x)²
        
        This spreads initial centroids far apart, proven to give
        O(log k) approximation to optimal clustering.
        
        Args:
            X: Data matrix (n_samples, n_features)
            rng: numpy random generator for reproducibility
        
        Returns:
            centroids: (n_clusters, n_features)
        """
        n_samples = X.shape[0]
        centroids = []

        # First centroid: uniformly random sample
        first_idx = rng.integers(0, n_samples)
        centroids.append(X[first_idx].copy())

        # Remaining centroids: weighted by squared distance
        for _ in range(1, self.n_clusters):
            d2 = _pairwise_sq_dists(X, np.array(centroids))
            min_d2 = np.min(d2, axis=1)  # Distance to closest centroid
            
            # Handle numerical precision: clip negative values
            min_d2 = np.maximum(min_d2, 0.0)
            
            # Probability proportional to squared distance
            total = min_d2.sum()
            if total < 1e-12:
                # All points are at existing centroids; pick randomly
                probs = np.ones(n_samples) / n_samples
            else:
                probs = min_d2 / total
            
            next_idx = rng.choice(n_samples, p=probs)
            centroids.append(X[next_idx].copy())

        return np.array(centroids, dtype=np.float64)

    def _single_run(self, X, rng):
        """
        Single K-Means run with Lloyd's algorithm.
        
        Returns:
            (centroids, labels, inertia, inertia_history, n_iter)
        """
        n_samples = X.shape[0]
        
        # Initialize centroids
        centroids = self._init_kmeanspp(X, rng)
        inertia_history = []
        
        for iteration in range(self.max_iter):
            # E-step: Assign points to nearest centroid
            d2 = _pairwise_sq_dists(X, centroids)
            labels = np.argmin(d2, axis=1)

            # M-step: Update centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                pts = X[labels == k]
                if len(pts) == 0:
                    # Empty cluster: re-seed from furthest point (not random!)
                    # Use the same RNG for reproducibility
                    furthest_idx = np.argmax(np.min(d2, axis=1))
                    new_centroids[k] = X[furthest_idx].copy()
                else:
                    new_centroids[k] = pts.mean(axis=0)

            # Calculate inertia BEFORE updating (consistent tracking)
            # Inertia = sum of squared distances to assigned centroids
            inertia = np.sum(d2[np.arange(n_samples), labels])
            inertia_history.append(float(inertia))
            
            # Check convergence
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            if shift < self.tol:
                break
        
        # Final assignment with converged centroids
        d2_final = _pairwise_sq_dists(X, centroids)
        labels_final = np.argmin(d2_final, axis=1)
        inertia_final = np.sum(d2_final[np.arange(n_samples), labels_final])
        
        return centroids, labels_final, inertia_final, inertia_history, iteration + 1

    def fit(self, X: np.ndarray):
        """
        Fit K-Means on data X.
        
        Runs K-Means n_init times with different initializations
        and keeps the best result (lowest inertia).
        
        Lloyd's algorithm per run:
        1. Initialize centroids (K-Means++)
        2. Repeat until convergence:
           a. Assign each point to nearest centroid
           b. Update centroids as mean of assigned points
           c. Check convergence (centroid shift < tol)
        
        Args:
            X: Data matrix, shape (n_samples, n_features)
        
        Returns:
            self
        
        Raises:
            ValueError: if n_clusters > n_samples or invalid input
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        
        # Validation
        if n_samples < self.n_clusters:
            raise ValueError(
                f"n_clusters={self.n_clusters} must be <= n_samples={n_samples}"
            )
        
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("Input contains NaN or Inf values")
        
        # Run K-Means multiple times and keep best result
        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_history = None
        best_n_iter = None
        
        for run in range(self.n_init):
            # Use different seed for each run but deterministic
            run_rng = np.random.default_rng(self.random_state + run)
            
            centroids, labels, inertia, history, n_iter = self._single_run(X, run_rng)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_history = history
                best_n_iter = n_iter
        
        self.centroids_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.inertia_history_ = best_history
        self.n_iter_ = best_n_iter
        
        return self

    def predict(self, X: np.ndarray):
        """
        Predict cluster labels for new data.
        
        Args:
            X: Data matrix, shape (n_samples, n_features)
        
        Returns:
            labels: Cluster assignment for each sample
        """
        if self.centroids_ is None:
            raise RuntimeError("Call fit() before predict()")
        
        X = np.asarray(X, dtype=np.float64)
        d2 = _pairwise_sq_dists(X, self.centroids_)
        return np.argmin(d2, axis=1)

    def fit_predict(self, X):
        """Fit K-Means and return cluster labels."""
        return self.fit(X).labels_
    
    def score(self, X: np.ndarray):
        """
        Compute negative inertia (higher is better, for sklearn compatibility).
        
        Args:
            X: Data matrix
        
        Returns:
            Negative inertia
        """
        if self.centroids_ is None:
            raise RuntimeError("Call fit() before score()")
        
        X = np.asarray(X, dtype=np.float64)
        d2 = _pairwise_sq_dists(X, self.centroids_)
        labels = np.argmin(d2, axis=1)
        inertia = np.sum(d2[np.arange(len(X)), labels])
        return -inertia
