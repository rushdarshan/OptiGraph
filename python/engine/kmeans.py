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
    """K-Means clustering implemented from first principles."""
    
    def __init__(self, n_clusters=8, max_iter=100, tol=1e-4, random_state=42):
        """
        Args:
            n_clusters: Number of clusters
            max_iter: Maximum iterations
            tol: Convergence tolerance (centroid shift)
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.inertia_history_ = []  # Track convergence

    def _init_kmeanspp(self, X):
        """
        K-Means++ initialization for better convergence.
        
        Algorithm:
        1. Choose first centroid uniformly at random
        2. For each remaining centroid:
           - Compute D(x) = min distance to existing centroids
           - Choose next centroid with probability proportional to D(x)^2
        
        This spreads initial centroids far apart.
        """
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        centroids = []

        # First centroid: random sample
        first_idx = rng.integers(0, n_samples)
        centroids.append(X[first_idx])

        # Remaining centroids: weighted by squared distance
        for _ in range(1, self.n_clusters):
            d2 = _pairwise_sq_dists(X, np.array(centroids))
            min_d2 = np.min(d2, axis=1)  # Distance to closest centroid
            probs = min_d2 / (min_d2.sum() + 1e-12)
            next_idx = rng.choice(n_samples, p=probs)
            centroids.append(X[next_idx])

        return np.array(centroids, dtype=np.float64)

    def fit(self, X: np.ndarray):
        """
        Fit K-Means on data X.
        
        Lloyd's algorithm:
        1. Initialize centroids (K-Means++)
        2. Repeat until convergence:
           a. Assign each point to nearest centroid
           b. Update centroids as mean of assigned points
           c. Check convergence (centroid shift < tol)
        
        Args:
            X: Data matrix, shape (n_samples, n_features)
        
        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]

        # Initialize centroids
        centroids = self._init_kmeanspp(X)

        for iteration in range(self.max_iter):
            # E-step: Assign points to nearest centroid
            d2 = _pairwise_sq_dists(X, centroids)
            labels = np.argmin(d2, axis=1)

            # M-step: Update centroids
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                pts = X[labels == k]
                if len(pts) == 0:
                    # Empty cluster: re-seed from a random point
                    new_centroids[k] = X[np.random.randint(0, n_samples)]
                else:
                    new_centroids[k] = pts.mean(axis=0)

            # Check convergence
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            # Calculate inertia (sum of squared distances to assigned centroid)
            d2 = _pairwise_sq_dists(X, centroids)
            inertia = np.sum(d2[np.arange(n_samples), labels])
            self.inertia_history_.append(float(inertia))

            if shift < self.tol:
                break

        self.centroids_ = centroids
        self.labels_ = labels
        self.inertia_ = self.inertia_history_[-1]
        return self

    def fit_predict(self, X):
        """Fit K-Means and return cluster labels."""
        return self.fit(X).labels_
