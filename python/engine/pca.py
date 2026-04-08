"""
PCA from Scratch - Mathematical Implementation
No sklearn allowed - pure NumPy eigen-decomposition
"""
import numpy as np


class PCAFromScratch:
    """Principal Component Analysis implemented from first principles."""
    
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None  # shape: (n_components, n_features)
        self.explained_variance_ = None  # eigenvalues for selected components
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray):
        """
        Fit PCA on data X.
        
        Steps:
        1. Center the data (zero mean)
        2. Compute covariance matrix
        3. Eigen decomposition 
        4. Sort by eigenvalue (descending)
        5. Select top-k components
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
        
        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        
        if self.n_components > n_features:
            raise ValueError("n_components cannot exceed n_features")

        # 1) Center the data - CRITICAL for PCA math
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # 2) Covariance matrix: Cov = (X^T @ X) / (n - 1)
        cov = (Xc.T @ Xc) / (n_samples - 1)

        # 3) Eigen decomposition (symmetric matrix => eigh is more efficient)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # 4) Sort descending by eigenvalue (largest variance first)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # 5) Keep top-k components
        k = self.n_components
        self.components_ = eigvecs[:, :k].T  # shape: (k, n_features)
        self.explained_variance_ = eigvals[:k]
        
        # Calculate variance ratio for interpretability
        total_var = eigvals.sum() + 1e-12  # avoid division by zero
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        
        return self

    def transform(self, X: np.ndarray):
        """
        Project X onto principal components.
        
        Args:
            X: Data to transform, shape (n_samples, n_features)
        
        Returns:
            X_transformed: Projected data, shape (n_samples, n_components)
        """
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("Call fit() before transform()")
        
        X = np.asarray(X, dtype=np.float64)
        Xc = X - self.mean_  # Center using training mean
        return Xc @ self.components_.T

    def fit_transform(self, X: np.ndarray):
        """Fit PCA and transform X in one step."""
        return self.fit(X).transform(X)
