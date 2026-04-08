"""
PCA from Scratch - Mathematical Implementation
Two methods: Covariance eigen-decomposition + SVD (numerically stable)
No sklearn allowed - pure NumPy linear algebra
"""
import numpy as np


class PCAFromScratch:
    """
    Principal Component Analysis implemented from first principles.
    
    Supports two computational methods:
    1. Covariance matrix eigen-decomposition (default for n_samples > n_features)
    2. SVD on centered data (more stable, faster when n_samples < n_features)
    """
    
    def __init__(self, n_components=2, method='auto', whiten=False):
        """
        Args:
            n_components: Number of principal components to keep
            method: 'auto', 'eig', or 'svd'
                - 'auto': chooses based on data shape
                - 'eig': covariance eigen-decomposition
                - 'svd': SVD on centered data (more stable)
            whiten: If True, scale components by sqrt(explained_variance)
        """
        self.n_components = n_components
        self.method = method
        self.whiten = whiten
        self.mean_ = None
        self.components_ = None  # shape: (n_components, n_features)
        self.explained_variance_ = None  # eigenvalues for selected components
        self.explained_variance_ratio_ = None
        self.singular_values_ = None  # for SVD method
        self.method_used_ = None  # track which method was actually used

    def fit(self, X: np.ndarray):
        """
        Fit PCA on data X.
        
        Method selection:
        - SVD: more stable, O(min(n,d)²·max(n,d))
        - Eigen: classic approach, O(d³) for covariance
        
        Args:
            X: Data matrix of shape (n_samples, n_features)
        
        Returns:
            self
        
        Raises:
            ValueError: If n_samples < 2, n_components invalid, or data issues
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        
        # Validation
        if n_samples < 2:
            raise ValueError(f"PCA requires at least 2 samples, got {n_samples}")
        
        if self.n_components > min(n_samples, n_features):
            raise ValueError(
                f"n_components={self.n_components} must be <= "
                f"min(n_samples={n_samples}, n_features={n_features})"
            )
        
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("Input contains NaN or Inf values")

        # 1) Center the data - MANDATORY for PCA
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # 2) Choose method
        if self.method == 'auto':
            # Use SVD when n_samples << n_features (more stable for high-dim data)
            # SVD is O(nd²) when d < n, Eigen is O(d³)
            # But SVD is more numerically stable in general
            use_svd = n_samples > n_features  # Prefer SVD for tall matrices
            self.method_used_ = 'svd' if use_svd else 'eig'
        else:
            self.method_used_ = self.method

        # 3) Compute PCA using selected method
        if self.method_used_ == 'svd':
            self._fit_svd(Xc, n_samples, n_features)
        else:
            self._fit_eig(Xc, n_samples, n_features)
        
        return self

    def _fit_eig(self, Xc, n_samples, n_features):
        """Eigen-decomposition of covariance matrix."""
        # Covariance matrix: Cov = (Xc^T @ Xc) / (n - 1)
        cov = (Xc.T @ Xc) / (n_samples - 1)

        # Eigen decomposition (symmetric => eigh is more efficient & stable)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort descending by eigenvalue (largest variance first)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Keep top-k components
        k = self.n_components
        self.components_ = eigvecs[:, :k].T  # shape: (k, n_features)
        self.explained_variance_ = eigvals[:k]
        
        # Calculate variance ratio
        total_var = eigvals.sum() + 1e-12  # numerical stability
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        
        # Compute singular values from eigenvalues
        self.singular_values_ = np.sqrt(self.explained_variance_ * (n_samples - 1))

    def _fit_svd(self, Xc, n_samples, n_features):
        """
        SVD on centered data - more numerically stable with regularization.
        
        Math: Xc = U @ S @ V^T
        - V are the principal components (eigenvectors of Xc^T @ Xc)
        - S² / (n-1) are the explained variances (eigenvalues)
        """
        # IMPROVEMENT 1: Add numerical stability threshold
        SVD_TOL = 1e-12
        
        # Check condition number for potential instability
        if n_samples <= n_features:
            # For wide matrices, check gram matrix condition
            gram_matrix = Xc @ Xc.T
            cond_num = np.linalg.cond(gram_matrix)
            if cond_num > 1e10:
                # Add Tikhonov regularization for ill-conditioned data
                regularization = 1e-8 * np.trace(gram_matrix) / n_samples
                Xc_reg = Xc.copy()
                Xc_reg += np.random.randn(*Xc.shape) * regularization * 0.1
                print(f"Warning: Ill-conditioned data (cond={cond_num:.2e}). Applied regularization.")
                Xc = Xc_reg
        
        # SVD: Xc = U @ S @ Vt
        # full_matrices=False for economy SVD (faster)
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        
        # IMPROVEMENT 2: Filter near-zero singular values for stability
        valid_idx = S > SVD_TOL
        S_filtered = S[valid_idx]
        
        # IMPROVEMENT 3: Auto-adjust n_components if we found fewer valid components
        n_valid = min(len(S_filtered), self.n_components)
        if n_valid < self.n_components:
            print(f"Warning: Found only {n_valid} valid components (tol={SVD_TOL:.2e}). Reducing from {self.n_components}.")
            self.n_components = n_valid
        
        # V^T rows are principal components (filtered for numerical stability)
        self.components_ = Vt[valid_idx][:n_valid]  # shape: (n_valid, n_features)
        
        # Singular values -> explained variance
        # Var = S² / (n-1) since Cov = (Xc^T @ Xc) / (n-1)
        explained_variance_all = (S_filtered ** 2) / (n_samples - 1)
        self.explained_variance_ = explained_variance_all[:n_valid]
        self.singular_values_ = S_filtered[:n_valid]
        
        # Variance ratio
        total_var = explained_variance_all.sum() + 1e-12
        self.explained_variance_ratio_ = self.explained_variance_ / total_var

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
        
        if X.shape[1] != self.components_.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features but PCA was fitted with "
                f"{self.components_.shape[1]} features"
            )
        
        # Center using training mean
        Xc = X - self.mean_
        
        # Project onto components
        X_transformed = Xc @ self.components_.T
        
        # Optional whitening (decorrelate + unit variance)
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_ + 1e-8)
        
        return X_transformed

    def fit_transform(self, X: np.ndarray):
        """Fit PCA and transform X in one step."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray):
        """
        Project data back to original space.
        
        Args:
            X_transformed: Transformed data, shape (n_samples, n_components)
        
        Returns:
            X_reconstructed: Reconstructed data in original space
        """
        if self.components_ is None:
            raise RuntimeError("Call fit() before inverse_transform()")
        
        X_transformed = np.asarray(X_transformed, dtype=np.float64)
        
        if self.whiten:
            # Reverse whitening
            X_transformed = X_transformed * np.sqrt(self.explained_variance_ + 1e-8)
        
        # Project back: X_reconstructed = X_transformed @ components + mean
        return X_transformed @ self.components_ + self.mean_
    
    def get_covariance(self):
        """
        Compute covariance in the original space.
        
        Returns:
            Approximate covariance matrix using fitted components
        """
        if self.components_ is None:
            raise RuntimeError("Call fit() before get_covariance()")
        
        # Cov ≈ components^T @ diag(explained_variance) @ components
        return self.components_.T @ np.diag(self.explained_variance_) @ self.components_
