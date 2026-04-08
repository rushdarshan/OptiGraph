"""
Unit tests for PCA implementation.
Validates mathematical correctness against known properties.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.pca import PCAFromScratch


class TestPCACorrectness:
    """Test mathematical properties of PCA."""
    
    def test_explained_variance_sum_leq_total(self):
        """Explained variance ratio should sum to <= 1.0."""
        X = np.random.randn(100, 20)
        pca = PCAFromScratch(n_components=5)
        pca.fit(X)
        
        assert np.sum(pca.explained_variance_ratio_) <= 1.0 + 1e-6
        assert np.all(pca.explained_variance_ratio_ >= 0)
    
    def test_components_orthonormal(self):
        """Principal components should be orthonormal vectors."""
        X = np.random.randn(100, 20)
        pca = PCAFromScratch(n_components=5)
        pca.fit(X)
        
        # Check orthonormality: C @ C^T = I
        gram = pca.components_ @ pca.components_.T
        expected = np.eye(pca.n_components)
        
        np.testing.assert_allclose(gram, expected, atol=1e-6)
    
    def test_transform_shape(self):
        """Transform should produce correct output shape."""
        X = np.random.randn(100, 20)
        pca = PCAFromScratch(n_components=3)
        X_proj = pca.fit_transform(X)
        
        assert X_proj.shape == (100, 3)
    
    def test_eig_vs_svd_equivalence(self):
        """Eigen and SVD methods should give same results."""
        X = np.random.randn(100, 20)
        
        pca_eig = PCAFromScratch(n_components=5, method='eig')
        pca_svd = PCAFromScratch(n_components=5, method='svd')
        
        X_eig = pca_eig.fit_transform(X)
        X_svd = pca_svd.fit_transform(X)
        
        # Results may have sign flips (eigenvectors are unique up to sign)
        # So check absolute values
        np.testing.assert_allclose(
            np.abs(X_eig), np.abs(X_svd), atol=1e-5
        )
        
        # Explained variance should match exactly
        np.testing.assert_allclose(
            pca_eig.explained_variance_,
            pca_svd.explained_variance_,
            rtol=1e-6
        )
    
    def test_inverse_transform_reconstruction(self):
        """Inverse transform should approximately reconstruct data."""
        X = np.random.randn(50, 10)
        pca = PCAFromScratch(n_components=10)  # Full rank
        
        X_proj = pca.fit_transform(X)
        X_recon = pca.inverse_transform(X_proj)
        
        # With all components, reconstruction should be near-perfect
        np.testing.assert_allclose(X, X_recon, atol=1e-10)
    
    def test_whitening(self):
        """Whitened components should have unit variance."""
        X = np.random.randn(200, 10)
        pca = PCAFromScratch(n_components=5, whiten=True)
        X_white = pca.fit_transform(X)
        
        # Whitened data should have approximately unit variance
        variances = np.var(X_white, axis=0)
        np.testing.assert_allclose(variances, 1.0, rtol=0.1)


class TestPCAEdgeCases:
    """Test edge cases and error handling."""
    
    def test_n_samples_validation(self):
        """Should raise error if n_samples < 2."""
        X = np.random.randn(1, 10)
        pca = PCAFromScratch(n_components=2)
        
        with pytest.raises(ValueError, match="at least 2 samples"):
            pca.fit(X)
    
    def test_n_components_validation(self):
        """Should raise error if n_components > min(n, d)."""
        X = np.random.randn(50, 10)
        pca = PCAFromScratch(n_components=15)
        
        with pytest.raises(ValueError, match="must be <="):
            pca.fit(X)
    
    def test_nan_input(self):
        """Should raise error on NaN input."""
        X = np.random.randn(50, 10)
        X[5, 3] = np.nan
        pca = PCAFromScratch(n_components=3)
        
        with pytest.raises(ValueError, match="NaN"):
            pca.fit(X)
    
    def test_transform_before_fit(self):
        """Should raise error if transform called before fit."""
        X = np.random.randn(50, 10)
        pca = PCAFromScratch(n_components=3)
        
        with pytest.raises(RuntimeError, match="Call fit"):
            pca.transform(X)
    
    def test_feature_mismatch(self):
        """Should raise error if transform gets wrong number of features."""
        X_train = np.random.randn(50, 10)
        X_test = np.random.randn(30, 8)  # Wrong feature count
        
        pca = PCAFromScratch(n_components=3)
        pca.fit(X_train)
        
        with pytest.raises(ValueError, match="features"):
            pca.transform(X_test)


class TestPCANumericalStability:
    """Test numerical stability in challenging scenarios."""
    
    def test_tall_skinny_matrix(self):
        """SVD should be more efficient for n >> d."""
        X = np.random.randn(5000, 20)
        
        pca = PCAFromScratch(n_components=10, method='auto')
        pca.fit(X)
        
        # Should automatically choose SVD
        assert pca.method_used_ == 'svd'
    
    def test_short_fat_matrix(self):
        """Eigen should work for d >> n."""
        X = np.random.randn(50, 500)
        
        pca = PCAFromScratch(n_components=20, method='auto')
        pca.fit(X)
        
        # Should automatically choose eigen for small sample count
        assert pca.method_used_ == 'eig'
    
    def test_zero_variance_features(self):
        """Should handle features with zero variance."""
        X = np.random.randn(100, 10)
        X[:, 5] = 0  # Zero variance feature
        
        pca = PCAFromScratch(n_components=5)
        pca.fit(X)
        
        # Should complete without error
        assert pca.components_.shape == (5, 10)
