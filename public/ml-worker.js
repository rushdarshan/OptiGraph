/**
 * OptiGraph ML Web Worker - Prevents UI freezing during computation
 * Runs PyOdide + ML pipeline in background thread
 */

// Import Pyodide from CDN
importScripts('https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js');

let pyodide = null;
let OptiGraphEngine = null;

// Initialize Pyodide and ML pipeline
async function initializeML() {
    try {
        console.log("Worker: Initializing Pyodide from CDN...");
        pyodide = await loadPyodide({
            indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/',
            stdout: (text) => console.log("Pyodide stdout:", text),
            stderr: (text) => console.warn("Pyodide stderr:", text)
        });
        
        // Load Python packages
        console.log("Worker: Loading NumPy package...");
        await pyodide.loadPackage(['numpy']);
        
        // Load ML engine code from local file or fallback to embedded code
        let pythonCode;
        try {
            pythonCode = await fetch('./python/ml_pipeline.py').then(r => r.text());
            console.log("Worker: Loaded ML pipeline from file");
        } catch (fetchError) {
            console.warn("Worker: Could not fetch ML pipeline file, using embedded code");
            // Fallback to embedded ML pipeline code
            pythonCode = `
import numpy as np
from typing import Tuple, Dict, Any, List

class OptiGraphEngine:
    """In-browser ML pipeline: PCA + K-Means from scratch"""
    
    def __init__(self):
        self.name = "OptiGraph Engine"
        
    def pca_from_scratch(self, X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """PCA implementation using eigendecomposition"""
        # Center the data
        X_centered = X - np.mean(X, axis=0)
        
        # Covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top components
        components = eigenvectors[:, :n_components]
        
        # Transform data
        X_transformed = X_centered @ components
        
        # Calculate explained variance ratio
        explained_variance_ratio = eigenvalues[:n_components] / np.sum(eigenvalues)
        
        return X_transformed, components, explained_variance_ratio
    
    def kmeans_from_scratch(self, X: np.ndarray, k: int, max_iters: int = 100, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """K-means clustering from scratch"""
        n_samples, n_features = X.shape
        
        # Initialize centroids randomly
        centroids = X[np.random.choice(n_samples, k, replace=False)]
        
        inertia_history = []
        
        for iteration in range(max_iters):
            # Assign points to closest centroid
            distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Calculate inertia
            inertia = np.sum(np.min(distances**2, axis=0))
            inertia_history.append(inertia)
            
            # Update centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
            
            # Check convergence
            if np.allclose(centroids, new_centroids, atol=tol):
                break
                
            centroids = new_centroids
        
        return labels, centroids, inertia_history
    
    def process(self, data: List[List[float]], n_components: int = 2, n_clusters: int = 8) -> Dict[str, Any]:
        """Main processing pipeline"""
        X = np.array(data)
        n_samples, n_features_original = X.shape
        
        # Step 1: PCA dimensionality reduction
        X_pca, components, explained_variance_ratio = self.pca_from_scratch(X, n_components)
        
        # Step 2: K-means clustering
        labels, centroids, inertia_history = self.kmeans_from_scratch(X_pca, n_clusters)
        
        # Calculate cumulative variance
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        return {
            'points': X_pca.tolist(),
            'labels': labels.tolist(),
            'centroids': centroids.tolist(),
            'explained_variance_ratio': explained_variance_ratio.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'inertia_history': inertia_history,
            'inertia': inertia_history[-1] if inertia_history else 0.0,
            'n_samples': n_samples,
            'n_features_original': n_features_original,
            'n_components': n_components,
            'n_clusters': n_clusters
        }

# Create global instance
engine = OptiGraphEngine()
`;
        }
        
        await pyodide.runPythonAsync(pythonCode);
        
        // Get the engine instance
        OptiGraphEngine = pyodide.globals.get('engine');
        
        console.log("Worker: ML engine initialized successfully");
        return { success: true };
    } catch (error) {
        console.error("Worker: Initialization failed:", error);
        return { success: false, error: error.message };
    }
}

// Run ML pipeline
async function runPipeline(data, nComponents, nClusters) {
    try {
        if (!OptiGraphEngine) {
            throw new Error("ML engine not initialized");
        }
        
        const startTime = performance.now();
        
        // Process data through PCA + K-Means pipeline
        const result = OptiGraphEngine.process(data, nComponents, nClusters);
        const resultJS = result.toJs({ depth: 10 });
        
        const endTime = performance.now();
        const runtime = endTime - startTime;
        
        console.log(`Worker: Pipeline completed in ${runtime.toFixed(2)}ms`);
        
        return {
            success: true,
            data: {
                ...Object.fromEntries(resultJS),
                runtime_ms: runtime
            }
        };
    } catch (error) {
        console.error("Worker: Pipeline failed:", error);
        return { success: false, error: error.message };
    }
}

// Message handler
self.onmessage = async function(e) {
    const { type, id, payload } = e.data;
    
    try {
        let result;
        
        switch (type) {
            case 'init':
                result = await initializeML();
                break;
                
            case 'run_pipeline':
                const { data, nComponents, nClusters } = payload;
                result = await runPipeline(data, nComponents, nClusters);
                break;
                
            default:
                result = { success: false, error: `Unknown message type: ${type}` };
        }
        
        // Send result back to main thread
        self.postMessage({
            type: `${type}_response`,
            id: id,
            payload: result
        });
        
    } catch (error) {
        self.postMessage({
            type: `${type}_response`,
            id: id,
            payload: { success: false, error: error.message }
        });
    }
};

// Handle worker errors
self.onerror = function(error) {
    console.error("Worker error:", error);
    self.postMessage({
        type: 'error',
        payload: { error: error.message }
    });
};

console.log("OptiGraph ML Worker initialized");