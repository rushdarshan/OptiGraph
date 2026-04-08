/**
 * OptiGraph ML Web Worker - Prevents UI freezing during computation
 * Runs PyOdide + ML pipeline in background thread
 */

// Import Pyodide in worker context
importScripts('./pyodide/pyodide.js');

let pyodide = null;
let OptiGraphEngine = null;

// Initialize Pyodide and ML pipeline
async function initializeML() {
    try {
        console.log("Worker: Initializing Pyodide...");
        pyodide = await loadPyodide({ indexURL: './pyodide/' });
        
        // Load Python packages
        await pyodide.loadPackage(['numpy']);
        
        // Load ML engine code
        const pythonCode = await fetch('./python/ml_pipeline.py').then(r => r.text());
        await pyodide.runPythonAsync(pythonCode);
        
        OptiGraphEngine = pyodide.globals.get('OptiGraphEngine');
        
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