
import { D4C } from "d4c-queue";

declare var loadPyodide: any;
declare var pyodide: any;
declare var globalThis: any;
declare var self: any;

function baseURL() {
    const regex = /chrome-extension:\/\/.*(?=\/index.html)/;
    const matchExtensionURL = window.location.href.match(regex)
    if (matchExtensionURL) {
        return matchExtensionURL[0] + "/"
    }
    return window.location.origin + window.location.pathname
}

const d4c = new D4C();

const initPyodideAndLoadPackages = d4c.wrap(async () => {
    console.log("OptiGraph: Initializing Pyodide...");
    
    globalThis.pyodide = await loadPyodide({ indexURL: baseURL() + "pyodide/" });
    
    const pythonCode = await (await fetch('python/pyodide_init.py')).text();
    await pyodide.loadPackagesFromImports(pythonCode);
    await pyodide.runPythonAsync(pythonCode);
    
    console.log("OptiGraph: Pyodide initialized successfully");
});

const loadMLPipeline = d4c.wrap(async () => {
    console.log("OptiGraph: Loading ML pipeline...");
    
    const pythonCode = await (await fetch('python/ml_pipeline.py')).text();
    await pyodide.loadPackagesFromImports(pythonCode);
    await pyodide.runPythonAsync(pythonCode);
    
    const OptiGraphEngine = pyodide.globals.get('OptiGraphEngine');
    console.log("OptiGraph: ML pipeline loaded successfully");
    
    return OptiGraphEngine;
});

const runMLPipeline = d4c.wrap(async (
    data: number[][],
    nComponents: number = 2,
    nClusters: number = 8
): Promise<any> => {
    console.log(`OptiGraph: Running pipeline with ${data.length} samples, ${nComponents} components, ${nClusters} clusters`);
    
    const startTime = performance.now();
    
    const OptiGraphEngine = pyodide.globals.get('OptiGraphEngine');
    const result = OptiGraphEngine.process(data, nComponents, nClusters);
    const resultJS = result.toJs({ depth: 10 });
    
    const endTime = performance.now();
    const runtime = endTime - startTime;
    
    console.log(`OptiGraph: Pipeline completed in ${runtime.toFixed(2)}ms`);
    
    return {
        ...Object.fromEntries(resultJS),
        runtime_ms: runtime
    };
});

export {
    initPyodideAndLoadPackages,
    loadMLPipeline,
    runMLPipeline
}
