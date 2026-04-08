
/**
 * OptiGraph PyOdide Helper - Web Worker Integration
 * Prevents UI freezing during ML computation
 */

interface MLResult {
    points: number[][];
    labels: number[];
    centroids: number[][];
    inertia: number;
    inertia_history: number[];
    explained_variance_ratio: number[];
    cumulative_variance: number[];
    n_samples: number;
    n_features_original: number;
    n_components: number;
    n_clusters: number;
    runtime_ms: number;
}

class OptiGraphMLWorker {
    private worker: Worker | null = null;
    private messageId = 0;
    private pendingMessages = new Map<number, { resolve: Function; reject: Function }>();
    private initialized = false;

    constructor() {
        this.initWorker();
    }

    private initWorker() {
        if (typeof Worker === 'undefined') {
            console.warn("Web Workers not supported, falling back to main thread");
            return;
        }

        try {
            this.worker = new Worker('./ml-worker.js');
            
            this.worker.onmessage = (e) => {
                const { type, id, payload } = e.data;
                
                if (type === 'error') {
                    console.error("Worker error:", payload.error);
                    return;
                }
                
                const pendingMessage = this.pendingMessages.get(id);
                if (pendingMessage) {
                    this.pendingMessages.delete(id);
                    
                    if (payload.success) {
                        pendingMessage.resolve(payload);
                    } else {
                        pendingMessage.reject(new Error(payload.error));
                    }
                }
            };
            
            this.worker.onerror = (error) => {
                console.error("Worker error:", error);
                this.rejectAllPending(new Error("Worker crashed"));
            };
            
            console.log("OptiGraph: ML Worker created successfully");
        } catch (error) {
            console.error("Failed to create ML Worker:", error);
            this.worker = null;
        }
    }

    private sendMessage(type: string, payload: any): Promise<any> {
        return new Promise((resolve, reject) => {
            if (!this.worker) {
                reject(new Error("Worker not available"));
                return;
            }

            const id = ++this.messageId;
            this.pendingMessages.set(id, { resolve, reject });

            this.worker.postMessage({ type, id, payload });

            // Timeout after 60 seconds
            setTimeout(() => {
                if (this.pendingMessages.has(id)) {
                    this.pendingMessages.delete(id);
                    reject(new Error(`Worker timeout for message type: ${type}`));
                }
            }, 60000);
        });
    }

    private rejectAllPending(error: Error) {
        for (const [id, { reject }] of this.pendingMessages) {
            reject(error);
        }
        this.pendingMessages.clear();
    }

    async initialize(): Promise<void> {
        if (this.initialized) {
            return;
        }

        if (!this.worker) {
            throw new Error("Web Worker not available. Cannot initialize ML engine.");
        }

        console.log("OptiGraph: Initializing ML engine in worker...");
        const result = await this.sendMessage('init', {});
        
        if (!result.success) {
            throw new Error(`ML initialization failed: ${result.error}`);
        }

        this.initialized = true;
        console.log("OptiGraph: ML engine initialized successfully");
    }

    async runPipeline(
        data: number[][],
        nComponents: number = 2,
        nClusters: number = 8
    ): Promise<MLResult> {
        if (!this.initialized) {
            throw new Error("ML engine not initialized. Call initialize() first.");
        }

        if (!this.worker) {
            throw new Error("Worker not available");
        }

        console.log(`OptiGraph: Running pipeline with ${data.length} samples...`);
        
        const result = await this.sendMessage('run_pipeline', {
            data,
            nComponents,
            nClusters
        });

        if (!result.success) {
            throw new Error(`Pipeline failed: ${result.error}`);
        }

        return result.data;
    }

    dispose() {
        if (this.worker) {
            this.rejectAllPending(new Error("Worker disposed"));
            this.worker.terminate();
            this.worker = null;
        }
        this.initialized = false;
    }
}

// Singleton instance
let mlWorker: OptiGraphMLWorker | null = null;

export const initPyodideAndLoadPackages = async (): Promise<void> => {
    if (!mlWorker) {
        mlWorker = new OptiGraphMLWorker();
    }
    await mlWorker.initialize();
};

export const loadMLPipeline = async (): Promise<void> => {
    // Pipeline is loaded as part of initialization
    return Promise.resolve();
};

export const runMLPipeline = async (
    data: number[][],
    nComponents: number = 2,
    nClusters: number = 8
): Promise<MLResult> => {
    if (!mlWorker) {
        throw new Error("ML Worker not initialized. Call initPyodideAndLoadPackages() first.");
    }
    
    return mlWorker.runPipeline(data, nComponents, nClusters);
};

export const disposePyodide = (): void => {
    if (mlWorker) {
        mlWorker.dispose();
        mlWorker = null;
    }
};
