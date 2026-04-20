
/**
 * OptiGraph PyOdide Helper - Web Worker Integration with Autonomous Optimization
 * Prevents UI freezing during ML computation
 * Now includes financial guardrails and intelligent routing
 */

import { MLResult, MLParams } from './types';
import { mlOptimizer } from './autonomousOptimizer';

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
            // Use absolute path to worker script
            const workerPath = `${process.env.PUBLIC_URL || ''}/ml-worker.js`;
            this.worker = new Worker(workerPath);
            
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
                this.rejectAllPending(new Error("Worker crashed: " + error.message));
            };
            
            console.log("OptiGraph: ML Worker created successfully at " + workerPath);
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

            // Timeout after 30 seconds to prevent hanging
            const timeoutId = setTimeout(() => {
                if (this.pendingMessages.has(id)) {
                    this.pendingMessages.delete(id);
                    console.error(`Worker timeout: ${type} took longer than 30s`);
                    reject(new Error(`Worker timeout for message type: ${type}`));
                }
            }, 30000);
            
            // Clear timeout on successful completion
            const originalResolve = resolve;
            const wrappedResolve = (value: any) => {
                clearTimeout(timeoutId);
                originalResolve(value);
            };
            this.pendingMessages.set(id, { resolve: wrappedResolve, reject });
        });
    }

    private rejectAllPending(error: Error) {
        this.pendingMessages.forEach(({ reject }) => {
            reject(error);
        });
        this.pendingMessages.clear();
    }

    async initialize(): Promise<void> {
        if (this.initialized) {
            return;
        }

        if (!this.worker) {
            console.warn("Web Worker not available, will run computations on main thread");
            this.initialized = true;
            return;
        }

        console.log("OptiGraph: Initializing ML engine in worker...");
        
        try {
            const result = await this.sendMessage('init', {});
            
            if (!result.success) {
                console.error(`ML initialization failed: ${result.error}, falling back to main thread`);
                this.worker = null;
                this.initialized = true;
                return;
            }

            this.initialized = true;
            console.log("OptiGraph: ML engine initialized successfully");
        } catch (err) {
            console.error("ML initialization timeout/error:", err);
            this.worker = null;
            this.initialized = true;
            // Don't throw - allow graceful fallback
        }
    }

    async runPipeline(
        data: number[][],
        nComponents: number = 2,
        nClusters: number = 8
    ): Promise<MLResult> {
        if (!this.initialized) {
            throw new Error("ML engine not initialized. Call initialize() first.");
        }

        // AUTONOMOUS OPTIMIZATION: Use intelligent routing with financial guardrails
        const params: MLParams = { nComponents, nClusters };
        
        try {
            console.log(`🤖 OptiGraph: Running autonomous optimization for ${data.length} samples...`);
            return await mlOptimizer.executeWithGuardrails(data, params);
        } catch (optimizationError) {
            console.warn('💰 Autonomous optimizer failed, falling back to direct execution:', optimizationError);
            return this.runDirectExecution(data, nComponents, nClusters);
        }
    }

    /**
     * Direct execution for fallback scenarios (exposed for autonomous optimizer)
     */
    async runDirectExecution(data: number[][], nComponents: number, nClusters: number): Promise<MLResult> {
        if (!this.worker) {
            throw new Error("Worker not available");
        }

        console.log(`OptiGraph: Running direct pipeline with ${data.length} samples...`);
        
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

// Export the class for autonomous optimizer
export { OptiGraphMLWorker };

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
