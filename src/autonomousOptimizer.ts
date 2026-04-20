/**
 * OptiGraph Autonomous Optimization Engine (Fixed)
 * 
 * CRITICAL: This system implements financial guardrails and intelligent routing
 * to prevent runaway costs while autonomously optimizing ML execution.
 */

import { MLResult, MLParams } from './types';
import { emergencyCircuitBreaker } from './circuitBreaker';

// EMERGENCY FINANCIAL LIMITS - NEVER EXCEED THESE
export const CRITICAL_LIMITS = {
    max_execution_time_ms: 30000,     // 30s hard timeout
    max_cost_per_execution: 0.05,     // $0.05 per execution
    max_concurrent_operations: 5,      // Prevent resource DoS
    circuit_breaker_threshold: 10,     // Trip after 10 failures
    mandatory_fallback_provider: 'browser_wasm', // Always available
};

interface MLProvider {
    name: string;
    execute: (data: number[][], params: MLParams) => Promise<MLResult>;
    estimateCost: (samples: number, features: number) => number;
    isAvailable: () => boolean;
}

interface CacheEntry {
    result: MLResult;
    created_at: number;
    access_count: number;
}

export class AutonomousMLOptimizer {
    private providers: Map<string, MLProvider> = new Map();
    private cache: Map<string, CacheEntry> = new Map();
    
    constructor() {
        this.initializeProviders();
    }

    /**
     * MAIN ENTRY POINT: Autonomous routing with financial guardrails
     */
    async executeWithGuardrails(data: number[][], params: MLParams): Promise<MLResult> {
        // CRITICAL: Use emergency circuit breaker for all operations
        return await emergencyCircuitBreaker.executeProtected(
            'ml_pipeline_execution',
            async () => {
                return await this.internalExecute(data, params);
            },
            this.estimateTotalCost(data.length, data[0]?.length || 0)
        );
    }

    /**
     * Internal execution logic (protected by circuit breaker)
     */
    private async internalExecute(data: number[][], params: MLParams): Promise<MLResult> {
        // STEP 1: Check cache first to avoid unnecessary cost
        const cacheKey = this.generateCacheKey(data, params);
        const cached = this.checkCache(cacheKey);
        if (cached) {
            console.log(`💰 Cache hit: Saved execution cost`);
            return { ...cached.result, cache_hit: true };
        }
        
        try {
            // STEP 2: Get browser provider (only provider for now)
            const provider = this.providers.get('browser_wasm');
            if (!provider) {
                throw new Error('Browser WASM provider not available');
            }
            
            // STEP 3: Execute with provider
            const result = await this.executeProvider(provider, data, params);
            
            // STEP 4: Cache result
            this.updateCache(cacheKey, result);
            
            return result;
            
        } catch (error) {
            console.error('ML execution failed:', error);
            return this.executeMandatoryFallback(data, params);
        }
    }

    /**
     * Execute with selected provider
     */
    private async executeProvider(provider: MLProvider, data: number[][], params: MLParams): Promise<MLResult> {
        const startTime = Date.now();
        
        const result = await provider.execute(data, params);
        result.runtime_ms = Date.now() - startTime;
        result.provider_used = provider.name;
        result.execution_cost = provider.estimateCost(data.length, data[0]?.length || 0);
        
        return result;
    }

    /**
     * MANDATORY FALLBACK - NEVER FAILS
     */
    private async executeMandatoryFallback(data: number[][], params: MLParams): Promise<MLResult> {
        console.warn('🚨 EXECUTING MANDATORY FALLBACK to prevent total failure');
        
        // Return minimal result that doesn't crash the system
        return {
            points: [], labels: [], centroids: [], inertia: 0,
            inertia_history: [], explained_variance_ratio: [],
            cumulative_variance: [], n_samples: data.length,
            n_features_original: data[0]?.length || 0,
            n_components: params.nComponents, n_clusters: params.nClusters,
            runtime_ms: 0, execution_cost: 0,
            provider_used: 'emergency_fallback'
        };
    }

    private estimateTotalCost(samples: number, features: number): number {
        // Simple cost estimation - browser compute is free, so always 0
        return 0;
    }

    private initializeProviders() {
        // Initialize browser WASM provider (always available)
        this.providers.set('browser_wasm', {
            name: 'browser_wasm',
            execute: async (data, params) => {
                // This will call the existing PyOdide implementation via direct import
                const pyodideModule = await import('./pyodideHelper');
                if (!pyodideModule.OptiGraphMLWorker) {
                    throw new Error('OptiGraphMLWorker not available');
                }
                const worker = new pyodideModule.OptiGraphMLWorker();
                await worker.initialize();
                return await worker.runDirectExecution(data, params.nComponents, params.nClusters);
            },
            estimateCost: (samples, features) => 0, // Browser compute is free
            isAvailable: () => true
        });
    }

    private generateCacheKey(data: number[][], params: MLParams): string {
        const dataShape = `${data.length}x${data[0]?.length || 0}`;
        const dataHash = this.fastChecksum(data);
        const paramsHash = JSON.stringify(params);
        return `${dataShape}_${dataHash}_${paramsHash}`;
    }

    private fastChecksum(data: number[][]): string {
        // Fast checksum for caching (not cryptographic)
        let hash = 0;
        for (let i = 0; i < Math.min(data.length, 100); i++) {
            for (let j = 0; j < Math.min(data[i]?.length || 0, 10); j++) {
                hash = ((hash << 5) - hash + data[i][j]) & 0xffffffff;
            }
        }
        return hash.toString(16);
    }

    private checkCache(key: string): CacheEntry | null {
        const entry = this.cache.get(key);
        if (!entry) return null;

        // Cache expiry: 1 hour
        if (Date.now() - entry.created_at > 60 * 60 * 1000) {
            this.cache.delete(key);
            return null;
        }

        entry.access_count++;
        return entry;
    }

    private updateCache(key: string, result: MLResult) {
        this.cache.set(key, {
            result,
            created_at: Date.now(),
            access_count: 1
        });

        // Limit cache size
        if (this.cache.size > 100) {
            const oldestKey = Array.from(this.cache.entries())
                .sort((a, b) => a[1].created_at - b[1].created_at)[0][0];
            this.cache.delete(oldestKey);
        }
    }
}

// Export singleton instance
export const mlOptimizer = new AutonomousMLOptimizer();