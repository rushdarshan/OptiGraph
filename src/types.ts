/**
 * OptiGraph Types - With Autonomous Optimization Support
 */

export interface MLResult {
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
    execution_cost?: number;
    provider_used?: string;
    cache_hit?: boolean;
}

export interface MLParams {
    nComponents: number;
    nClusters: number;
    maxIterations?: number;
    tolerance?: number;
    randomSeed?: number;
}

export interface ProviderMetrics {
    name: string;
    avg_latency_ms: number;
    cost_per_1k_samples: number;
    success_rate: number;
    last_failure_time?: number;
    circuit_breaker_tripped: boolean;
    total_executions: number;
    total_failures: number;
}

export interface ExecutionLimits {
    max_execution_time_ms: number;
    max_cost_per_execution: number;
    max_concurrent_operations: number;
    circuit_breaker_threshold: number;
    mandatory_fallback_provider: string;
}

export interface CacheEntry {
    result: MLResult;
    created_at: number;
    access_count: number;
    data_hash: string;
    params_hash: string;
}

export interface ShadowTestResult {
    baseline_result: MLResult;
    shadow_result: MLResult;
    quality_score: number;
    cost_efficiency: number;
    winner: 'baseline' | 'shadow';
    should_promote: boolean;
}