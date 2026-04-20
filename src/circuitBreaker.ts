/**
 * Emergency Circuit Breaker System
 * 
 * CRITICAL: This provides mandatory financial protection for the ML system.
 * All ML operations MUST go through this system to prevent runaway costs.
 */

export interface CircuitBreakerConfig {
    maxExecutionTimeMs: number;
    maxCostPerExecution: number;
    maxConcurrentOperations: number;
    failureThreshold: number;
    resetTimeoutMs: number;
}

export interface ExecutionMetrics {
    totalExecutions: number;
    totalFailures: number;
    totalCost: number;
    averageLatencyMs: number;
    lastExecutionTime: number;
}

export class EmergencyCircuitBreaker {
    private config: CircuitBreakerConfig;
    private metrics: Map<string, ExecutionMetrics> = new Map();
    private circuitState: Map<string, 'CLOSED' | 'OPEN' | 'HALF_OPEN'> = new Map();
    private lastFailureTime: Map<string, number> = new Map();
    private concurrentOperations = 0;

    constructor(config?: Partial<CircuitBreakerConfig>) {
        this.config = {
            maxExecutionTimeMs: 30000,      // 30s hard timeout
            maxCostPerExecution: 0.05,      // $0.05 per execution
            maxConcurrentOperations: 5,     // Prevent DoS
            failureThreshold: 10,           // Trip after 10 failures
            resetTimeoutMs: 300000,         // 5min reset timeout
            ...config
        };
    }

    /**
     * MANDATORY: All operations must go through this method
     */
    async executeProtected<T>(
        operationId: string,
        operation: () => Promise<T>,
        estimatedCost: number = 0
    ): Promise<T> {
        // GUARDRAIL 1: Concurrent operation limit
        if (this.concurrentOperations >= this.config.maxConcurrentOperations) {
            throw new Error(`🚨 CIRCUIT BREAKER: Concurrent limit exceeded (${this.concurrentOperations}/${this.config.maxConcurrentOperations})`);
        }

        // GUARDRAIL 2: Cost limit
        if (estimatedCost > this.config.maxCostPerExecution) {
            throw new Error(`🚨 CIRCUIT BREAKER: Cost limit exceeded ($${estimatedCost} > $${this.config.maxCostPerExecution})`);
        }

        // GUARDRAIL 3: Check circuit state
        const circuitState = this.getCircuitState(operationId);
        if (circuitState === 'OPEN') {
            throw new Error(`🚨 CIRCUIT BREAKER: Circuit OPEN for ${operationId} - too many failures`);
        }

        this.concurrentOperations++;
        const startTime = Date.now();

        try {
            // GUARDRAIL 4: Execution timeout
            const result = await Promise.race([
                operation(),
                new Promise<never>((_, reject) => 
                    setTimeout(() => reject(new Error('Execution timeout')), this.config.maxExecutionTimeMs)
                )
            ]);

            this.recordSuccess(operationId, Date.now() - startTime, estimatedCost);
            return result;

        } catch (error) {
            this.recordFailure(operationId, Date.now() - startTime, error as Error);
            throw error;
        } finally {
            this.concurrentOperations--;
        }
    }

    private getCircuitState(operationId: string): 'CLOSED' | 'OPEN' | 'HALF_OPEN' {
        const currentState = this.circuitState.get(operationId) || 'CLOSED';
        
        if (currentState === 'OPEN') {
            const lastFailure = this.lastFailureTime.get(operationId) || 0;
            const timeSinceFailure = Date.now() - lastFailure;
            
            if (timeSinceFailure > this.config.resetTimeoutMs) {
                console.log(`🟡 Circuit breaker for ${operationId}: OPEN -> HALF_OPEN (timeout reset)`);
                this.circuitState.set(operationId, 'HALF_OPEN');
                return 'HALF_OPEN';
            }
        }
        
        return currentState;
    }

    private recordSuccess(operationId: string, latencyMs: number, cost: number) {
        const metrics = this.getOrCreateMetrics(operationId);
        
        metrics.totalExecutions++;
        metrics.totalCost += cost;
        metrics.averageLatencyMs = (metrics.averageLatencyMs + latencyMs) / 2;
        metrics.lastExecutionTime = Date.now();

        // Reset circuit if it was half-open
        const currentState = this.circuitState.get(operationId);
        if (currentState === 'HALF_OPEN') {
            console.log(`🟢 Circuit breaker for ${operationId}: HALF_OPEN -> CLOSED (success)`);
            this.circuitState.set(operationId, 'CLOSED');
            
            // Reset failure count on successful recovery
            metrics.totalFailures = Math.max(0, metrics.totalFailures - 1);
        }
    }

    private recordFailure(operationId: string, latencyMs: number, error: Error) {
        const metrics = this.getOrCreateMetrics(operationId);
        
        metrics.totalExecutions++;
        metrics.totalFailures++;
        metrics.lastExecutionTime = Date.now();
        this.lastFailureTime.set(operationId, Date.now());

        console.warn(`💥 Operation failed: ${operationId} - ${error.message}`);

        // Check if circuit should trip
        const failureRate = metrics.totalFailures / metrics.totalExecutions;
        const shouldTrip = (
            metrics.totalFailures >= this.config.failureThreshold ||
            (failureRate > 0.5 && metrics.totalExecutions >= 5)
        );

        if (shouldTrip) {
            console.error(`🔴 CIRCUIT BREAKER TRIPPED: ${operationId} (${metrics.totalFailures} failures)`);
            this.circuitState.set(operationId, 'OPEN');
        }
    }

    private getOrCreateMetrics(operationId: string): ExecutionMetrics {
        if (!this.metrics.has(operationId)) {
            this.metrics.set(operationId, {
                totalExecutions: 0,
                totalFailures: 0,
                totalCost: 0,
                averageLatencyMs: 0,
                lastExecutionTime: 0
            });
        }
        return this.metrics.get(operationId)!;
    }

    /**
     * Get system health report
     */
    getHealthReport(): {
        totalOperations: number;
        totalCost: number;
        circuitStates: Record<string, string>;
        activeOperations: number;
        systemHealth: 'HEALTHY' | 'DEGRADED' | 'CRITICAL';
    } {
        const totalOperations = Array.from(this.metrics.values()).reduce((sum, m) => sum + m.totalExecutions, 0);
        const totalCost = Array.from(this.metrics.values()).reduce((sum, m) => sum + m.totalCost, 0);
        
        const circuitStates: Record<string, string> = {};
        this.circuitState.forEach((state, id) => {
            circuitStates[id] = state;
        });

        const openCircuits = Array.from(this.circuitState.values()).filter(state => state === 'OPEN').length;
        
        let systemHealth: 'HEALTHY' | 'DEGRADED' | 'CRITICAL' = 'HEALTHY';
        if (openCircuits > 0) {
            systemHealth = 'DEGRADED';
        }
        if (this.concurrentOperations >= this.config.maxConcurrentOperations * 0.8) {
            systemHealth = 'CRITICAL';
        }

        return {
            totalOperations,
            totalCost,
            circuitStates,
            activeOperations: this.concurrentOperations,
            systemHealth
        };
    }

    /**
     * Force reset a circuit (admin override)
     */
    forceResetCircuit(operationId: string) {
        console.warn(`⚠️ ADMIN OVERRIDE: Force resetting circuit for ${operationId}`);
        this.circuitState.set(operationId, 'CLOSED');
        
        // Reset failure metrics
        const metrics = this.metrics.get(operationId);
        if (metrics) {
            metrics.totalFailures = 0;
        }
    }
}

// Global circuit breaker instance
export const emergencyCircuitBreaker = new EmergencyCircuitBreaker();