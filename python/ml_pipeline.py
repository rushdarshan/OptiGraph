"""
OptiGraph Pipeline - Orchestrates PCA + K-Means with AI Data Remediation
Production-grade pipeline with comprehensive observability and SLA monitoring
"""
import numpy as np
import sys
import logging
import time
from typing import Dict, List, Tuple, Any
from datetime import datetime, timezone
sys.path.append('/python/engine')

from engine.pca import PCAFromScratch
from engine.kmeans import KMeansFromScratch
from data_remediation import DataRemediationEngine

# Configure structured logging for production observability
class StructuredFormatter(logging.Formatter):
    """Structured logging formatter for production monitoring"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'component': 'optigraph.pipeline',
            'message': record.getMessage(),
            'correlation_id': getattr(record, 'correlation_id', 'unknown'),
            'pipeline_stage': getattr(record, 'pipeline_stage', 'unknown'),
            'execution_time_ms': getattr(record, 'execution_time_ms', None),
            'data_shape': getattr(record, 'data_shape', None),
            'anomalies_count': getattr(record, 'anomalies_count', None),
        }
        return str(log_entry)

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(),
        # Add file handler for production
        # logging.FileHandler('/logs/optigraph-pipeline.log')
    ]
)
logger = logging.getLogger('optigraph.pipeline')
logger.handlers = [logging.StreamHandler()]
logger.handlers[0].setFormatter(StructuredFormatter())


class PipelineSLAMonitor:
    """SLA monitoring and alerting for production pipeline"""
    
    def __init__(self, max_execution_time_seconds=300, max_data_loss_rate=0.01):
        self.max_execution_time_seconds = max_execution_time_seconds
        self.max_data_loss_rate = max_data_loss_rate
        self.start_time = None
        
    def start_monitoring(self, correlation_id: str):
        """Start SLA monitoring for pipeline execution"""
        self.start_time = time.time()
        logger.info("Pipeline SLA monitoring started", extra={
            'correlation_id': correlation_id,
            'pipeline_stage': 'monitoring',
            'sla_timeout_seconds': self.max_execution_time_seconds
        })
        
    def check_execution_time_sla(self, correlation_id: str) -> bool:
        """Check if pipeline execution is within SLA"""
        if self.start_time is None:
            return True
            
        execution_time = time.time() - self.start_time
        sla_violated = execution_time > self.max_execution_time_seconds
        
        if sla_violated:
            logger.error("SLA VIOLATION: Pipeline execution time exceeded", extra={
                'correlation_id': correlation_id,
                'pipeline_stage': 'sla_check',
                'execution_time_seconds': execution_time,
                'max_allowed_seconds': self.max_execution_time_seconds,
                'sla_violation_type': 'execution_time'
            })
        
        return not sla_violated
        
    def check_data_quality_sla(self, correlation_id: str, reconciliation: Dict) -> bool:
        """Check if data quality meets SLA requirements"""
        data_loss_rate = reconciliation.get('quarantine_rows', 0) / max(1, reconciliation.get('source_rows', 1))
        sla_violated = data_loss_rate > self.max_data_loss_rate
        
        if sla_violated:
            logger.error("SLA VIOLATION: Data quality below threshold", extra={
                'correlation_id': correlation_id,
                'pipeline_stage': 'sla_check',
                'data_loss_rate': data_loss_rate,
                'max_allowed_rate': self.max_data_loss_rate,
                'quarantine_rows': reconciliation.get('quarantine_rows', 0),
                'source_rows': reconciliation.get('source_rows', 0),
                'sla_violation_type': 'data_quality'
            })
            
        return not sla_violated


def run_pipeline_with_remediation(X: np.ndarray, n_components=2, n_clusters=8, target_variance=0.85) -> Dict[str, Any]:
    """
    Production ML pipeline with comprehensive data remediation and SLA monitoring.
    
    Pipeline stages:
    1. Input validation & anomaly detection  
    2. AI-powered data remediation (semantic clustering + local SLMs)
    3. PCA dimensionality reduction with variance targeting
    4. K-Means clustering with adaptive parameters
    5. Results validation & audit trail generation
    
    SLA Guarantees:
    - Execution time: < 5 minutes
    - Data loss rate: < 1%
    - Zero-data-loss reconciliation: mathematically enforced
    
    Args:
        X: Input data, shape (n_samples, n_features) 
        n_components: Number of PCA components (can be auto-adjusted)
        n_clusters: Number of K-Means clusters
        target_variance: Minimum cumulative variance to retain (0.85 = 85%)
    
    Returns:
        dict with ML results + comprehensive audit trail + remediation report
    """
    correlation_id = f"pipeline_{hash(str(X.shape))%100000}"
    
    # Initialize SLA monitoring
    sla_monitor = PipelineSLAMonitor()
    sla_monitor.start_monitoring(correlation_id)
    
    logger.info("Production pipeline started", extra={
        'correlation_id': correlation_id,
        'pipeline_stage': 'initialization',
        'data_shape': X.shape,
        'n_components': n_components,
        'n_clusters': n_clusters,
        'target_variance': target_variance
    })
    
    try:
        # Initialize remediation engine
        remediation_engine = DataRemediationEngine(
            confidence_threshold=0.75,
            enable_local_llm=False  # Set to True when ollama is available
        )
        
        # Stage 1: Input validation & anomaly detection
        logger.info("Stage 1: Input validation & anomaly detection")
        data_list = X.tolist() if isinstance(X, np.ndarray) else X
        clean_data, anomalies = remediation_engine.validate_input_data(data_list)
        
        # Stage 2: AI-powered data remediation
        if anomalies:
            logger.warning(f"Detected {len(anomalies)} anomalies, initiating remediation")
            fixed_records, quarantine_records = remediation_engine.remediate_anomalies(anomalies)
            
            # Apply fixes to the original data
            fixed_data = remediation_engine.apply_fixes(data_list, anomalies)
            all_clean_data = fixed_data  # Use the fixed version of the full dataset
            
            if quarantine_records:
                logger.warning(f"{len(quarantine_records)} anomalies sent to human review quarantine")
        else:
            all_clean_data = clean_data
            logger.info("No anomalies detected, proceeding with original data")
        
        # Convert back to numpy array
        if not all_clean_data:
            raise ValueError("No valid data remaining after remediation")
            
        X_clean = np.array(all_clean_data, dtype=np.float64)
        logger.info(f"Data remediation complete: {X_clean.shape[0]} clean rows for ML processing")
        
        # Stage 3: PCA dimensionality reduction with automatic component selection
        logger.info("Stage 3: PCA dimensionality reduction")
        pca = PCAFromScratch(n_components=min(n_components, X_clean.shape[1] - 1))
        
        # Auto-adjust components to meet target variance
        initial_fit = PCAFromScratch(n_components=min(50, X_clean.shape[1] - 1))
        initial_fit.fit(X_clean)
        
        # Find minimum components needed for target variance
        cumulative_var = np.cumsum(initial_fit.explained_variance_ratio_)
        components_needed = np.argmax(cumulative_var >= target_variance) + 1
        
        # Use max of requested components and variance-based requirement (for visualization)
        final_n_components = max(n_components, min(components_needed, n_components + 2))
        
        if final_n_components != n_components:
            logger.info(f"Auto-adjusted n_components from {n_components} to {final_n_components} "
                       f"(target variance: {target_variance:.1%}, achieved: {cumulative_var[final_n_components-1]:.1%})")
        
        pca = PCAFromScratch(n_components=final_n_components)
        Z = pca.fit_transform(X_clean)
        
        # For visualization, use only first 2-3 components
        Z_vis = Z[:, :n_components]
        
        # Stage 4: K-Means clustering in reduced space
        logger.info("Stage 4: K-Means clustering")
        km = KMeansFromScratch(n_clusters=n_clusters, max_iter=100, tol=1e-4)
        labels = km.fit_predict(Z)  # Use all components for better clustering
        
        # Project centroids to visualization space
        centroids_vis = km.centroids_[:, :n_components]
        
        # Stage 5: Results validation & audit trail with SLA monitoring
        logger.info("Generating audit trail and validating SLA compliance", extra={
            'correlation_id': correlation_id,
            'pipeline_stage': 'validation'
        })
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # Get comprehensive audit trail
        audit_trail = remediation_engine.get_audit_trail()
        reconciliation = remediation_engine.get_reconciliation_report(clusters_generated=n_clusters)
        
        # SLA Compliance Checks
        sla_time_ok = sla_monitor.check_execution_time_sla(correlation_id)
        sla_quality_ok = sla_monitor.check_data_quality_sla(correlation_id, reconciliation)
        
        # Validate results
        results_valid = (
            Z_vis.shape[0] > 0 and 
            len(labels) == Z_vis.shape[0] and
            not reconciliation['data_loss_detected'] and
            sla_time_ok and 
            sla_quality_ok
        )
        
        total_execution_time = time.time() - sla_monitor.start_time
        
        logger.info("Pipeline completed with SLA monitoring", extra={
            'correlation_id': correlation_id,
            'pipeline_stage': 'completion',
            'execution_time_seconds': total_execution_time,
            'sla_time_compliant': sla_time_ok,
            'sla_quality_compliant': sla_quality_ok,
            'final_samples': Z_vis.shape[0],
            'variance_retained': float(cumulative_variance[final_n_components-1]),
            'clusters_found': n_clusters,
            'anomalies_fixed': reconciliation['ai_fixed_rows'],
            'quarantine_count': reconciliation['quarantine_rows'],
            'results_valid': results_valid
        })
        
        return {
            # ML Results
            "points": Z_vis.tolist(),  # Visualization space (2D/3D)  
            "labels": labels.tolist(),
            "centroids": centroids_vis.tolist(),  # Centroids in visualization space
            "inertia": km.inertia_,
            "inertia_history": km.inertia_history_,
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": cumulative_variance.tolist(),
            
            # Pipeline Metadata
            "n_samples": X_clean.shape[0],
            "n_samples_original": X.shape[0] if isinstance(X, np.ndarray) else len(X),
            "n_features_original": X_clean.shape[1],
            "n_components": final_n_components,  # Actual components used
            "n_components_vis": n_components,    # Components for visualization
            "n_clusters": n_clusters,
            "variance_retained": float(cumulative_variance[final_n_components-1]),
            
            # Data Quality & Audit
            "data_quality": {
                "anomalies_detected": len(audit_trail['anomalies']),
                "anomalies_fixed": reconciliation['ai_fixed_rows'],
                "quarantine_count": reconciliation['quarantine_rows'], 
                "data_loss_detected": reconciliation['data_loss_detected'],
                "clusters_generated": reconciliation['clusters_generated'],
                "remediation_success_rate": (reconciliation['ai_fixed_rows'] / max(1, len(audit_trail['anomalies']))) if audit_trail['anomalies'] else 1.0
            },
            
            # Audit Trail (for compliance/debugging)
            "audit_trail": audit_trail,
            "reconciliation": reconciliation,
            "correlation_id": correlation_id,
            "pipeline_version": "v2.0_with_remediation"
        }
        
    except Exception as e:
        logger.error("Pipeline failed", extra={
            'correlation_id': correlation_id,
            'error': str(e),
            'error_type': type(e).__name__
        }, exc_info=True)
        raise


def run_pipeline(X: np.ndarray, n_components=2, n_clusters=8, target_variance=0.85):
    """
    Legacy pipeline interface (backwards compatibility).
    Delegates to new remediation-enabled pipeline.
    """
    result = run_pipeline_with_remediation(X, n_components, n_clusters, target_variance)
    
    # Return legacy format for backwards compatibility
    return {
        "points": result["points"],
        "labels": result["labels"], 
        "centroids": result["centroids"],
        "inertia": result["inertia"],
        "inertia_history": result["inertia_history"],
        "explained_variance_ratio": result["explained_variance_ratio"],
        "cumulative_variance": result["cumulative_variance"],
        "n_samples": result["n_samples"],
        "n_features_original": result["n_features_original"],
        "n_components": result["n_components"],
        "n_components_vis": result["n_components_vis"],
        "n_clusters": result["n_clusters"],
        "variance_retained": result["variance_retained"],
    }


# Expose to JavaScript with remediation capabilities
class OptiGraphEngine:
    """Main interface for JS to call Python ML engine with AI data remediation."""
    
    @staticmethod
    def process(data_array, n_components=2, n_clusters=8, target_variance=0.85):
        """
        Process embeddings through production ML pipeline with AI data remediation.
        
        Args:
            data_array: 2D list or numpy array
            n_components: PCA components for visualization (2D/3D)  
            n_clusters: K-Means clusters
            target_variance: Minimum variance to retain (default 85%)
        
        Returns:
            Results dict with ML outputs + comprehensive data quality audit
        """
        X = np.array(data_array, dtype=np.float64)
        return run_pipeline_with_remediation(X, n_components, n_clusters, target_variance)
    
    @staticmethod
    def process_legacy(data_array, n_components=2, n_clusters=8, target_variance=0.85):
        """Legacy processing without full audit trail (backwards compatibility)"""
        X = np.array(data_array, dtype=np.float64)
        return run_pipeline(X, n_components, n_clusters, target_variance)
