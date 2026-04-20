"""
Tests for AI Data Remediation Engine
Validates anomaly detection, semantic clustering, and fix generation
"""
import pytest
import numpy as np
import sys
import os

# Add python directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_remediation import DataRemediationEngine, AnomalyRecord
from ml_pipeline import run_pipeline_with_remediation


class TestDataRemediationEngine:
    """Test suite for AI-powered data remediation"""
    
    def setup_method(self):
        """Setup for each test"""
        self.engine = DataRemediationEngine(confidence_threshold=0.75, enable_local_llm=False)
    
    def test_clean_data_validation(self):
        """Should pass clean data without anomalies"""
        clean_data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        
        clean_result, anomalies = self.engine.validate_input_data(clean_data)
        
        assert len(clean_result) == 3
        assert len(anomalies) == 0
        assert clean_result == clean_data
        
        # Verify reconciliation
        reconciliation = self.engine.get_reconciliation_report()
        assert reconciliation['source_rows'] == 3
        assert reconciliation['success_rows'] == 3
        assert not reconciliation['data_loss_detected']
    
    def test_nan_anomaly_detection(self):
        """Should detect NaN/Inf values as critical anomalies"""
        data_with_nans = [
            [1.0, 2.0, 3.0],           # Clean
            [4.0, float('nan'), 6.0],  # Contains NaN
            [7.0, 8.0, float('inf')],  # Contains Inf
            [10.0, 11.0, 12.0]         # Clean
        ]
        
        clean_result, anomalies = self.engine.validate_input_data(data_with_nans)
        
        # Should have 2 clean rows, 2 anomalies
        assert len(clean_result) == 2
        assert len(anomalies) == 2
        
        # Verify anomaly properties
        nan_anomaly = next(a for a in anomalies if 'nan' in str(a.original_value).lower())
        assert nan_anomaly.error_type == "NaN_or_Inf_values"
        assert nan_anomaly.severity == "CRITICAL"
        assert len(nan_anomaly.pk_hash) == 16  # SHA256 first 16 chars
        
        inf_anomaly = next(a for a in anomalies if 'inf' in str(a.original_value).lower())
        assert inf_anomaly.error_type == "NaN_or_Inf_values"
        assert inf_anomaly.severity == "CRITICAL"
    
    def test_dimension_mismatch_detection(self):
        """Should detect inconsistent row dimensions"""
        data_inconsistent_dims = [
            [1.0, 2.0, 3.0],        # 3 features
            [4.0, 5.0, 6.0, 7.0],   # 4 features - ANOMALY
            [8.0, 9.0, 10.0],       # 3 features
            [11.0, 12.0]            # 2 features - ANOMALY
        ]
        
        clean_result, anomalies = self.engine.validate_input_data(data_inconsistent_dims)
        
        assert len(clean_result) == 2  # Only consistent dimension rows
        assert len(anomalies) == 2    # Two dimension mismatches
        
        # Check dimension mismatch anomalies
        dim_anomalies = [a for a in anomalies if a.error_type == "dimension_mismatch"]
        assert len(dim_anomalies) == 2
        
        for anomaly in dim_anomalies:
            assert anomaly.severity == "HIGH"
            assert "expected_3_got_" in anomaly.original_value
    
    def test_extreme_outlier_detection(self):
        """Should detect statistical extreme outliers"""
        # Create data with extreme outliers
        normal_values = [1.0, 2.0, 3.0, 4.0, 5.0] * 10  # 50 normal values
        extreme_outlier = [1e10]  # Extremely large value
        
        outlier_data = [
            normal_values + [6.0],           # Normal
            normal_values + extreme_outlier, # Contains extreme outlier  
            normal_values + [7.0],           # Normal
        ]
        
        clean_result, anomalies = self.engine.validate_input_data(outlier_data)
        
        # Should detect the extreme outlier
        outlier_anomalies = [a for a in anomalies if a.error_type == "extreme_outlier"]
        assert len(outlier_anomalies) >= 1
        
        if outlier_anomalies:
            outlier_anomaly = outlier_anomalies[0]
            assert outlier_anomaly.severity == "MEDIUM"
            assert "1e+10" in str(outlier_anomaly.original_value) or "1000000000000" in str(outlier_anomaly.original_value)
    
    def test_semantic_clustering(self):
        """Should cluster similar anomalies semantically"""
        # Create multiple NaN anomalies (should cluster together)
        nan_data = [
            [1.0, float('nan'), 3.0],
            [4.0, float('nan'), 6.0], 
            [7.0, float('nan'), 9.0],
        ]
        
        clean_result, anomalies = self.engine.validate_input_data(nan_data)
        
        # All should be NaN anomalies
        assert len(anomalies) == 3
        assert all(a.error_type == "NaN_or_Inf_values" for a in anomalies)
        
        # Test semantic clustering
        clusters = self.engine._cluster_anomalies_semantically(anomalies)
        
        # All NaN anomalies should cluster together (or in very few clusters)
        assert len(clusters) <= 2  # Should be highly clustered
        
        # Convert to list if it's a dict (handle both return types)
        if isinstance(clusters, dict):
            cluster_list = list(clusters.values())
        else:
            cluster_list = clusters
            
        # Largest cluster should contain most anomalies
        cluster_sizes = [len(cluster) for cluster in cluster_list]
        max_cluster_size = max(cluster_sizes)
        assert max_cluster_size >= 2  # Similar anomalies clustered
    
    def test_deterministic_nan_fixing(self):
        """Should fix NaN values deterministically when LLM disabled"""
        nan_anomalies = [
            AnomalyRecord(
                row_index=0,
                column_index=1, 
                original_value=float('nan'),
                error_type="NaN_or_Inf_values",
                confidence=0.95,
                fix_suggestion=0.0
            )
        ]
        
        fixed_rows, quarantine = self.engine.remediate_anomalies(nan_anomalies)
        
        assert len(fixed_rows) >= 1
        fixed_row = fixed_rows[0]
        assert 'row_index' in fixed_row
        assert fixed_row['confidence'] >= 0.9
    
    def test_lambda_safety_validation(self):
        """Should reject unsafe lambda expressions"""
        # Safe lambdas
        safe_lambdas = [
            "lambda x: x",
            "lambda x: 0.0 if np.isnan(x) else x", 
            "lambda x: np.clip(x, -1000, 1000)"
        ]
        
        for safe_lambda in safe_lambdas:
            assert self.engine._validate_lambda_safety(safe_lambda)
        
        # Unsafe lambdas  
        unsafe_lambdas = [
            "import os; lambda x: x",
            "lambda x: exec('print(x)')",
            "lambda x: eval(x)",
            "lambda x: os.system('rm -rf /')",
            "lambda x: __import__('subprocess').call(['ls'])", 
            "def foo(): pass"  # Not a lambda
        ]
        
        for unsafe_lambda in unsafe_lambdas:
            assert not self.engine._validate_lambda_safety(unsafe_lambda)
    
    def test_reconciliation_guarantee(self):
        """Should maintain mathematical zero-data-loss guarantee"""
        mixed_data = [
            [1.0, 2.0, 3.0],           # Clean
            [4.0, float('nan'), 6.0],  # NaN anomaly
            [7.0, 8.0, 9.0],           # Clean  
            [10.0, 11.0],              # Dimension anomaly
        ]
        
        clean_result, anomalies = self.engine.validate_input_data(mixed_data)
        fixed_data, quarantine = self.engine.remediate_anomalies(anomalies)
        
        reconciliation = self.engine.get_reconciliation_report()
        
        # Mathematical guarantee: Source = Success + Quarantine (NOT including ai_fixed)
        # AI-fixed rows are ALREADY included in the success count  
        assert reconciliation['source_rows'] == (reconciliation['success_rows'] + 
                                                 reconciliation['quarantine_rows'])
        assert not reconciliation['data_loss_detected']
        assert reconciliation['missing_rows'] == 0
    
    def test_audit_trail_completeness(self):
        """Should maintain complete audit trail for all operations"""
        test_data = [
            [1.0, 2.0],
            [3.0, float('nan')], 
            [5.0, 6.0, 7.0],  # Dimension mismatch
        ]
        
        clean_result, anomalies = self.engine.validate_input_data(test_data)
        fixed_data, quarantine = self.engine.remediate_anomalies(anomalies)
        
        audit_trail = self.engine.get_audit_trail()
        
        # Should have complete audit records
        assert 'anomalies' in audit_trail
        assert 'fixes' in audit_trail
        assert 'reconciliation' in audit_trail
        
        # Anomaly records should be serializable
        for anomaly_dict in audit_trail['anomalies']:
            assert 'row_id' in anomaly_dict
            assert 'error_type' in anomaly_dict
            assert 'severity' in anomaly_dict
            assert 'timestamp' in anomaly_dict
            assert 'pk_hash' in anomaly_dict
        
        # Reconciliation should be complete
        reconciliation = audit_trail['reconciliation']
        assert reconciliation['source_rows'] > 0
        assert 'data_loss_detected' in reconciliation


class TestIntegratedPipeline:
    """Test full ML pipeline with remediation integration"""
    
    def test_pipeline_with_clean_data(self):
        """Should process clean data normally"""
        clean_data = np.random.randn(100, 10)  # 100 samples, 10 features
        
        result = run_pipeline_with_remediation(clean_data, n_components=3, n_clusters=5)
        
        # Should have normal ML results
        assert 'points' in result
        assert 'labels' in result
        assert 'centroids' in result
        assert len(result['points']) == 100
        assert len(result['labels']) == 100
        
        # Should have data quality report
        assert 'data_quality' in result
        data_quality = result['data_quality']
        assert data_quality['anomalies_detected'] == 0
        assert data_quality['remediation_success_rate'] == 1.0
        
        # Should have audit trail
        assert 'audit_trail' in result
        assert 'reconciliation' in result
    
    def test_pipeline_with_anomalous_data(self):
        """Should remediate anomalies and continue processing"""
        # Create data with NaN anomalies
        data_with_issues = [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [6.0, float('nan'), 8.0, 9.0, 10.0],  # NaN
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, float('inf'), 19.0, 20.0],  # Inf  
            [21.0, 22.0, 23.0, 24.0, 25.0],
        ]
        
        result = run_pipeline_with_remediation(
            np.array(data_with_issues), 
            n_components=2, 
            n_clusters=3
        )
        
        # Should successfully process despite anomalies
        assert 'points' in result
        assert len(result['points']) > 0  # Some data should be processed
        
        # Should report data quality issues
        data_quality = result['data_quality']
        assert data_quality['anomalies_detected'] > 0
        assert data_quality['anomalies_fixed'] >= 0
        
        # Should maintain data integrity
        reconciliation = result['reconciliation']
        assert not reconciliation['data_loss_detected']
    
    def test_pipeline_backward_compatibility(self):
        """Legacy pipeline interface should still work"""
        from ml_pipeline import run_pipeline
        
        clean_data = np.random.randn(50, 8)
        
        # Legacy interface should work
        legacy_result = run_pipeline(clean_data, n_components=2, n_clusters=4)
        
        # Should have expected legacy fields
        expected_fields = [
            'points', 'labels', 'centroids', 'inertia',
            'explained_variance_ratio', 'n_samples', 'n_clusters'
        ]
        
        for field in expected_fields:
            assert field in legacy_result


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])