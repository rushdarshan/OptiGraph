import React from 'react';
import { MLResult } from '../types';

interface Props {
  result: MLResult | null;
}

const MetricsDashboard: React.FC<Props> = ({ result }) => {
  if (!result) return null;

  const {
    n_samples,
    n_features_original,
    n_components,
    n_clusters,
    inertia,
    explained_variance_ratio,
    cumulative_variance,
    runtime_ms
  } = result;

  const formatNumber = (num: number) => num.toLocaleString();
  const formatPercent = (num: number) => (num * 100).toFixed(2) + '%';
  const formatTime = (ms: number) => ms.toFixed(2) + 'ms';

  return (
    <div style={{ 
      padding: '20px', 
      backgroundColor: '#ffffff', 
      borderRadius: '8px',
      boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
      marginBottom: '20px'
    }}>
      <h2 style={{ marginTop: 0, marginBottom: '20px', color: '#333' }}>
        📊 Performance Metrics & Analysis
      </h2>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px' }}>
        <MetricCard 
          title="Samples" 
          value={formatNumber(n_samples)}
          subtitle="Data points"
          icon="📦"
        />
        <MetricCard 
          title="Original Dimensions" 
          value={formatNumber(n_features_original)}
          subtitle="Features"
          icon="📐"
        />
        <MetricCard 
          title="PCA Components" 
          value={n_components.toString()}
          subtitle={`Reduced to ${n_components}D`}
          icon="🎯"
        />
        <MetricCard 
          title="Clusters (K)" 
          value={n_clusters.toString()}
          subtitle="K-Means groups"
          icon="🔵"
        />
        <MetricCard 
          title="Inertia" 
          value={inertia.toFixed(2)}
          subtitle="Cluster tightness"
          icon="🎪"
        />
        <MetricCard 
          title="Runtime" 
          value={formatTime(runtime_ms)}
          subtitle="WASM in-browser"
          icon="⚡"
        />
      </div>

      <div style={{ marginTop: '30px', padding: '15px', backgroundColor: '#f0f8ff', borderRadius: '6px' }}>
        <h3 style={{ marginTop: 0, color: '#0066cc' }}>
          🔬 PCA Variance Explained
        </h3>
        <div style={{ marginTop: '15px' }}>
          {explained_variance_ratio.map((ratio, idx) => (
            <div key={idx} style={{ marginBottom: '12px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                <span style={{ fontWeight: 'bold' }}>PC{idx + 1}:</span>
                <span>{formatPercent(ratio)} | Cumulative: {formatPercent(cumulative_variance[idx])}</span>
              </div>
              <div style={{ 
                height: '8px', 
                backgroundColor: '#e0e0e0', 
                borderRadius: '4px',
                overflow: 'hidden'
              }}>
                <div style={{ 
                  height: '100%', 
                  width: formatPercent(ratio),
                  backgroundColor: '#0066cc',
                  transition: 'width 0.3s ease'
                }} />
              </div>
            </div>
          ))}
        </div>
        
        <div style={{ 
          marginTop: '15px', 
          padding: '10px', 
          backgroundColor: '#ffffff', 
          borderRadius: '4px',
          border: '1px solid #b3d9ff'
        }}>
          <strong>💡 Interpretation:</strong> Top {n_components} components retain{' '}
          <strong>{formatPercent(cumulative_variance[n_components - 1])}</strong> of original variance.
          {cumulative_variance[n_components - 1] > 0.8 ? 
            ' ✅ Excellent dimensionality reduction!' : 
            ' ⚠️ Consider adding more components for better retention.'}
        </div>
      </div>

      <div style={{ 
        marginTop: '20px', 
        padding: '15px', 
        backgroundColor: '#fff3cd', 
        borderRadius: '6px',
        border: '1px solid #ffc107'
      }}>
        <h4 style={{ marginTop: 0, color: '#856404' }}>
          🎓 MSR Interview Talking Points
        </h4>
        <ul style={{ marginBottom: 0, paddingLeft: '20px', lineHeight: '1.8' }}>
          <li>
            <strong>Mathematical Foundation:</strong> Implemented PCA via eigen-decomposition of covariance matrix (not sklearn).
          </li>
          <li>
            <strong>Algorithmic Optimization:</strong> K-Means++ initialization reduces convergence time vs random init.
          </li>
          <li>
            <strong>Scalability Tradeoff:</strong> {runtime_ms < 500 ? 
              `In-browser WASM processing achieved ${formatTime(runtime_ms)} latency - zero backend cost` :
              `WASM processing at ${formatTime(runtime_ms)} - acceptable for ${formatNumber(n_samples)} samples`}.
          </li>
          <li>
            <strong>Dimensionality Statistics:</strong> Reduced from {n_features_original}D → {n_components}D while retaining {formatPercent(cumulative_variance[n_components - 1])} variance.
          </li>
        </ul>
      </div>
    </div>
  );
};

const MetricCard: React.FC<{ title: string; value: string; subtitle: string; icon: string }> = 
  ({ title, value, subtitle, icon }) => (
    <div style={{ 
      padding: '15px', 
      backgroundColor: '#f8f9fa', 
      borderRadius: '6px',
      border: '1px solid #e0e0e0'
    }}>
      <div style={{ fontSize: '24px', marginBottom: '8px' }}>{icon}</div>
      <div style={{ fontSize: '12px', color: '#666', marginBottom: '4px' }}>{title}</div>
      <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#333', marginBottom: '4px' }}>
        {value}
      </div>
      <div style={{ fontSize: '11px', color: '#888' }}>{subtitle}</div>
    </div>
);

export default MetricsDashboard;
