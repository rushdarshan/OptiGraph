import React from 'react';
import Plot from 'react-plotly.js';
import { MLResult } from '../types';

interface Props {
  result: MLResult | null;
}

const ClusterVisualization: React.FC<Props> = ({ result }) => {
  if (!result) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: '#666' }}>
        <h3>No results yet</h3>
        <p>Upload a CSV file with embeddings to visualize clusters</p>
      </div>
    );
  }

  const { points, labels, centroids, n_components } = result;

  const uniqueLabels = Array.from(new Set(labels));
  const colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
  ];

  const traces: any[] = uniqueLabels.map((label) => {
    const clusterPoints = points.filter((_, idx) => labels[idx] === label);
    
    if (n_components === 2) {
      return {
        x: clusterPoints.map(p => p[0]),
        y: clusterPoints.map(p => p[1]),
        mode: 'markers',
        type: 'scatter',
        name: `Cluster ${label}`,
        marker: {
          size: 6,
          color: colors[label % colors.length],
          opacity: 0.7
        }
      };
    } else {
      return {
        x: clusterPoints.map(p => p[0]),
        y: clusterPoints.map(p => p[1]),
        z: clusterPoints.map(p => p[2]),
        mode: 'markers',
        type: 'scatter3d',
        name: `Cluster ${label}`,
        marker: {
          size: 4,
          color: colors[label % colors.length],
          opacity: 0.7
        }
      };
    }
  });

  if (n_components === 2) {
    traces.push({
      x: centroids.map(c => c[0]),
      y: centroids.map(c => c[1]),
      mode: 'markers',
      type: 'scatter',
      name: 'Centroids',
      marker: {
        size: 15,
        color: 'black',
        symbol: 'x',
        line: { width: 2, color: 'white' }
      }
    });
  } else {
    traces.push({
      x: centroids.map(c => c[0]),
      y: centroids.map(c => c[1]),
      z: centroids.map(c => c[2]),
      mode: 'markers',
      type: 'scatter3d',
      name: 'Centroids',
      marker: {
        size: 10,
        color: 'black',
        symbol: 'cross'
      }
    });
  }

  const layout: any = {
    title: `${n_components}D Cluster Visualization (PCA + K-Means)`,
    showlegend: true,
    hovermode: 'closest',
    paper_bgcolor: '#f8f9fa',
    plot_bgcolor: '#ffffff',
  };

  if (n_components === 2) {
    layout.xaxis = { title: 'PC1', gridcolor: '#e0e0e0' };
    layout.yaxis = { title: 'PC2', gridcolor: '#e0e0e0' };
  } else {
    layout.scene = {
      xaxis: { title: 'PC1', backgroundcolor: '#ffffff', gridcolor: '#e0e0e0' },
      yaxis: { title: 'PC2', backgroundcolor: '#ffffff', gridcolor: '#e0e0e0' },
      zaxis: { title: 'PC3', backgroundcolor: '#ffffff', gridcolor: '#e0e0e0' },
    };
  }

  return (
    <div style={{ width: '100%', height: '600px' }}>
      <Plot
        data={traces}
        layout={layout}
        style={{ width: '100%', height: '100%' }}
        config={{ responsive: true }}
      />
    </div>
  );
};

export default ClusterVisualization;
