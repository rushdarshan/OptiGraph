/**
 * Lazy-loaded Plotly Visualization Component
 * Reduces initial bundle size by code splitting
 */

import React from 'react';
import Plot from 'react-plotly.js';
import { MLResult } from '../types';

interface PlotlyVisualizationProps {
  results: MLResult;
  nComponents: number;
}

const PlotlyVisualization: React.FC<PlotlyVisualizationProps> = ({ results, nComponents }) => {
  const renderVisualization = () => {
    const { points, labels, centroids } = results;
    
    const uniqueLabels = Array.from(new Set(labels));
    const colors = [
      '#e6194b', '#3cb44b', '#ffe119', '#4363d8',
      '#f58231', '#911eb4', '#46f0f0', '#f032e6',
      '#bcf60c', '#fabebe', '#008080', '#e6beff'
    ];

    const traces: any[] = uniqueLabels.map(label => {
      const clusterPoints = points.filter((_, idx) => labels[idx] === label);
      
      if (nComponents === 3) {
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
            opacity: 0.7,
          },
        };
      } else {
        return {
          x: clusterPoints.map(p => p[0]),
          y: clusterPoints.map(p => p[1]),
          mode: 'markers',
          type: 'scatter',
          name: `Cluster ${label}`,
          marker: {
            size: 8,
            color: colors[label % colors.length],
            opacity: 0.7,
          },
        };
      }
    });

    if (nComponents === 3 && centroids.length > 0) {
      traces.push({
        x: centroids.map(c => c[0]),
        y: centroids.map(c => c[1]),
        z: centroids.map(c => c[2]),
        mode: 'markers',
        type: 'scatter3d',
        name: 'Centroids',
        marker: {
          size: 10,
          color: '#000000',
          symbol: 'diamond',
          line: { color: '#ffffff', width: 2 },
        },
      });
    } else if (nComponents === 2 && centroids.length > 0) {
      traces.push({
        x: centroids.map(c => c[0]),
        y: centroids.map(c => c[1]),
        mode: 'markers',
        type: 'scatter',
        name: 'Centroids',
        marker: {
          size: 15,
          color: '#000000',
          symbol: 'star',
          line: { color: '#ffffff', width: 2 },
        },
      });
    }

    const layout: any = {
      title: `${nComponents}D Embedding Space (PCA + K-Means)`,
      autosize: true,
      height: 600,
      showlegend: true,
      legend: { x: 1.02, y: 1 },
    };

    if (nComponents === 3) {
      layout.scene = {
        xaxis: { title: 'PC1' },
        yaxis: { title: 'PC2' },
        zaxis: { title: 'PC3' },
      };
    } else {
      layout.xaxis = { title: 'PC1' };
      layout.yaxis = { title: 'PC2' };
    }

    return (
      <div className="visualization-container">
        <Plot
          data={traces}
          layout={layout}
          config={{ responsive: true, displayModeBar: true }}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    );
  };

  const renderConvergencePlot = () => {
    if (!results.inertia_history) return null;

    const trace: any = {
      x: results.inertia_history.map((_, idx) => idx),
      y: results.inertia_history,
      mode: 'lines+markers',
      type: 'scatter',
      name: 'Inertia',
      line: { color: '#4363d8', width: 2 },
      marker: { size: 6 },
    };

    const layout: any = {
      title: 'K-Means Convergence',
      xaxis: { title: 'Iteration' },
      yaxis: { title: 'Inertia' },
      height: 300,
    };

    return (
      <Plot
        data={[trace]}
        layout={layout}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%' }}
      />
    );
  };

  const renderVariancePlot = () => {
    if (!results.explained_variance_ratio) return null;

    const trace1: any = {
      x: results.explained_variance_ratio.map((_, idx) => `PC${idx + 1}`),
      y: results.explained_variance_ratio,
      type: 'bar',
      name: 'Individual Variance',
      marker: { color: '#3cb44b' },
    };

    const trace2: any = {
      x: results.cumulative_variance.map((_, idx) => `PC${idx + 1}`),
      y: results.cumulative_variance,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Cumulative Variance',
      yaxis: 'y2',
      line: { color: '#e6194b', width: 3 },
      marker: { size: 8 },
    };

    const layout: any = {
      title: 'PCA Variance Explained',
      xaxis: { title: 'Principal Component' },
      yaxis: { title: 'Variance Ratio', range: [0, 1] },
      yaxis2: {
        title: 'Cumulative Variance',
        overlaying: 'y',
        side: 'right',
        range: [0, 1],
      },
      height: 300,
      showlegend: true,
    };

    return (
      <Plot
        data={[trace1, trace2]}
        layout={layout}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: '100%' }}
      />
    );
  };

  return (
    <>
      {renderVisualization()}
      <div style={{ marginTop: '20px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        <div>{renderConvergencePlot()}</div>
        <div>{renderVariancePlot()}</div>
      </div>
    </>
  );
};

export default PlotlyVisualization;