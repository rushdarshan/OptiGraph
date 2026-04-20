/**
 * Professional Control Panel Component
 * Clean, functional design for ML parameters
 */

import React from 'react';
import { Button } from './foundation/Button';
import './ControlPanel.css';

interface ControlPanelProps {
  nComponents: number;
  nClusters: number;
  onComponentsChange: (value: number) => void;
  onClustersChange: (value: number) => void;
  onRunDemo: () => void;
  onFileUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  loading: boolean;
  pyodideReady: boolean;
  buttonState?: 'idle' | 'loading' | 'error' | 'success';
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  nComponents,
  nClusters,
  onComponentsChange,
  onClustersChange,
  onRunDemo,
  onFileUpload,
  loading,
  pyodideReady,
  buttonState = 'idle'
}) => {
  return (
    <div className="control-panel">
      <div className="panel-header">
        <h2>Pipeline Configuration</h2>
        <div className="status-indicator">
          <div className={`status-dot ${pyodideReady ? 'ready' : 'loading'}`} />
          <span className="status-text">
            {pyodideReady ? 'Engine Ready' : 'Initializing...'}
          </span>
        </div>
      </div>

      <div className="controls-grid">
        {/* PCA Components Control */}
        <div className="control-group">
          <label className="control-label">
            <span className="label-text">PCA Components</span>
            <span className="label-hint">Dimensionality for visualization</span>
          </label>
          <div className="slider-container">
            <input
              type="range"
              min="2"
              max="3"
              value={nComponents}
              onChange={(e) => onComponentsChange(parseInt(e.target.value))}
              className="slider"
              disabled={loading}
            />
            <div className="slider-labels">
              <span className={nComponents === 2 ? 'active' : ''}>2D</span>
              <span className={nComponents === 3 ? 'active' : ''}>3D</span>
            </div>
          </div>
          <div className="current-value">{nComponents}D Space</div>
        </div>

        {/* K-Means Clusters Control */}
        <div className="control-group">
          <label className="control-label">
            <span className="label-text">K-Means Clusters</span>
            <span className="label-hint">Number of cluster groups</span>
          </label>
          <div className="number-input-container">
            <button 
              type="button"
              className="number-btn"
              onClick={() => onClustersChange(Math.max(2, nClusters - 1))}
              disabled={loading || nClusters <= 2}
            >
              −
            </button>
            <input
              type="number"
              min="2"
              max="20"
              value={nClusters}
              onChange={(e) => onClustersChange(parseInt(e.target.value) || 2)}
              className="number-input"
              disabled={loading}
            />
            <button 
              type="button"
              className="number-btn"
              onClick={() => onClustersChange(Math.min(20, nClusters + 1))}
              disabled={loading || nClusters >= 20}
            >
              +
            </button>
          </div>
          <div className="current-value">{nClusters} clusters</div>
        </div>
      </div>

      <div className="actions-section">
        <Button
          variant="primary"
          size="md"
          state={buttonState}
          disabled={!pyodideReady || loading}
          onClick={onRunDemo}
          title="Run ML pipeline with current parameters"
        >
          {buttonState === 'loading' ? 'Processing...' : 'Run Demo Pipeline'}
        </Button>

        <Button
          variant="secondary"
          size="md"
          state={loading ? 'disabled' : 'idle'}
          disabled={!pyodideReady || loading}
          onClick={() => document.getElementById('csv-upload')?.click()}
          title="Upload CSV data file for processing"
        >
          Upload CSV Data
        </Button>

        <input
          id="csv-upload"
          type="file"
          accept=".csv"
          style={{ display: 'none' }}
          onChange={onFileUpload}
        />
      </div>

      <div className="data-format-info">
        <h4>CSV Format Requirements</h4>
        <ul>
          <li>Each row represents one embedding vector</li>
          <li>Each column represents one feature dimension</li>
          <li>Example: 768-column CSV for BERT embeddings</li>
          <li>Numeric values only, no headers required</li>
        </ul>
      </div>
    </div>
  );
};