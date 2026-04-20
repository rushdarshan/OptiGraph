/**
 * Professional Loading State Component
 * Modern, engaging loading experience
 */

import React from 'react';
import './LoadingState.css';

interface LoadingStateProps {
  title?: string;
  subtitle?: string;
  progress?: number;
  stage?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({
  title = "Initializing ML Engine",
  subtitle = "Setting up PyOdide and NumPy packages",
  progress,
  stage
}) => {
  return (
    <div className="loading-container">
      <div className="loading-content">
        {/* Animated Icon */}
        <div className="loading-icon">
          <svg width="48" height="48" viewBox="0 0 48 48" className="spinner">
            <circle
              cx="24"
              cy="24"
              r="20"
              fill="none"
              stroke="#e2e8f0"
              strokeWidth="4"
            />
            <circle
              cx="24"
              cy="24"
              r="20"
              fill="none"
              stroke="#3b82f6"
              strokeWidth="4"
              strokeLinecap="round"
              strokeDasharray="126"
              strokeDashoffset="126"
              className="progress-circle"
            />
          </svg>
          <div className="loading-dots">
            <div className="dot"></div>
            <div className="dot"></div>
            <div className="dot"></div>
          </div>
        </div>

        {/* Loading Text */}
        <div className="loading-text">
          <h3>{title}</h3>
          <p>{subtitle}</p>
          {stage && <span className="loading-stage">{stage}</span>}
        </div>

        {/* Progress Bar */}
        {progress !== undefined && (
          <div className="progress-container">
            <div className="progress-bar">
              <div 
                className="progress-fill"
                style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
              />
            </div>
            <span className="progress-text">{Math.round(progress)}%</span>
          </div>
        )}

        {/* Technical Details */}
        <div className="loading-details">
          <div className="detail-item">
            <span className="detail-label">Runtime:</span>
            <span className="detail-value">WebAssembly + Python 3.10</span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Packages:</span>
            <span className="detail-value">NumPy, PyOdide Core</span>
          </div>
          <div className="detail-item">
            <span className="detail-label">Status:</span>
            <span className="detail-value status-active">Initializing...</span>
          </div>
        </div>
      </div>
    </div>
  );
};