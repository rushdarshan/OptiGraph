/**
 * Modern Professional Header Component
 * Clean, sophisticated design for technical audiences
 */

import React from 'react';
import './ModernHeader.css';

export const ModernHeader: React.FC = () => {
  return (
    <header className="modern-header">
      <div className="header-content">
        <div className="logo-section">
          <div className="logo-icon">
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
              <path 
                d="M8 24L24 8M24 8H16M24 8V16" 
                stroke="currentColor" 
                strokeWidth="2.5" 
                strokeLinecap="round" 
                strokeLinejoin="round"
              />
              <circle cx="8" cy="24" r="2" fill="currentColor"/>
              <circle cx="16" cy="16" r="2" fill="currentColor"/>
              <circle cx="24" cy="8" r="2" fill="currentColor"/>
            </svg>
          </div>
          <div className="logo-text">
            <h1>OptiGraph</h1>
            <p>Semantic Embedding Optimizer</p>
          </div>
        </div>

        <div className="header-meta">
          <span className="version">v1.0</span>
          <span className="tech-badge">WebAssembly</span>
        </div>
      </div>

      <div className="capabilities-grid">
        <div className="capability">
          <div className="capability-icon">⚡</div>
          <div>
            <h3>Client-Side Processing</h3>
            <p>Full ML pipeline runs in-browser via PyOdide + WASM</p>
          </div>
        </div>
        
        <div className="capability">
          <div className="capability-icon">🔬</div>
          <div>
            <h3>From-Scratch Algorithms</h3>
            <p>Custom NumPy-based PCA & K-Means implementation</p>
          </div>
        </div>
        
        <div className="capability">
          <div className="capability-icon">🚀</div>
          <div>
            <h3>Production Performance</h3>
            <p>768-dim embeddings processed in sub-second time</p>
          </div>
        </div>
        
        <div className="capability">
          <div className="capability-icon">🔒</div>
          <div>
            <h3>Privacy-First</h3>
            <p>Zero data transmission - everything stays local</p>
          </div>
        </div>
      </div>
    </header>
  );
};