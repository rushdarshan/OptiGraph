import React, { ReactNode } from 'react';
import { tokens } from '../tokens/tokens';
import './ErrorBoundary.css';

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
    this.setState({
      error,
      errorInfo,
    });
  }

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <div className="error-container">
            <div className="error-icon">⚠️</div>
            <h2 className="error-title">Something went wrong</h2>
            <p className="error-message">
              The application encountered an unexpected error. Please try again or refresh the page.
            </p>

            {process.env.NODE_ENV === 'development' && this.state.error && (
              <details className="error-details">
                <summary>Error details (development only)</summary>
                <pre className="error-stack">
                  <code>
                    {this.state.error.toString()}
                    {'\n\n'}
                    {this.state.errorInfo?.componentStack}
                  </code>
                </pre>
              </details>
            )}

            <div className="error-actions">
              <button
                className="error-button"
                onClick={this.handleReset}
                style={{
                  backgroundColor: tokens.colors.primary[500],
                  color: '#ffffff',
                  padding: `${tokens.spacing.sm} ${tokens.spacing.base}`,
                  borderRadius: tokens.borderRadius.md,
                  border: 'none',
                  cursor: 'pointer',
                  fontSize: tokens.typography.fontSize.base,
                  fontFamily: tokens.typography.fontFamily.body,
                  fontWeight: tokens.typography.fontWeight.semibold,
                }}
              >
                Try Again
              </button>
              <button
                className="error-button error-button-secondary"
                onClick={() => window.location.href = '/'}
                style={{
                  backgroundColor: tokens.colors.neutral[100],
                  color: tokens.colors.neutral[900],
                  padding: `${tokens.spacing.sm} ${tokens.spacing.base}`,
                  borderRadius: tokens.borderRadius.md,
                  border: `2px solid ${tokens.colors.neutral[300]}`,
                  cursor: 'pointer',
                  fontSize: tokens.typography.fontSize.base,
                  fontFamily: tokens.typography.fontFamily.body,
                  fontWeight: tokens.typography.fontWeight.semibold,
                }}
              >
                Go Home
              </button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
