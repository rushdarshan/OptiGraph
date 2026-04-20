import React, { useState, useEffect, Suspense } from 'react';
import { Container, Message, Card, Statistic, Segment, Header, Icon } from 'semantic-ui-react';
import { initPyodideAndLoadPackages, loadMLPipeline, runMLPipeline } from './pyodideHelper';
import { OptimizationMonitor } from './OptimizationMonitor';
import { ModernHeader } from './components/ModernHeader';
import { LoadingState } from './components/LoadingState';
import { ControlPanel } from './components/ControlPanel';
import { ErrorBoundary } from './components/ErrorBoundary';
import { Button } from './components/foundation/Button';
import { MLResult } from './types';
import './App.css';
import 'semantic-ui-css/semantic.min.css';

// Lazy load the heavy Plotly visualization component
const PlotlyVisualization = React.lazy(() => import('./components/PlotlyVisualization'));

interface PipelineResults extends MLResult {}

const App: React.FC = () => {
  const [pyodideReady, setPyodideReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<PipelineResults | null>(null);
  const [nComponents, setNComponents] = useState(3);
  const [nClusters, setNClusters] = useState(8);
  const [buttonState, setButtonState] = useState<'idle' | 'loading'>('idle');

  useEffect(() => {
    const initializePyodide = async () => {
      try {
        setLoading(true);
        await initPyodideAndLoadPackages();
        await loadMLPipeline();
        setPyodideReady(true);
        setLoading(false);
      } catch (err: any) {
        setError(`Failed to initialize Pyodide: ${err.message}`);
        setLoading(false);
      }
    };

    initializePyodide();
  }, []);

  const generateSyntheticData = (nSamples: number = 500, nFeatures: number = 768): number[][] => {
    const data: number[][] = [];
    const numClusters = 8;
    const samplesPerCluster = Math.floor(nSamples / numClusters);
    
    for (let cluster = 0; cluster < numClusters; cluster++) {
      const centerOffset = cluster * 2;
      for (let i = 0; i < samplesPerCluster; i++) {
        const sample: number[] = [];
        for (let j = 0; j < nFeatures; j++) {
          const base = Math.sin(cluster * 0.5 + j * 0.01) * centerOffset;
          const noise = (Math.random() - 0.5) * 0.5;
          sample.push(base + noise);
        }
        data.push(sample);
      }
    }
    
    return data;
  };

  const handleRunDemo = async () => {
    if (!pyodideReady) {
      setError("Pyodide not ready yet. Please wait.");
      return;
    }

    try {
      setButtonState('loading');
      setLoading(true);
      setError(null);
      
      const syntheticData = generateSyntheticData(500, 768);
      
      const pipelineResults = await runMLPipeline(syntheticData, nComponents, nClusters);
      setResults(pipelineResults);
      setButtonState('idle');
      setLoading(false);
    } catch (err: any) {
      setError(`Pipeline error: ${err.message}`);
      setButtonState('idle');
      setLoading(false);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !pyodideReady) return;

    try {
      setButtonState('loading');
      setLoading(true);
      setError(null);

      const text = await file.text();
      const lines = text.trim().split('\n');
      const data: number[][] = [];

      for (const line of lines) {
        const values = line.split(',').map(v => parseFloat(v.trim()));
        if (values.some(isNaN)) {
          throw new Error('Invalid CSV format. Ensure all values are numeric.');
        }
        data.push(values);
      }

      if (data.length === 0) {
        throw new Error('Empty CSV file');
      }

      const pipelineResults = await runMLPipeline(data, nComponents, nClusters);
      setResults(pipelineResults);
      setButtonState('idle');
      setLoading(false);
    } catch (err: any) {
      setError(`File processing error: ${err.message}`);
      setButtonState('idle');
      setLoading(false);
    }
  };

  return (
    <ErrorBoundary>
      <div className="modern-app">
        <ModernHeader />
        
        <Container className="app-container">
          {!pyodideReady && (
            <LoadingState 
              title="Initializing PyOdide + NumPy ML Engine"
              subtitle="Loading WebAssembly runtime and Python packages"
              stage="Setting up client-side ML environment..."
            />
          )}

          {pyodideReady && (
            <>
              <ControlPanel
                nComponents={nComponents}
                nClusters={nClusters}
                onComponentsChange={setNComponents}
                onClustersChange={setNClusters}
                onRunDemo={handleRunDemo}
                onFileUpload={handleFileUpload}
                loading={loading}
                pyodideReady={pyodideReady}
                buttonState={buttonState}
              />

              {error && (
                <Message negative>
                  <Message.Header>Processing Error</Message.Header>
                  <p>{error}</p>
                  <Button variant="secondary" size="sm" onClick={() => setError(null)}>
                    Dismiss
                  </Button>
                </Message>
              )}

              {loading && !results && (
                <LoadingState 
                  title="Running ML Pipeline"
                  subtitle="Processing data through PCA → K-Means clustering"
                  stage="Computing embeddings and clusters..."
                />
              )}

              {results && (
                <>
                  <Segment className="results-overview">
                    <Header as="h3">
                      <Icon name="chart bar" />
                      Pipeline Results
                    </Header>
                    <Card.Group itemsPerRow={4} stackable>
                      <Card>
                        <Card.Content textAlign="center">
                          <Statistic size="small">
                            <Statistic.Value>{results.n_samples}</Statistic.Value>
                            <Statistic.Label>Samples</Statistic.Label>
                          </Statistic>
                        </Card.Content>
                      </Card>
                      <Card>
                        <Card.Content textAlign="center">
                          <Statistic size="small">
                            <Statistic.Value>{results.n_features_original}</Statistic.Value>
                            <Statistic.Label>Original Dims</Statistic.Label>
                          </Statistic>
                        </Card.Content>
                      </Card>
                      <Card>
                        <Card.Content textAlign="center">
                          <Statistic size="small">
                            <Statistic.Value>{results.inertia.toFixed(2)}</Statistic.Value>
                            <Statistic.Label>Final Inertia</Statistic.Label>
                          </Statistic>
                        </Card.Content>
                      </Card>
                      <Card>
                        <Card.Content textAlign="center">
                          <Statistic size="small" color="green">
                            <Statistic.Value>{results.runtime_ms.toFixed(0)}ms</Statistic.Value>
                            <Statistic.Label>Runtime</Statistic.Label>
                          </Statistic>
                        </Card.Content>
                      </Card>
                    </Card.Group>
                  </Segment>

                  <Segment className="visualization-section">
                    <Header as="h3">
                      <Icon name="cube" />
                      Interactive Visualization
                    </Header>
                    <Suspense fallback={
                      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '600px' }}>
                        <LoadingState 
                          title="Loading Visualization"
                          subtitle="Preparing interactive 3D plot components"
                        />
                      </div>
                    }>
                      <PlotlyVisualization results={results} nComponents={nComponents} />
                    </Suspense>
                  </Segment>
                </>
              )}
            </>
          )}

          <footer className="modern-footer">
            <div className="footer-content">
              <div className="tech-stack">
                <h4>Technical Architecture</h4>
                <p>React 18 + TypeScript • PyOdide + WebAssembly • NumPy • Plotly.js</p>
              </div>
              <div className="key-insight">
                <h4>Research Contribution</h4>
                <p>Demonstrates client-side ML optimization with financial guardrails</p>
              </div>
            </div>
          </footer>
          
          <OptimizationMonitor />
        </Container>
      </div>
    </ErrorBoundary>
  );
};

export default App;
