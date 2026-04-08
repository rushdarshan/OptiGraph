import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';
import { Container, Header, Segment, Button, Divider, Message, Grid, Card, Statistic, Loader, Label, Input, Icon } from 'semantic-ui-react';
import { initPyodideAndLoadPackages, loadMLPipeline, runMLPipeline } from './pyodideHelper';
import './App.css';
import 'semantic-ui-css/semantic.min.css';

interface PipelineResults {
  points: number[][];
  labels: number[];
  centroids: number[][];
  inertia: number;
  inertia_history: number[];
  explained_variance_ratio: number[];
  cumulative_variance: number[];
  n_samples: number;
  n_features_original: number;
  n_components: number;
  n_clusters: number;
  runtime_ms: number;
}

const App: React.FC = () => {
  const [pyodideReady, setPyodideReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<PipelineResults | null>(null);
  const [nComponents, setNComponents] = useState(3);
  const [nClusters, setNClusters] = useState(8);

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
      setLoading(true);
      setError(null);
      
      const syntheticData = generateSyntheticData(500, 768);
      
      const pipelineResults = await runMLPipeline(syntheticData, nComponents, nClusters);
      setResults(pipelineResults);
      setLoading(false);
    } catch (err: any) {
      setError(`Pipeline error: ${err.message}`);
      setLoading(false);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !pyodideReady) return;

    try {
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
      setLoading(false);
    } catch (err: any) {
      setError(`File processing error: ${err.message}`);
      setLoading(false);
    }
  };

  const renderVisualization = () => {
    if (!results) return null;

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
    if (!results || !results.inertia_history) return null;

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
    if (!results || !results.explained_variance_ratio) return null;

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
    <Container className="app-container">
      <Segment className="header-segment" textAlign="center" padded="very">
        <Header as="h1" className="app-title">
          <Icon name="chart line" />
          OptiGraph
          <Header.Subheader>
            In-Browser Semantic Embedding Optimizer & Visualizer
          </Header.Subheader>
        </Header>
        
        <Message info className="msr-info">
          <Message.Header>
            <Icon name="graduation cap" />
            MSR Interview Talking Points
          </Message.Header>
          <Message.List>
            <Message.Item>
              <strong>Zero-Server Architecture:</strong> Entire ML pipeline (PCA + K-Means) runs client-side via PyOdide + WebAssembly
            </Message.Item>
            <Message.Item>
              <strong>From-Scratch Implementation:</strong> Custom NumPy-based PCA & K-Means without sklearn dependencies
            </Message.Item>
            <Message.Item>
              <strong>Production-Grade Performance:</strong> Handles 768-dim embeddings (BERT/OpenAI scale) with sub-second inference
            </Message.Item>
            <Message.Item>
              <strong>Privacy-First Design:</strong> No data leaves the browser - critical for sensitive enterprise embeddings
            </Message.Item>
            <Message.Item>
              <strong>Interactive Visualization:</strong> Real-time 2D/3D scatter plots with Plotly.js for exploratory analysis
            </Message.Item>
          </Message.List>
        </Message>
      </Segment>

      {!pyodideReady && (
        <Segment placeholder textAlign="center" padded="very">
          <Loader active inline="centered" size="large">
            Initializing PyOdide + NumPy ML Engine...
          </Loader>
          <p style={{ marginTop: '20px', color: '#666' }}>
            Loading WebAssembly runtime and Python packages (first load may take 10-15s)
          </p>
        </Segment>
      )}

      {pyodideReady && (
        <>
          <Segment>
            <Header as="h3">
              <Icon name="settings" />
              Configuration
            </Header>
            <Grid columns={2} stackable>
              <Grid.Column>
                <div className="config-control">
                  <Label pointing="right">PCA Components (2 or 3 for visualization)</Label>
                  <Input
                    type="number"
                    value={nComponents}
                    onChange={(e) => setNComponents(Math.max(2, Math.min(3, parseInt(e.target.value) || 2)))}
                    min={2}
                    max={3}
                    style={{ width: '100px' }}
                  />
                </div>
              </Grid.Column>
              <Grid.Column>
                <div className="config-control">
                  <Label pointing="right">K-Means Clusters</Label>
                  <Input
                    type="number"
                    value={nClusters}
                    onChange={(e) => setNClusters(Math.max(2, Math.min(20, parseInt(e.target.value) || 8)))}
                    min={2}
                    max={20}
                    style={{ width: '100px' }}
                  />
                </div>
              </Grid.Column>
            </Grid>

            <Divider />

            <div className="control-buttons">
              <Button
                primary
                size="large"
                icon
                labelPosition="left"
                onClick={handleRunDemo}
                disabled={loading}
                loading={loading}
              >
                <Icon name="play" />
                Run Demo (500 samples × 768 dims)
              </Button>

              <Button
                secondary
                size="large"
                icon
                labelPosition="left"
                onClick={() => document.getElementById('csv-upload')?.click()}
                disabled={loading}
              >
                <Icon name="upload" />
                Upload CSV Embeddings
              </Button>
              <input
                id="csv-upload"
                type="file"
                accept=".csv"
                style={{ display: 'none' }}
                onChange={handleFileUpload}
              />
            </div>

            <Message info>
              <Message.Header>CSV Format Requirements</Message.Header>
              <p>
                Each row = one embedding vector. Each column = one feature dimension.
                <br />
                Example: For 768-dim BERT embeddings, CSV should have 768 columns per row.
              </p>
            </Message>
          </Segment>

          {error && (
            <Message negative>
              <Message.Header>Error</Message.Header>
              <p>{error}</p>
            </Message>
          )}

          {loading && !results && (
            <Segment placeholder textAlign="center" padded="very">
              <Loader active inline="centered" size="large">
                Running ML Pipeline (PCA → K-Means)...
              </Loader>
            </Segment>
          )}

          {results && (
            <>
              <Segment>
                <Header as="h3">
                  <Icon name="chart bar" />
                  Pipeline Metrics
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
                  <Card>
                    <Card.Content textAlign="center">
                      <Statistic size="small">
                        <Statistic.Value>
                          {(results.cumulative_variance[results.cumulative_variance.length - 1] * 100).toFixed(1)}%
                        </Statistic.Value>
                        <Statistic.Label>Variance Retained</Statistic.Label>
                      </Statistic>
                    </Card.Content>
                  </Card>
                  <Card>
                    <Card.Content textAlign="center">
                      <Statistic size="small">
                        <Statistic.Value>{results.n_clusters}</Statistic.Value>
                        <Statistic.Label>Clusters</Statistic.Label>
                      </Statistic>
                    </Card.Content>
                  </Card>
                  <Card>
                    <Card.Content textAlign="center">
                      <Statistic size="small">
                        <Statistic.Value>{results.inertia_history.length}</Statistic.Value>
                        <Statistic.Label>Iterations</Statistic.Label>
                      </Statistic>
                    </Card.Content>
                  </Card>
                  <Card>
                    <Card.Content textAlign="center">
                      <Statistic size="small">
                        <Statistic.Value>{results.n_components}D</Statistic.Value>
                        <Statistic.Label>Reduced Dims</Statistic.Label>
                      </Statistic>
                    </Card.Content>
                  </Card>
                </Card.Group>
              </Segment>

              <Segment>
                <Header as="h3">
                  <Icon name="cube" />
                  3D Visualization
                </Header>
                {renderVisualization()}
              </Segment>

              <Grid columns={2} stackable>
                <Grid.Column>
                  <Segment>
                    {renderConvergencePlot()}
                  </Segment>
                </Grid.Column>
                <Grid.Column>
                  <Segment>
                    {renderVariancePlot()}
                  </Segment>
                </Grid.Column>
              </Grid>
            </>
          )}
        </>
      )}

      <Segment className="footer-segment" textAlign="center">
        <p>
          <strong>Technical Stack:</strong> React + TypeScript · PyOdide (Python 3.10) · NumPy · Plotly.js · WebAssembly
        </p>
        <p>
          <strong>Key Insight:</strong> Demonstrates how modern browsers can run production ML workloads without backend infrastructure
        </p>
      </Segment>
    </Container>
  );
};

export default App;
