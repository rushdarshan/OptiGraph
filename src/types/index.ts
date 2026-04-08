export interface MLResult {
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

export interface DatasetInfo {
  name: string;
  n_samples: number;
  n_features: number;
  description: string;
}
