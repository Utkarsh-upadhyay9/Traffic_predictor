import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './ResultsPanel.css';

function ResultsPanel({ prediction, comparison, loading }) {
  if (loading) {
    return (
      <div className="results-panel">
        <div className="loading-state">
          <div className="spinner"></div>
          <p>Analyzing traffic patterns...</p>
        </div>
      </div>
    );
  }

  if (comparison) {
    return (
      <div className="results-panel">
        <h2>📊 Scenario Comparison</h2>
        
        <div className="comparison-grid">
          <div className="metric-card baseline">
            <h3>Baseline</h3>
            <div className="metric-value">
              {comparison.baseline.predicted_travel_time_min.toFixed(1)} min
            </div>
            <div className="metric-label">Travel Time</div>
            <div className="metric-value">
              {(comparison.baseline.predicted_congestion_level * 100).toFixed(0)}%
            </div>
            <div className="metric-label">Congestion</div>
          </div>

          <div className="arrow">→</div>

          <div className="metric-card modified">
            <h3>Modified</h3>
            <div className="metric-value">
              {comparison.modified.predicted_travel_time_min.toFixed(1)} min
            </div>
            <div className="metric-label">Travel Time</div>
            <div className="metric-value">
              {(comparison.modified.predicted_congestion_level * 100).toFixed(0)}%
            </div>
            <div className="metric-label">Congestion</div>
          </div>
        </div>

        <div className="changes-section">
          <h3>📈 Impact Analysis</h3>
          <div className="change-item">
            <span className="change-label">Travel Time Change</span>
            <span className={`change-value ${comparison.changes.travel_time_change_pct > 0 ? 'negative' : 'positive'}`}>
              {comparison.changes.travel_time_change_pct > 0 ? '+' : ''}
              {comparison.changes.travel_time_change_pct.toFixed(1)}%
            </span>
          </div>
          <div className="change-item">
            <span className="change-label">Congestion Change</span>
            <span className={`change-value ${comparison.changes.congestion_change_pct > 0 ? 'negative' : 'positive'}`}>
              {comparison.changes.congestion_change_pct > 0 ? '+' : ''}
              {comparison.changes.congestion_change_pct.toFixed(1)}%
            </span>
          </div>
          <div className="change-item">
            <span className="change-label">Vehicle Count Change</span>
            <span className="change-value">
              {comparison.changes.vehicle_count_change_pct > 0 ? '+' : ''}
              {comparison.changes.vehicle_count_change_pct.toFixed(1)}%
            </span>
          </div>
        </div>

        <div className="recommendation">
          {comparison.recommendation}
        </div>
      </div>
    );
  }

  if (prediction) {
    const congestionColor = prediction.predicted_congestion_level > 0.7 ? '#e74c3c' : 
                           prediction.predicted_congestion_level > 0.4 ? '#f39c12' : '#27ae60';

    return (
      <div className="results-panel">
        <h2>🎯 Prediction Results</h2>
        
        <div className="metrics-grid">
          <div className="metric-box">
            <div className="metric-icon">⏱️</div>
            <div className="metric-content">
              <div className="metric-value">
                {prediction.predicted_travel_time_min.toFixed(1)}
                <span className="unit">min</span>
              </div>
              <div className="metric-label">Travel Time</div>
            </div>
          </div>

          <div className="metric-box">
            <div className="metric-icon">🚗</div>
            <div className="metric-content">
              <div className="metric-value">
                {prediction.predicted_vehicle_count}
                <span className="unit">vehicles</span>
              </div>
              <div className="metric-label">Traffic Volume</div>
            </div>
          </div>
        </div>

        <div className="congestion-indicator">
          <h3>Congestion Level</h3>
          <div className="congestion-bar">
            <div 
              className="congestion-fill" 
              style={{ 
                width: `${prediction.predicted_congestion_level * 100}%`,
                background: congestionColor
              }}
            ></div>
          </div>
          <div className="congestion-value" style={{ color: congestionColor }}>
            {(prediction.predicted_congestion_level * 100).toFixed(0)}%
          </div>
        </div>

        <div className="confidence-badge">
          <span className={`confidence ${prediction.confidence}`}>
            {prediction.confidence === 'high' ? '✓' : '~'} {prediction.confidence} confidence
          </span>
        </div>

        {prediction.timestamp && (
          <div className="timestamp">
            Predicted at: {new Date(prediction.timestamp).toLocaleTimeString()}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="results-panel">
      <div className="empty-state">
        <div className="empty-icon">📊</div>
        <h3>No Results Yet</h3>
        <p>Run a prediction or comparison to see results here</p>
      </div>
    </div>
  );
}

export default ResultsPanel;
