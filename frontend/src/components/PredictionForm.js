import React, { useState } from 'react';
import axios from 'axios';
import './PredictionForm.css';

function PredictionForm({ onPrediction, onComparison, setLoading }) {
  const [mode, setMode] = useState('single'); // 'single' or 'compare'
  
  // Single prediction state
  const [formData, setFormData] = useState({
    hour: new Date().getHours(),
    day_of_week: new Date().getDay(),
    num_lanes: 3,
    road_capacity: 2000,
    current_vehicle_count: 1000,
    weather_condition: 0,
    is_holiday: false,
    road_closure: false,
    speed_limit: 55
  });

  // Comparison state
  const [baselineData, setBaselineData] = useState({
    hour: 8,
    num_lanes: 3,
    current_vehicle_count: 1500,
    road_closure: false
  });

  const [modifiedData, setModifiedData] = useState({
    hour: 8,
    num_lanes: 3,
    current_vehicle_count: 1500,
    road_closure: true
  });

  const handleSingleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : (type === 'number' ? Number(value) : value)
    }));
  };

  const handleBaselineChange = (e) => {
    const { name, value, type, checked } = e.target;
    setBaselineData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : Number(value)
    }));
  };

  const handleModifiedChange = (e) => {
    const { name, value, type, checked } = e.target;
    setModifiedData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : Number(value)
    }));
  };

  const handleSingleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const response = await axios.post('/api/predict', null, {
        params: formData
      });
      onPrediction(response.data);
    } catch (error) {
      console.error('Prediction error:', error);
      alert('Error making prediction. Make sure backend is running!');
    } finally {
      setLoading(false);
    }
  };

  const handleCompareSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    try {
      const response = await axios.post('/api/compare', {
        baseline: baselineData,
        modified: modifiedData
      });
      onComparison(response.data);
    } catch (error) {
      console.error('Comparison error:', error);
      alert('Error comparing scenarios. Make sure backend is running!');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="prediction-form">
      <div className="form-header">
        <h2>Traffic Predictor</h2>
        <div className="mode-toggle">
          <button 
            className={mode === 'single' ? 'active' : ''} 
            onClick={() => setMode('single')}
          >
            Single Prediction
          </button>
          <button 
            className={mode === 'compare' ? 'active' : ''} 
            onClick={() => setMode('compare')}
          >
            Compare Scenarios
          </button>
        </div>
      </div>

      {mode === 'single' ? (
        <form onSubmit={handleSingleSubmit} className="form-content">
          <div className="form-section">
            <h3>Time & Date</h3>
            <div className="form-row">
              <div className="form-group">
                <label>Hour of Day</label>
                <input
                  type="number"
                  name="hour"
                  min="0"
                  max="23"
                  value={formData.hour}
                  onChange={handleSingleChange}
                />
              </div>
              <div className="form-group">
                <label>Day of Week</label>
                <select
                  name="day_of_week"
                  value={formData.day_of_week}
                  onChange={handleSingleChange}
                >
                  <option value="0">Monday</option>
                  <option value="1">Tuesday</option>
                  <option value="2">Wednesday</option>
                  <option value="3">Thursday</option>
                  <option value="4">Friday</option>
                  <option value="5">Saturday</option>
                  <option value="6">Sunday</option>
                </select>
              </div>
            </div>
          </div>

          <div className="form-section">
            <h3>Road Conditions</h3>
            <div className="form-row">
              <div className="form-group">
                <label>Number of Lanes</label>
                <input
                  type="number"
                  name="num_lanes"
                  min="1"
                  max="5"
                  value={formData.num_lanes}
                  onChange={handleSingleChange}
                />
              </div>
              <div className="form-group">
                <label>Speed Limit (mph)</label>
                <input
                  type="number"
                  name="speed_limit"
                  min="25"
                  max="75"
                  step="5"
                  value={formData.speed_limit}
                  onChange={handleSingleChange}
                />
              </div>
            </div>
            
            <div className="form-row">
              <div className="form-group">
                <label>Road Capacity</label>
                <input
                  type="number"
                  name="road_capacity"
                  min="500"
                  max="5000"
                  step="100"
                  value={formData.road_capacity}
                  onChange={handleSingleChange}
                />
              </div>
              <div className="form-group">
                <label>Current Vehicles</label>
                <input
                  type="number"
                  name="current_vehicle_count"
                  min="0"
                  max="5000"
                  step="50"
                  value={formData.current_vehicle_count}
                  onChange={handleSingleChange}
                />
              </div>
            </div>
          </div>

          <div className="form-section">
            <h3>Special Conditions</h3>
            <div className="form-group">
              <label>Weather</label>
              <select
                name="weather_condition"
                value={formData.weather_condition}
                onChange={handleSingleChange}
              >
                <option value="0">Clear</option>
                <option value="1">Rain</option>
                <option value="2">Snow</option>
              </select>
            </div>
            
            <div className="checkbox-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  name="is_holiday"
                  checked={formData.is_holiday}
                  onChange={handleSingleChange}
                />
                <span>Holiday</span>
              </label>
              
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  name="road_closure"
                  checked={formData.road_closure}
                  onChange={handleSingleChange}
                />
                <span>Road Closure</span>
              </label>
            </div>
          </div>

          <button type="submit" className="submit-btn">
            🔮 Predict Traffic
          </button>
        </form>
      ) : (
        <form onSubmit={handleCompareSubmit} className="form-content">
          <div className="comparison-container">
            <div className="scenario-column">
              <h3 className="scenario-title">Baseline Scenario</h3>
              <div className="form-group">
                <label>Hour</label>
                <input
                  type="number"
                  name="hour"
                  min="0"
                  max="23"
                  value={baselineData.hour}
                  onChange={handleBaselineChange}
                />
              </div>
              <div className="form-group">
                <label>Lanes</label>
                <input
                  type="number"
                  name="num_lanes"
                  min="1"
                  max="5"
                  value={baselineData.num_lanes}
                  onChange={handleBaselineChange}
                />
              </div>
              <div className="form-group">
                <label>Vehicles</label>
                <input
                  type="number"
                  name="current_vehicle_count"
                  min="0"
                  max="5000"
                  step="50"
                  value={baselineData.current_vehicle_count}
                  onChange={handleBaselineChange}
                />
              </div>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  name="road_closure"
                  checked={baselineData.road_closure}
                  onChange={handleBaselineChange}
                />
                <span>Road Closure</span>
              </label>
            </div>

            <div className="vs-divider">VS</div>

            <div className="scenario-column">
              <h3 className="scenario-title">Modified Scenario</h3>
              <div className="form-group">
                <label>Hour</label>
                <input
                  type="number"
                  name="hour"
                  min="0"
                  max="23"
                  value={modifiedData.hour}
                  onChange={handleModifiedChange}
                />
              </div>
              <div className="form-group">
                <label>Lanes</label>
                <input
                  type="number"
                  name="num_lanes"
                  min="1"
                  max="5"
                  value={modifiedData.num_lanes}
                  onChange={handleModifiedChange}
                />
              </div>
              <div className="form-group">
                <label>Vehicles</label>
                <input
                  type="number"
                  name="current_vehicle_count"
                  min="0"
                  max="5000"
                  step="50"
                  value={modifiedData.current_vehicle_count}
                  onChange={handleModifiedChange}
                />
              </div>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  name="road_closure"
                  checked={modifiedData.road_closure}
                  onChange={handleModifiedChange}
                />
                <span>Road Closure</span>
              </label>
            </div>
          </div>

          <button type="submit" className="submit-btn">
            ⚖️ Compare Scenarios
          </button>
        </form>
      )}
    </div>
  );
}

export default PredictionForm;
