import React, { useState } from 'react';
import './App.css';
import TrafficMap from './components/TrafficMap';
import PredictionForm from './components/PredictionForm';
import ResultsPanel from './components/ResultsPanel';
import Header from './components/Header';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [comparison, setComparison] = useState(null);

  const handlePrediction = (result) => {
    setPrediction(result);
    setComparison(null);
  };

  const handleComparison = (result) => {
    setComparison(result);
    setPrediction(null);
  };

  return (
    <div className="App">
      <Header />
      
      <div className="main-container">
        <div className="left-panel">
          <PredictionForm 
            onPrediction={handlePrediction}
            onComparison={handleComparison}
            setLoading={setLoading}
          />
          
          {(prediction || comparison) && (
            <ResultsPanel 
              prediction={prediction}
              comparison={comparison}
              loading={loading}
            />
          )}
        </div>

        <div className="right-panel">
          <TrafficMap prediction={prediction} />
        </div>
      </div>
    </div>
  );
}

export default App;
