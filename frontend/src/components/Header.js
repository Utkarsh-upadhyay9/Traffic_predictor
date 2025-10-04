import React from 'react';
import './Header.css';

function Header() {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo">
          <span className="logo-icon">🏙️</span>
          <h1>SimCity AI</h1>
        </div>
        <p className="tagline">AI-Powered Urban Traffic Simulation</p>
        <div className="tech-badges">
          <span className="badge">Gemini AI</span>
          <span className="badge">ML Predictions</span>
          <span className="badge">Real-time</span>
        </div>
      </div>
    </header>
  );
}

export default Header;
