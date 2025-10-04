import React, { useState, useRef, useEffect } from 'react';
import Map, { Marker, Source, Layer } from 'react-map-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import './TrafficMap.css';

// Use your Mapbox token
const MAPBOX_TOKEN = process.env.REACT_APP_MAPBOX_TOKEN || 'pk.eyJ1IjoidXRrYXJzOTUiLCJhIjoiY21nY3U3Njg1MHZnZTJscHVheThyNjdidiJ9.fmQIpz87lspbbIhk1lfQLQ';

function TrafficMap({ prediction }) {
  const [viewport, setViewport] = useState({
    latitude: 32.7299,
    longitude: -97.1161,
    zoom: 13
  });

  const mapRef = useRef();

  // Generate heatmap data based on prediction
  const generateHeatmapData = () => {
    if (!prediction) return null;

    const congestionLevel = prediction.predicted_congestion_level || 0.5;
    
    // Generate points around UT Arlington
    const points = [];
    for (let i = 0; i < 50; i++) {
      const latOffset = (Math.random() - 0.5) * 0.03;
      const lngOffset = (Math.random() - 0.5) * 0.03;
      
      points.push({
        type: 'Feature',
        properties: {
          intensity: congestionLevel + (Math.random() - 0.5) * 0.3
        },
        geometry: {
          type: 'Point',
          coordinates: [
            -97.1161 + lngOffset,
            32.7299 + latOffset
          ]
        }
      });
    }

    return {
      type: 'FeatureCollection',
      features: points
    };
  };

  const heatmapData = generateHeatmapData();

  const heatmapLayer = {
    id: 'traffic-heat',
    type: 'heatmap',
    paint: {
      'heatmap-weight': ['get', 'intensity'],
      'heatmap-intensity': 1,
      'heatmap-color': [
        'interpolate',
        ['linear'],
        ['heatmap-density'],
        0, 'rgba(33,102,172,0)',
        0.2, 'rgb(103,169,207)',
        0.4, 'rgb(209,229,240)',
        0.6, 'rgb(253,219,199)',
        0.8, 'rgb(239,138,98)',
        1, 'rgb(178,24,43)'
      ],
      'heatmap-radius': 30,
      'heatmap-opacity': 0.7
    }
  };

  return (
    <div className="traffic-map">
      <Map
        ref={mapRef}
        {...viewport}
        onMove={evt => setViewport(evt.viewState)}
        style={{ width: '100%', height: '100%' }}
        mapStyle="mapbox://styles/mapbox/dark-v10"
        mapboxAccessToken={MAPBOX_TOKEN}
      >
        {/* UT Arlington Marker */}
        <Marker longitude={-97.1161} latitude={32.7299} anchor="bottom">
          <div className="map-marker">
            <div className="marker-icon">🏛️</div>
            <div className="marker-label">UT Arlington</div>
          </div>
        </Marker>

        {/* Heatmap Layer */}
        {heatmapData && (
          <Source type="geojson" data={heatmapData}>
            <Layer {...heatmapLayer} />
          </Source>
        )}
      </Map>

      {/* Map Legend */}
      <div className="map-legend">
        <h4>Traffic Density</h4>
        <div className="legend-scale">
          <div className="legend-item">
            <div className="legend-color" style={{ background: 'rgb(103,169,207)' }}></div>
            <span>Low</span>
          </div>
          <div className="legend-item">
            <div className="legend-color" style={{ background: 'rgb(253,219,199)' }}></div>
            <span>Medium</span>
          </div>
          <div className="legend-item">
            <div className="legend-color" style={{ background: 'rgb(178,24,43)' }}></div>
            <span>High</span>
          </div>
        </div>
      </div>

      {/* Map Info */}
      {prediction && (
        <div className="map-info">
          <div className="info-item">
            <span className="info-label">Current Congestion:</span>
            <span className="info-value">
              {(prediction.predicted_congestion_level * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

export default TrafficMap;
