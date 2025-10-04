"""
ML Service - Traffic Prediction
Integrates trained ML model into the backend API
"""

import sys
import os

# Add parent directory to path to import ml module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from typing import Dict, Optional
from datetime import datetime

# Import the predictor
try:
    from ml.traffic_model import TrafficPredictor
except ImportError:
    print("⚠️  Could not import ml.traffic_model")
    print("   Make sure ml/ directory exists and traffic_model.py is present")
    TrafficPredictor = None


class MLPredictionService:
    """Service for ML-based traffic predictions"""
    
    def __init__(self):
        """Initialize and load trained models"""
        if TrafficPredictor is None:
            print("⚠️  TrafficPredictor not available")
            self.predictor = None
            self.model_loaded = False
            return
            
        self.predictor = TrafficPredictor()
        self.model_loaded = self.predictor.load_model()
        
        if not self.model_loaded:
            print("⚠️  ML models not found. Train models first:")
            print("   python ml/traffic_model.py")
    
    def predict_traffic(
        self,
        hour: Optional[int] = None,
        day_of_week: Optional[int] = None,
        num_lanes: int = 3,
        road_capacity: int = 2000,
        current_vehicle_count: int = 1000,
        weather_condition: int = 0,
        is_holiday: bool = False,
        road_closure: bool = False,
        speed_limit: int = 55
    ) -> Dict:
        """
        Predict traffic conditions using ML model
        
        Args:
            hour: Hour of day (0-23), defaults to current hour
            day_of_week: 0=Monday, 6=Sunday, defaults to current day
            num_lanes: Number of lanes (1-5)
            road_capacity: Vehicles per hour capacity
            current_vehicle_count: Current number of vehicles
            weather_condition: 0=clear, 1=rain, 2=snow
            is_holiday: Boolean for holiday
            road_closure: Boolean for road closure
            speed_limit: Speed limit in mph
            
        Returns:
            Dict with predictions
        """
        
        if not self.model_loaded:
            return self._mock_prediction()
        
        # Use current time if not provided
        now = datetime.now()
        if hour is None:
            hour = now.hour
        if day_of_week is None:
            day_of_week = now.weekday()
        
        # Prepare features
        features = {
            'hour_of_day': hour,
            'day_of_week': day_of_week,
            'num_lanes': num_lanes,
            'road_capacity': road_capacity,
            'current_vehicle_count': current_vehicle_count,
            'weather_condition': weather_condition,
            'is_holiday': 1 if is_holiday else 0,
            'road_closure': 1 if road_closure else 0,
            'speed_limit': speed_limit
        }
        
        # Make prediction
        prediction = self.predictor.predict(features)
        
        # Add context
        prediction['input_features'] = features
        prediction['timestamp'] = now.isoformat()
        prediction['model_version'] = '1.0.0'
        
        return prediction
    
    def compare_scenarios(
        self,
        baseline_params: Dict,
        modified_params: Dict
    ) -> Dict:
        """
        Compare two scenarios (e.g., before/after road closure)
        
        Args:
            baseline_params: Normal conditions
            modified_params: Modified conditions
            
        Returns:
            Dict with comparison metrics
        """
        
        baseline_pred = self.predict_traffic(**baseline_params)
        modified_pred = self.predict_traffic(**modified_params)
        
        # Calculate differences
        travel_time_diff = modified_pred['predicted_travel_time_min'] - baseline_pred['predicted_travel_time_min']
        congestion_diff = modified_pred['predicted_congestion_level'] - baseline_pred['predicted_congestion_level']
        vehicle_diff = modified_pred['predicted_vehicle_count'] - baseline_pred['predicted_vehicle_count']
        
        # Calculate percentage changes
        travel_time_pct = (travel_time_diff / baseline_pred['predicted_travel_time_min']) * 100
        congestion_pct = (congestion_diff / max(baseline_pred['predicted_congestion_level'], 0.01)) * 100
        vehicle_pct = (vehicle_diff / baseline_pred['predicted_vehicle_count']) * 100
        
        return {
            'baseline': baseline_pred,
            'modified': modified_pred,
            'changes': {
                'travel_time_diff_min': round(travel_time_diff, 2),
                'travel_time_change_pct': round(travel_time_pct, 1),
                'congestion_diff': round(congestion_diff, 3),
                'congestion_change_pct': round(congestion_pct, 1),
                'vehicle_count_diff': int(vehicle_diff),
                'vehicle_count_change_pct': round(vehicle_pct, 1)
            },
            'recommendation': self._generate_recommendation(travel_time_pct, congestion_pct)
        }
    
    def _generate_recommendation(self, travel_time_pct: float, congestion_pct: float) -> str:
        """Generate human-readable recommendation"""
        if travel_time_pct > 30:
            return "⚠️ Significant delay expected. Consider alternative routes or public transit."
        elif travel_time_pct > 15:
            return "⚠️ Moderate delay expected. Allow extra travel time."
        elif travel_time_pct > 5:
            return "✓ Minor delay expected. Minimal impact on travel."
        elif travel_time_pct < -10:
            return "✓ Improved travel conditions. Faster than normal."
        else:
            return "✓ Normal traffic conditions. No significant changes."
    
    def _mock_prediction(self) -> Dict:
        """Fallback mock prediction if models not loaded"""
        return {
            'predicted_travel_time_min': 15.0,
            'predicted_congestion_level': 0.6,
            'predicted_vehicle_count': 1000,
            'confidence': 'low',
            'note': 'Mock prediction - ML models not loaded'
        }


# Standalone testing
if __name__ == "__main__":
    service = MLPredictionService()
    
    print("\n=== Testing ML Prediction Service ===\n")
    
    # Test single prediction
    print("Test 1: Morning Rush Hour")
    pred = service.predict_traffic(
        hour=8,
        day_of_week=1,
        num_lanes=3,
        current_vehicle_count=1500
    )
    print(f"  Travel Time: {pred['predicted_travel_time_min']} min")
    print(f"  Congestion: {pred['predicted_congestion_level']:.1%}")
    print(f"  Vehicles: {pred['predicted_vehicle_count']}")
    
    # Test scenario comparison
    print("\nTest 2: Road Closure Impact")
    baseline = {
        'hour': 17,
        'num_lanes': 3,
        'current_vehicle_count': 1500,
        'road_closure': False
    }
    
    modified = {
        'hour': 17,
        'num_lanes': 3,
        'current_vehicle_count': 1500,
        'road_closure': True
    }
    
    comparison = service.compare_scenarios(baseline, modified)
    print(f"  Travel Time Change: {comparison['changes']['travel_time_change_pct']:+.1f}%")
    print(f"  Congestion Change: {comparison['changes']['congestion_change_pct']:+.1f}%")
    print(f"  Recommendation: {comparison['recommendation']}")
    
    print("\n✓ ML Service tests complete!")
