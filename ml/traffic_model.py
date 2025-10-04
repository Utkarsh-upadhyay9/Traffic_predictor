"""
Lightweight Traffic Prediction Model
Uses scikit-learn RandomForest for fast training and inference
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, List, Tuple
from datetime import datetime


class TrafficPredictor:
    """
    Lightweight ML model for traffic prediction
    Predicts: travel time, congestion level, vehicle count
    """
    
    def __init__(self):
        self.model_travel_time = None
        self.model_congestion = None
        self.model_vehicle_count = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_dir = "ml/models"
        
    def generate_training_data(self, n_samples: int = 5000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic training data for traffic patterns
        
        Features:
        - hour_of_day (0-23)
        - day_of_week (0-6, Mon=0)
        - num_lanes (1-5)
        - road_capacity (vehicles/hour)
        - current_vehicle_count
        - weather_condition (0=clear, 1=rain, 2=snow)
        - is_holiday (0 or 1)
        - road_closure (0 or 1)
        - speed_limit (mph)
        
        Targets:
        - travel_time_min
        - congestion_level (0-1)
        - predicted_vehicle_count
        """
        
        print(f"Generating {n_samples} training samples...")
        
        np.random.seed(42)
        
        # Generate features
        data = {
            'hour_of_day': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'num_lanes': np.random.randint(1, 6, n_samples),
            'road_capacity': np.random.randint(800, 3000, n_samples),
            'current_vehicle_count': np.random.randint(50, 2500, n_samples),
            'weather_condition': np.random.choice([0, 0, 0, 1, 2], n_samples),  # Mostly clear
            'is_holiday': np.random.choice([0, 0, 0, 0, 1], n_samples),  # 20% holidays
            'road_closure': np.random.choice([0, 0, 0, 0, 1], n_samples),  # 20% closures
            'speed_limit': np.random.choice([25, 35, 45, 55, 65], n_samples)
        }
        
        X = pd.DataFrame(data)
        
        # Generate realistic target values based on features
        # Travel time increases with traffic, decreases with more lanes
        base_travel_time = 10  # minutes
        travel_time = base_travel_time + (
            (X['current_vehicle_count'] / X['road_capacity']) * 20 +  # Congestion effect
            (X['num_lanes'] * -2) +  # More lanes = faster
            (X['weather_condition'] * 3) +  # Bad weather = slower
            (X['road_closure'] * 15) +  # Closure = much slower
            np.where((X['hour_of_day'] >= 7) & (X['hour_of_day'] <= 9), 5, 0) +  # Morning rush
            np.where((X['hour_of_day'] >= 16) & (X['hour_of_day'] <= 18), 5, 0) +  # Evening rush
            np.random.randn(n_samples) * 2  # Random noise
        )
        travel_time = np.clip(travel_time, 5, 60)  # Reasonable bounds
        
        # Congestion level (0-1)
        congestion = (X['current_vehicle_count'] / X['road_capacity']) * (1 + X['road_closure'] * 0.5)
        congestion = np.clip(congestion + np.random.randn(n_samples) * 0.1, 0, 1)
        
        # Predicted vehicle count (for next hour)
        vehicle_count = X['current_vehicle_count'] * (
            1 + np.where((X['hour_of_day'] >= 7) & (X['hour_of_day'] <= 9), 0.3, 0) +
            np.where((X['hour_of_day'] >= 16) & (X['hour_of_day'] <= 18), 0.3, 0) -
            (X['is_holiday'] * 0.2) +
            np.random.randn(n_samples) * 0.1
        )
        vehicle_count = np.clip(vehicle_count, 10, 3000)
        
        y = pd.DataFrame({
            'travel_time_min': travel_time,
            'congestion_level': congestion,
            'predicted_vehicle_count': vehicle_count
        })
        
        print(f"✓ Generated {n_samples} samples")
        print(f"  Feature shape: {X.shape}")
        print(f"  Target shape: {y.shape}")
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.2):
        """
        Train the traffic prediction models
        """
        print("\n=== Training Traffic Prediction Models ===\n")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples\n")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model for travel time
        print("Training Travel Time model...")
        self.model_travel_time = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model_travel_time.fit(X_train_scaled, y_train['travel_time_min'])
        score_tt = self.model_travel_time.score(X_test_scaled, y_test['travel_time_min'])
        print(f"  ✓ R² Score: {score_tt:.3f}")
        
        # Train model for congestion
        print("Training Congestion model...")
        self.model_congestion = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model_congestion.fit(X_train_scaled, y_train['congestion_level'])
        score_cong = self.model_congestion.score(X_test_scaled, y_test['congestion_level'])
        print(f"  ✓ R² Score: {score_cong:.3f}")
        
        # Train model for vehicle count
        print("Training Vehicle Count model...")
        self.model_vehicle_count = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model_vehicle_count.fit(X_train_scaled, y_train['predicted_vehicle_count'])
        score_vc = self.model_vehicle_count.score(X_test_scaled, y_test['predicted_vehicle_count'])
        print(f"  ✓ R² Score: {score_vc:.3f}")
        
        self.is_trained = True
        
        # Feature importance
        feature_names = X.columns
        importances = self.model_travel_time.feature_importances_
        print("\n📊 Top 5 Feature Importances (Travel Time):")
        for idx in np.argsort(importances)[-5:][::-1]:
            print(f"  {feature_names[idx]}: {importances[idx]:.3f}")
        
        print(f"\n✓ All models trained successfully!")
        
    def predict(self, features: Dict) -> Dict:
        """
        Make predictions for given traffic conditions
        
        Args:
            features: Dict with keys matching training features
            
        Returns:
            Dict with predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Call train() first or load a saved model.")
        
        # Convert to DataFrame
        X_input = pd.DataFrame([features])
        X_scaled = self.scaler.transform(X_input)
        
        # Make predictions
        travel_time = float(self.model_travel_time.predict(X_scaled)[0])
        congestion = float(self.model_congestion.predict(X_scaled)[0])
        vehicle_count = int(self.model_vehicle_count.predict(X_scaled)[0])
        
        return {
            'predicted_travel_time_min': round(travel_time, 2),
            'predicted_congestion_level': round(congestion, 3),
            'predicted_vehicle_count': vehicle_count,
            'confidence': 'high' if congestion < 0.7 else 'medium'
        }
    
    def save_model(self):
        """Save trained models to disk"""
        if not self.is_trained:
            raise ValueError("No trained model to save!")
        
        os.makedirs(self.model_dir, exist_ok=True)
        
        joblib.dump(self.model_travel_time, f"{self.model_dir}/travel_time_model.pkl")
        joblib.dump(self.model_congestion, f"{self.model_dir}/congestion_model.pkl")
        joblib.dump(self.model_vehicle_count, f"{self.model_dir}/vehicle_count_model.pkl")
        joblib.dump(self.scaler, f"{self.model_dir}/scaler.pkl")
        
        print(f"\n✓ Models saved to {self.model_dir}/")
        
    def load_model(self):
        """Load trained models from disk"""
        try:
            self.model_travel_time = joblib.load(f"{self.model_dir}/travel_time_model.pkl")
            self.model_congestion = joblib.load(f"{self.model_dir}/congestion_model.pkl")
            self.model_vehicle_count = joblib.load(f"{self.model_dir}/vehicle_count_model.pkl")
            self.scaler = joblib.load(f"{self.model_dir}/scaler.pkl")
            self.is_trained = True
            print(f"✓ Models loaded from {self.model_dir}/")
            return True
        except FileNotFoundError:
            print(f"⚠️  No saved models found in {self.model_dir}/")
            return False


def main():
    """Train and save the traffic prediction model"""
    print("=" * 60)
    print("SimCity AI - Lightweight Traffic Model Training")
    print("=" * 60)
    print()
    
    # Initialize predictor
    predictor = TrafficPredictor()
    
    # Generate training data
    X, y = predictor.generate_training_data(n_samples=5000)
    
    # Train models
    predictor.train(X, y)
    
    # Save models
    predictor.save_model()
    
    # Test prediction
    print("\n" + "=" * 60)
    print("Testing Predictions")
    print("=" * 60)
    
    test_scenarios = [
        {
            'name': 'Morning Rush Hour - Clear Weather',
            'features': {
                'hour_of_day': 8,
                'day_of_week': 1,  # Tuesday
                'num_lanes': 3,
                'road_capacity': 2000,
                'current_vehicle_count': 1500,
                'weather_condition': 0,
                'is_holiday': 0,
                'road_closure': 0,
                'speed_limit': 55
            }
        },
        {
            'name': 'Evening Rush - Road Closure',
            'features': {
                'hour_of_day': 17,
                'day_of_week': 3,  # Thursday
                'num_lanes': 2,
                'road_capacity': 1500,
                'current_vehicle_count': 1800,
                'weather_condition': 0,
                'is_holiday': 0,
                'road_closure': 1,  # CLOSED!
                'speed_limit': 45
            }
        },
        {
            'name': 'Night Time - Low Traffic',
            'features': {
                'hour_of_day': 2,
                'day_of_week': 5,
                'num_lanes': 4,
                'road_capacity': 2500,
                'current_vehicle_count': 200,
                'weather_condition': 0,
                'is_holiday': 0,
                'road_closure': 0,
                'speed_limit': 65
            }
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']}:")
        prediction = predictor.predict(scenario['features'])
        print(f"  Travel Time: {prediction['predicted_travel_time_min']:.1f} minutes")
        print(f"  Congestion: {prediction['predicted_congestion_level']:.1%}")
        print(f"  Vehicle Count: {prediction['predicted_vehicle_count']} vehicles")
        print(f"  Confidence: {prediction['confidence']}")
    
    print("\n" + "=" * 60)
    print("✓ Training Complete! Models ready for deployment.")
    print("=" * 60)


if __name__ == "__main__":
    main()
