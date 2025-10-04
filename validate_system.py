"""Quick validation test for full application"""
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("SimCity AI - System Validation Test")
print("=" * 60)
print()

# Test 1: ML Model
print("Test 1: ML Model Loading...")
try:
    from ml.traffic_model import TrafficPredictor
    predictor = TrafficPredictor()
    if predictor.load_model():
        print("  ✅ ML models loaded successfully")
        
        # Test prediction
        result = predictor.predict({
            'hour_of_day': 8,
            'day_of_week': 1,
            'num_lanes': 3,
            'road_capacity': 2000,
            'current_vehicle_count': 1500,
            'weather_condition': 0,
            'is_holiday': 0,
            'road_closure': 1,
            'speed_limit': 55
        })
        print(f"  ✅ Prediction successful")
        print(f"     Travel Time: {result['predicted_travel_time_min']:.1f} min")
        print(f"     Congestion: {result['predicted_congestion_level']:.1%}")
    else:
        print("  ❌ Models not found - run: python ml/traffic_model.py")
except Exception as e:
    print(f"  ❌ Error: {e}")

print()

# Test 2: Backend Imports
print("Test 2: Backend Services...")
try:
    from backend.ml_service import MLPredictionService
    print("  ✅ ML Service imported")
    
    from backend.gemini_service import GeminiService
    print("  ✅ Gemini Service imported")
    
    from backend.auth_service import verify_token
    print("  ✅ Auth Service imported")
    
    from backend.matlab_service import MATLABSimulationService
    print("  ✅ MATLAB Service imported")
except Exception as e:
    print(f"  ❌ Error: {e}")

print()

# Test 3: Check Files
print("Test 3: File System...")
import os

checks = [
    ("ML models", "ml/models/travel_time_model.pkl"),
    ("Backend main", "backend/main.py"),
    ("Frontend package", "frontend/package.json"),
    ("Frontend app", "frontend/src/App.js"),
    ("Startup script", "start-app.ps1"),
]

for name, path in checks:
    if os.path.exists(path):
        print(f"  ✅ {name} exists")
    else:
        print(f"  ❌ {name} missing: {path}")

print()
print("=" * 60)
print("✅ VALIDATION COMPLETE")
print("=" * 60)
print()
print("🚀 To start the application:")
print("   .\\start-app.ps1")
print()
print("📚 For documentation:")
print("   See FULL_APP_GUIDE.md")
