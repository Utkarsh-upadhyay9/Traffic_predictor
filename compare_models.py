"""
Comprehensive Model Comparison: Deep Learning vs Random Forest
Testing multiple scenarios across different times and locations
"""
import sys
sys.path.insert(0, 'backend')

from deep_learning_service import get_deep_learning_service
from location_prediction_service import LocationPredictionService
import backend.location_prediction_service as lps

print('=' * 80)
print('COMPREHENSIVE MODEL COMPARISON: DEEP LEARNING vs RANDOM FOREST')
print('=' * 80)

# Test scenarios
scenarios = [
    {"name": "Morning Rush - UT Arlington", "lat": 32.7357, "lon": -97.1081, "hour": 8, "day": 1},
    {"name": "Midday - UT Arlington", "lat": 32.7357, "lon": -97.1081, "hour": 12, "day": 1},
    {"name": "Evening Rush - Downtown", "lat": 32.75, "lon": -97.13, "hour": 17, "day": 3},
    {"name": "Late Night - Campus", "lat": 32.7357, "lon": -97.1081, "hour": 23, "day": 4},
    {"name": "Weekend Morning - Arlington", "lat": 32.72, "lon": -97.10, "hour": 9, "day": 5},
]

dl_service = get_deep_learning_service()

# Force Random Forest for comparison
lps.DEEP_LEARNING_AVAILABLE = False
rf_service = LocationPredictionService()

print(f'\n{"Scenario":<30} {"Model":<15} {"Congestion":<12} {"Travel Time":<12} {"Speed":<10}')
print('=' * 80)

for scenario in scenarios:
    name = scenario["name"]
    lat = scenario["lat"]
    lon = scenario["lon"]
    hour = scenario["hour"]
    day = scenario["day"]
    
    # Deep Learning prediction
    try:
        dl_result = dl_service.predict(
            latitude=lat,
            longitude=lon,
            hour=hour,
            day_of_week=day,
            is_holiday=False,
            speed_limit=45
        )
        print(f'{name:<30} {"DL (PyTorch)":<15} {dl_result["congestion_level"]:.3f} ({dl_result["congestion_category"]:<6}) {dl_result["travel_time_index"]*30:.1f} min       {dl_result["average_speed_mph"]:.1f} mph')
    except Exception as e:
        print(f'{name:<30} {"DL (PyTorch)":<15} ERROR: {e}')
    
    # Random Forest prediction
    try:
        rf_result = rf_service.predict_simple(
            latitude=lat,
            longitude=lon,
            hour=hour,
            day_of_week=day
        )
        travel_time = rf_result.get("travel_time_min", 0)
        congestion = rf_result.get("congestion_level", 0)
        category = "high" if congestion > 0.7 else ("medium" if congestion > 0.3 else "low")
        print(f'{"":<30} {"RF (Baseline)":<15} {congestion:.3f} ({category:<6}) {travel_time:.1f} min       -')
    except Exception as e:
        print(f'{"":<30} {"RF (Baseline)":<15} ERROR: {e}')
    
    print()

print('=' * 80)
print('SUMMARY')
print('=' * 80)
print('✅ Deep Learning Model: TrafficNet (PyTorch)')
print('   - Architecture: 14 inputs → 2048 → 2048 → 2048 → 2048 → 3 outputs')
print('   - Model Size: 48 MB')
print('   - Outputs: Congestion level, Travel time index, Average speed')
print()
print('✅ Random Forest Model: Gradient Boosting Ensemble')
print('   - 3 separate models (congestion, travel time, vehicle count)')
print('   - Feature engineering: hour_sin/cos, weekend, rush hour')
print()
print('🎯 Key Differences:')
print('   - DL model provides direct speed estimates')
print('   - DL model handles complex non-linear patterns')
print('   - RF models are faster to load and run')
print('   - Both models integrate holiday and time-of-day factors')
print('=' * 80)
