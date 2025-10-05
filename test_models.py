"""
Test Deep Learning Model vs Random Forest
Compare predictions between new PyTorch model and old Random Forest
"""
import sys
sys.path.insert(0, 'backend')

print('=' * 60)
print('TESTING DEEP LEARNING MODEL vs RANDOM FOREST')
print('=' * 60)

# Test location: UT Arlington area
test_lat = 32.7357
test_lon = -97.1081
test_hour = 8  # Rush hour
test_day = 1   # Tuesday

print(f'\nTest Location: {test_lat}, {test_lon}')
print(f'Time: Hour {test_hour}, Day {test_day} (Tuesday)')
print(f'Scenario: Morning rush hour at UT Arlington')

# Test 1: Deep Learning Model
print('\n' + '-' * 60)
print('TEST 1: DEEP LEARNING (PyTorch TrafficNet)')
print('-' * 60)

try:
    from deep_learning_service import get_deep_learning_service
    dl_service = get_deep_learning_service()
    
    if dl_service.is_ready():
        result = dl_service.predict(
            latitude=test_lat,
            longitude=test_lon,
            hour=test_hour,
            day_of_week=test_day,
            is_holiday=False,
            speed_limit=45
        )
        print(f'✅ Model: {result["model_name"]}')
        print(f'📊 Congestion Level: {result["congestion_level"]:.3f} ({result["congestion_category"]})')
        print(f'🚗 Average Speed: {result["average_speed_mph"]:.1f} mph')
        print(f'⏱️  Travel Time Index: {result["travel_time_index"]:.3f}')
        print(f'🕒 Timestamp: {result["timestamp"]}')
    else:
        print('❌ Deep learning model not ready')
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()

# Test 2: Random Forest Model (force fallback)
print('\n' + '-' * 60)
print('TEST 2: RANDOM FOREST (Baseline)')
print('-' * 60)

try:
    from location_prediction_service import LocationPredictionService
    
    # Create instance without deep learning
    import backend.location_prediction_service as lps
    lps.DEEP_LEARNING_AVAILABLE = False
    
    rf_service = LocationPredictionService()
    result = rf_service.predict_simple(
        latitude=test_lat,
        longitude=test_lon,
        hour=test_hour,
        day_of_week=test_day
    )
    print(f'✅ Model: {result.get("model_name", "Random Forest")}')
    print(f'📊 Congestion Level: {result.get("congestion_level", 0):.3f}')
    print(f'⏱️  Travel Time: {result.get("travel_time_min", 0):.1f} min')
    print(f'🚗 Vehicle Count: {result.get("vehicle_count", 0)}')
    print(f'📍 Area: {result.get("area", "Unknown")}')
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()

# Test 3: Via API with Deep Learning
print('\n' + '-' * 60)
print('TEST 3: LOCATION PREDICTION SERVICE (Integrated)')
print('-' * 60)

try:
    from location_prediction_service import get_location_service
    
    service = get_location_service()
    result = service.predict_simple(
        latitude=test_lat,
        longitude=test_lon,
        hour=test_hour,
        day_of_week=test_day
    )
    print(f'✅ Model Type: {result.get("model_type", "unknown")}')
    print(f'✅ Model Name: {result.get("model_name", "unknown")}')
    print(f'📊 Congestion Level: {result.get("congestion_level", 0):.3f}')
    print(f'⏱️  Travel Time: {result.get("travel_time_min", 0):.1f} min')
    if 'average_speed_mph' in result:
        print(f'🚗 Average Speed: {result["average_speed_mph"]:.1f} mph')
    print(f'🗓️  Date: {result.get("date", "N/A")}')
    print(f'🎯 Status: {result.get("status", "N/A")}')
    print(f'🎯 Confidence: {result.get("confidence", "N/A")}')
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()

print('\n' + '=' * 60)
print('COMPARISON COMPLETE')
print('=' * 60)
