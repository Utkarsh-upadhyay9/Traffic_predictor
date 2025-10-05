"""
Simple backend for testing rural detection with Gemini API
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from datetime import datetime, timedelta
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables FIRST
load_dotenv()

from location_prediction_service import get_location_service
from calendar_service import get_calendar_service
from distance_service import get_distance_service
from gemini_service import GeminiService
from deep_learning_service import DeepLearningPredictionService
import numpy as np

# Initialize FastAPI
app = FastAPI(title="Traffic Predictor - Hybrid Mode")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
location_service = get_location_service()
calendar_service = get_calendar_service()
distance_service = get_distance_service()

# Initialize Deep Learning model (primary)
try:
    dl_service = DeepLearningPredictionService()
    if dl_service.is_ready():
        print("✅ Lightweight Deep Learning model loaded successfully")
    else:
        print("⚠️ Deep Learning model not ready, will use Gemini fallback")
except Exception as e:
    print(f"⚠️ Could not load DL model: {e}")
    dl_service = None

# Initialize Gemini for fallback/rural predictions
try:
    gemini_service = GeminiService()
    print("✅ Gemini API configured as fallback")
except Exception as e:
    print(f"⚠️ Gemini not available: {e}")
    gemini_service = None

# Fetch holidays
print("\n📅 Loading holidays...")
holidays = calendar_service.fetch_us_holidays(30)
print(f"✅ Loaded {len(holidays)} holidays\n")


@app.post("/api/predict-location")
async def predict_by_location(
    dest_latitude: float,
    dest_longitude: float,
    origin_latitude: Optional[float] = None,
    origin_longitude: Optional[float] = None,
    time: Optional[str] = None,  # Format: "HH:MM" or "HH:MM:SS"
    day_of_week: int = 0,
    date: Optional[str] = None,
    speed_limit: int = 45
):
    """Location-based traffic prediction with rural detection"""
    try:
        if not location_service.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Location prediction service not ready"
            )
        
        # Default origin
        if origin_latitude is None or origin_longitude is None:
            origin_latitude = 32.7357
            origin_longitude = -97.1081
        
        # Parse time to extract hour
        hour = 8  # Default
        if time:
            try:
                time_parts = time.split(":")
                hour = int(time_parts[0])
                if hour < 0 or hour > 23:
                    raise ValueError("Hour must be between 0 and 23")
            except (ValueError, IndexError):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid time format. Use HH:MM")
        
        # Parse date
        date_obj = None
        if date:
            try:
                date_obj = datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid date")
        
        # Get travel info
        speed_kmh = speed_limit * 1.60934
        travel_info = distance_service.get_travel_time(
            origin_lat=origin_latitude,
            origin_lng=origin_longitude,
            dest_lat=dest_latitude,
            dest_lng=dest_longitude,
            hour=hour,
            day_of_week=day_of_week,
            departure_time=date_obj,
            avg_speed_kmh=speed_kmh
        )
        
        # Major urban centers in Texas
        urban_centers = {
            "Dallas": (32.7767, -96.7970),
            "Fort Worth": (32.7555, -97.3308),
            "UT Arlington": (32.7357, -97.1081),
            "Austin": (30.2672, -97.7431),
            "Houston": (29.7604, -95.3698),
            "San Antonio": (29.4241, -98.4936),
            "El Paso": (31.7619, -106.4850),
            "Plano": (33.0198, -96.6989),
            "Irving": (32.8140, -96.9489),
            "Lubbock": (33.5779, -101.8552),
            "Garland": (32.9126, -96.6389),
            "McKinney": (33.1972, -96.6397),
            "Frisco": (33.1507, -96.8236),
            "Corpus Christi": (27.8006, -97.3964),
            "Arlington": (32.7357, -97.1081)
        }
        
        # Calculate minimum distance to any major urban center
        distances = []
        for city, (lat, lng) in urban_centers.items():
            dist = np.sqrt((dest_latitude - lat)**2 + (dest_longitude - lng)**2) * 111
            distances.append((city, dist))
        
        # Find nearest city
        nearest_city, distance_from_urban = min(distances, key=lambda x: x[1])
        
        print(f"\n📍 Location: ({dest_latitude}, {dest_longitude})")
        print(f"🏙️  Nearest city: {nearest_city}")
        print(f"📏 Distance from {nearest_city}: {distance_from_urban:.1f}km")
        
        # Use the LIGHTWEIGHT DEEP LEARNING MODEL for predictions
        # Falls back to Gemini only if model fails to load
        use_gemini_for_rural = not dl_service.is_ready() and gemini_service is not None
        
        # Commented out - we now use DL model for all areas:
        # OLD: use_gemini_for_rural = gemini_service is not None
        # OLDER: use_gemini_for_rural = distance_from_urban > 15 and gemini_service is not None
        
        if use_gemini_for_rural:
            # RURAL AREA - Use Gemini API
            print("🤖 Using Gemini API for rural traffic prediction")
            
            is_holiday = location_service.calendar_service.is_holiday(date_obj) if date_obj else False
            
            gemini_result = gemini_service.predict_rural_traffic(
                latitude=dest_latitude,
                longitude=dest_longitude,
                hour=hour,
                day_of_week=day_of_week,
                is_holiday=is_holiday,
                nearest_city=nearest_city,
                distance_from_city=distance_from_urban
            )
            
            ml_prediction = {
                "congestion_level": gemini_result["congestion_level"],
                "travel_time_min": gemini_result.get("travel_time_index", 1.0) * 30,
                "vehicle_count": int(gemini_result["congestion_level"] * 500),  # Lower for rural
                "average_speed_mph": gemini_result["average_speed_mph"],
                "latitude": dest_latitude,
                "longitude": dest_longitude,
                "hour": hour,
                "day_of_week": day_of_week,
                "date": date_obj.strftime('%Y-%m-%d') if date_obj else datetime.now().strftime('%Y-%m-%d'),
                "is_holiday": is_holiday,
                "confidence": "medium",
                "status": "light_traffic" if gemini_result["congestion_level"] < 0.3 else "moderate_traffic",
                "area": gemini_result.get("location_type", "Rural Area").title(),
                "distance_from_campus_km": distance_from_urban,
                "model_type": "gemini_api",
                "model_name": "Gemini 2.0 Flash",
                "prediction_source": f"Gemini API - {gemini_result.get('reasoning', 'N/A')}"
            }
        else:
            # URBAN AREA - Use Deep Learning model
            print("🧠 Using Deep Learning model for urban traffic prediction")
            
            ml_prediction = location_service.predict_simple(
                latitude=dest_latitude,
                longitude=dest_longitude,
                hour=hour,
                day_of_week=day_of_week,
                date=date_obj
            )
        
        # Combine results
        baseline_time = travel_info["duration_minutes"]
        congestion_multiplier = 1.0 + (ml_prediction["congestion_level"] * 0.5)
        adjusted_time = baseline_time * congestion_multiplier
        
        prediction = {
            **ml_prediction,
            "baseline_travel_time_min": round(baseline_time, 1),
            "adjusted_travel_time_min": round(adjusted_time, 1),
            "travel_time_increase_pct": round((congestion_multiplier - 1.0) * 100, 1),
            "distance_km": travel_info["distance_km"],
            "distance_text": travel_info["distance_text"],
            "route_method": travel_info.get("method", "estimated"),
            "origin": {"latitude": origin_latitude, "longitude": origin_longitude},
            "destination": {"latitude": dest_latitude, "longitude": dest_longitude},
            "speed_limit_mph": speed_limit,
            "speed_limit_kmh": round(speed_kmh, 1)
        }
        
        return {
            "success": True,
            "prediction": prediction,
            "model_version": "3.1.1-rural-detection"
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Simple Traffic Predictor Backend")
    print("🌾 Rural area detection: ENABLED\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
