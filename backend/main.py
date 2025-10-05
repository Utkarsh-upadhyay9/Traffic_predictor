"""
SimCity AI - Main FastAPI Application
Backend API for urban simulation digital twin
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

from auth_service import verify_token
from gemini_service import GeminiService
from agentuity_client import trigger_simulation_workflow
from ml_service import MLPredictionService
from location_prediction_service import get_location_service
from calendar_service import get_calendar_service
from distance_service import get_distance_service
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="SimCity AI API",
    description="Urban simulation and traffic flow digital twin with ML predictions",
    version="1.0.0"
)

# CORS Configuration - Allow file:// URLs and localhost
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
cors_origins.append("null")  # Allow file:// protocol for local HTML files

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2 scheme for Auth0
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize services
gemini_service = GeminiService()
ml_service = MLPredictionService()
location_service = get_location_service()
calendar_service = get_calendar_service()
distance_service = get_distance_service()

# Fetch holidays on startup
print("\n📅 Fetching holidays for next 30 days...")
holidays = calendar_service.fetch_us_holidays(30)
print(f"✅ Loaded {len(holidays)} holidays/events\n")


# Pydantic models
class SimulationRequest(BaseModel):
    """Request model for simulation endpoint"""
    prompt: str
    user_location: Optional[dict] = None
    

class SimulationResponse(BaseModel):
    """Response model for simulation endpoint"""
    status: str
    simulation_id: str
    message: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    services: dict


# Routes
@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "SimCity AI API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify all services"""
    services_status = {
        "auth0": "configured" if os.getenv("AUTH0_DOMAIN") else "not_configured",
        "gemini": "configured" if os.getenv("GEMINI_API_KEY") else "not_configured",
        "agentuity": "configured" if os.getenv("AGENTUITY_API_KEY") else "not_configured",
        "elevenlabs": "configured" if os.getenv("ELEVENLABS_API_KEY") else "not_configured",
        "matlab": "configured" if os.getenv("MATLAB_SERVICE_URL") else "not_configured",
    }
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": services_status
    }


@app.post("/api/simulation", response_model=SimulationResponse)
async def run_simulation(
    request: SimulationRequest,
    token: str = Depends(oauth2_scheme)
):
    """
    Main endpoint to trigger urban traffic simulation
    
    Requires Auth0 JWT token for authentication.
    Triggers Agentuity workflow that orchestrates:
    1. Gemini for NLU (parse prompt)
    2. MATLAB for simulation
    3. Gemini for summarization
    4. ElevenLabs for audio generation
    """
    try:
        # Verify Auth0 token
        payload = verify_token(token)
        user_id = payload.get("sub")
        
        print(f"User {user_id} requested simulation: {request.prompt}")
        
        # Trigger the Agentuity workflow
        simulation_id = await trigger_simulation_workflow(
            prompt=request.prompt,
            user_id=user_id,
            location=request.user_location
        )
        
        return {
            "status": "simulation_started",
            "simulation_id": simulation_id,
            "message": "Your simulation is being processed. Check back shortly for results."
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation failed: {str(e)}"
        )


@app.get("/api/simulation/{simulation_id}")
async def get_simulation_results(
    simulation_id: str,
    token: str = Depends(oauth2_scheme)
):
    """
    Get results of a completed simulation
    """
    try:
        # Verify Auth0 token
        payload = verify_token(token)
        user_id = payload.get("sub")
        
        # TODO: Implement result retrieval from database or Agentuity
        # For now, return placeholder
        return {
            "simulation_id": simulation_id,
            "status": "processing",
            "message": "Results will be available when simulation completes"
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {str(e)}",
        )


@app.get("/api/user/simulations")
async def get_user_simulations(token: str = Depends(oauth2_scheme)):
    """
    Get all simulations for the authenticated user
    """
    try:
        payload = verify_token(token)
        user_id = payload.get("sub")
        
        # TODO: Retrieve from database
        return {
            "user_id": user_id,
            "simulations": []
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {str(e)}",
        )


# ===== ML PREDICTION ENDPOINTS =====

@app.post("/api/predict")
async def predict_traffic(
    hour: Optional[int] = None,
    day_of_week: Optional[int] = None,
    num_lanes: int = 3,
    road_capacity: int = 2000,
    current_vehicle_count: int = 1000,
    weather_condition: int = 0,
    is_holiday: bool = False,
    road_closure: bool = False,
    speed_limit: int = 55
):
    """
    ML-based traffic prediction endpoint
    
    Example:
    POST /api/predict?hour=8&current_vehicle_count=1500&road_closure=true
    """
    try:
        prediction = ml_service.predict_traffic(
            hour=hour,
            day_of_week=day_of_week,
            num_lanes=num_lanes,
            road_capacity=road_capacity,
            current_vehicle_count=current_vehicle_count,
            weather_condition=weather_condition,
            is_holiday=is_holiday,
            road_closure=road_closure,
            speed_limit=speed_limit
        )
        return prediction
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


class ScenarioComparison(BaseModel):
    baseline: dict
    modified: dict


@app.post("/api/compare")
async def compare_scenarios(comparison: ScenarioComparison):
    """
    Compare two traffic scenarios
    
    Example:
    POST /api/compare
    {
      "baseline": {"hour": 8, "road_closure": false},
      "modified": {"hour": 8, "road_closure": true}
    }
    """
    try:
        result = ml_service.compare_scenarios(
            baseline_params=comparison.baseline,
            modified_params=comparison.modified
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison error: {str(e)}"
        )


@app.post("/api/predict-location")
async def predict_by_location(
    dest_latitude: float,
    dest_longitude: float,
    origin_latitude: Optional[float] = None,
    origin_longitude: Optional[float] = None,
    hour: int = 8,
    day_of_week: int = 0,
    date: Optional[str] = None,
    speed_limit: int = 45
):
    """
    Location-based traffic prediction with real travel time from origin to destination
    Uses Google Maps API (or Gemini AI fallback) for accurate travel times
    
    Example:
    POST /api/predict-location?dest_latitude=32.7357&dest_longitude=-97.1081&origin_latitude=32.7500&origin_longitude=-97.1300&hour=8&day_of_week=1&date=2025-12-25&speed_limit=55
    
    Args:
        dest_latitude: Destination latitude
        dest_longitude: Destination longitude
        origin_latitude: Origin latitude (optional, defaults to UT Arlington)
        origin_longitude: Origin longitude (optional)
        hour: Hour of day (0-23)
        day_of_week: Day of week (0=Monday, 6=Sunday)
        date: Optional date in YYYY-MM-DD format (for holiday checking)
    
    Returns:
        Prediction with actual travel time, distance, congestion, vehicle count, holiday info
    """
    try:
        if not location_service.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Location prediction service not ready. Please train models first."
            )
        
        # Use default origin if not provided (UT Arlington)
        if origin_latitude is None or origin_longitude is None:
            origin_latitude = 32.7357
            origin_longitude = -97.1081
        
        # Convert speed_limit from mph to km/h for distance service
        speed_kmh = speed_limit * 1.60934
        
        # Parse date if provided
        date_obj = None
        if date:
            try:
                date_obj = datetime.strptime(date, '%Y-%m-%d')
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid date format. Use YYYY-MM-DD"
                )
        
        # Get actual travel time and distance using Google Maps / Gemini
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
        
        # Get ML prediction for traffic conditions at destination
        ml_prediction = location_service.predict_simple(
            latitude=dest_latitude,
            longitude=dest_longitude,
            hour=hour,
            day_of_week=day_of_week,
            date=date_obj
        )
        
        # Adjust baseline travel time based on predicted congestion
        baseline_time = travel_info["duration_minutes"]
        congestion_multiplier = 1.0 + (ml_prediction["congestion_level"] * 0.5)  # 0-50% increase
        adjusted_time = baseline_time * congestion_multiplier
        
        # Combine results
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
            "model_version": "3.1.0",
        }
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Location prediction error: {str(e)}"
        )


@app.get("/api/location-metadata")
async def get_location_metadata():
    """
    Get location metadata for map initialization
    Returns center point, radius, and major locations
    """
    try:
        metadata = location_service.get_location_info()
        return {
            "success": True,
            "metadata": metadata
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metadata error: {str(e)}"
        )


@app.get("/api/holidays")
async def get_holidays(days_ahead: int = 30):
    """
    Get holidays for the next N days
    
    Example:
    GET /api/holidays?days_ahead=30
    
    Args:
        days_ahead: Number of days to fetch (default 30, max 90)
    
    Returns:
        List of holidays with dates, names, and types
    """
    try:
        # Limit to 90 days maximum
        days_ahead = min(days_ahead, 90)
        
        holidays = calendar_service.fetch_us_holidays(days_ahead)
        
        return {
            "success": True,
            "holidays": holidays,
            "count": len(holidays),
            "days_ahead": days_ahead
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Holiday fetch error: {str(e)}"
        )


@app.get("/api/is-holiday")
async def check_holiday(date: str):
    """
    Check if a specific date is a holiday
    
    Example:
    GET /api/is-holiday?date=2025-12-25
    
    Args:
        date: Date in YYYY-MM-DD format
    
    Returns:
        Boolean indicating if date is a holiday, with holiday name if applicable
    """
    try:
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        is_holiday = calendar_service.is_holiday(date_obj)
        holiday_name = calendar_service.get_holiday_name(date_obj)
        
        return {
            "success": True,
            "date": date,
            "is_holiday": is_holiday,
            "holiday_name": holiday_name,
            "traffic_impact": calendar_service.get_traffic_impact_factor(date_obj, 12)  # Noon as default
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date format. Use YYYY-MM-DD"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Holiday check error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,  # Using 8001 since 8000 is occupied
        reload=False  # Disable reload to prevent issues
    )
