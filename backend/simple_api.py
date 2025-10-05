"""
Simplified API endpoints for professional frontend
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from ml_service import MLPredictionService
from gemini_service import GeminiService
from location_prediction_service import get_location_service

router = APIRouter(prefix="/api", tags=["api"])

# Initialize services
ml_service = MLPredictionService()
gemini_service = GeminiService()
location_service = get_location_service()


class SimplePredictionRequest(BaseModel):
    location: str
    date: str
    time: str


class GeminiInsightRequest(BaseModel):
    location: str
    date: str
    time: str
    prediction: dict


@router.post("/predict/simple")
async def simple_predict(request: SimplePredictionRequest):
    """
    Simplified prediction endpoint - Returns only essential traffic metrics
    Removes: baseline_time, num_vehicles, origin selection
    """
    try:
        # Get location data
        location_data = location_service.get_location_coordinates(request.location)
        if not location_data:
            raise HTTPException(status_code=404, detail="Location not found")
        
        # Parse datetime
        from datetime import datetime
        datetime_str = f"{request.date} {request.time}"
        dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
        
        # Get prediction
        prediction = ml_service.predict(
            location=request.location,
            datetime_obj=dt,
            location_coords=location_data
        )
        
        # Extract only the essential metrics
        simplified_prediction = {
            "congestion_level": prediction.get("congestion_level", 0),
            "travel_time_index": prediction.get("travel_time_index", 0),
            "average_speed": prediction.get("average_speed", 0)
        }
        
        return {
            "success": True,
            "location": request.location,
            "datetime": datetime_str,
            "location_data": location_data,
            "predictions": simplified_prediction,
            "status": {
                "congestion": "Low" if simplified_prediction["congestion_level"] < 4 else "Moderate" if simplified_prediction["congestion_level"] < 7 else "High"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gemini/insights")
async def get_gemini_insights(request: GeminiInsightRequest):
    """
    Get AI-powered insights using Gemini API
    Provides context-aware traffic recommendations
    """
    try:
        congestion = request.prediction.get("predictions", {}).get("congestion_level", 0)
        travel_time = request.prediction.get("predictions", {}).get("travel_time_index", 0)
        speed = request.prediction.get("predictions", {}).get("average_speed", 0)
        
        # Create prompt for Gemini
        prompt = f"""Analyze this traffic prediction and provide concise, actionable insights in 2-3 sentences:

Location: {request.location}
Date: {request.date}
Time: {request.time}
Congestion Level: {congestion}/10
Travel Time: {travel_time} minutes
Average Speed: {speed} mph

Provide:
1. Brief assessment of traffic conditions
2. One specific recommendation for travelers
3. Mention if it's near peak hours or special events

Keep response under 100 words, professional tone."""

        # Get Gemini response
        response = gemini_service.chat(prompt)
        
        return {
            "success": True,
            "insights": response.get("response", "Traffic conditions analyzed. Plan accordingly."),
            "powered_by": "Gemini API"
        }
        
    except Exception as e:
        # Fallback insights if Gemini fails
        congestion = request.prediction.get("predictions", {}).get("congestion_level", 0)
        if congestion < 4:
            fallback = "Traffic conditions are favorable. Expect smooth travel with minimal delays."
        elif congestion < 7:
            fallback = "Moderate traffic expected. Consider allowing extra time for your journey."
        else:
            fallback = "Heavy congestion predicted. Alternative routes or different timing recommended."
        
        return {
            "success": True,
            "insights": fallback,
            "powered_by": "Fallback Logic"
        }
