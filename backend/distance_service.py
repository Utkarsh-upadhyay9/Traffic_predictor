"""
Distance and Route Service
Calculates accurate travel time and distance using Google Maps Distance Matrix API
With Gemini AI fallback for estimation
"""

import os
import requests
from typing import Dict, Optional, Tuple
from datetime import datetime
import json
import google.generativeai as genai

class DistanceService:
    """Service to calculate distances and travel times between locations"""
    
    def __init__(self):
        """Initialize with Google Maps API key and Gemini fallback"""
        self.google_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-pro')
        else:
            self.gemini_model = None
        
        # Default origin (can be overridden)
        self.default_origin = {"lat": 32.7357, "lng": -97.1081}  # UT Arlington
        
        print(f"🗺️  Distance Service initialized")
        print(f"   Google Maps API: {'✓' if self.google_api_key else '✗ (using estimation)'}")
        print(f"   Gemini Fallback: {'✓' if self.gemini_model else '✗'}")
    
    def get_travel_time_google(
        self, 
        origin_lat: float, 
        origin_lng: float, 
        dest_lat: float, 
        dest_lng: float,
        mode: str = "driving",
        departure_time: Optional[datetime] = None
    ) -> Dict:
        """
        Get travel time using Google Maps Distance Matrix API
        
        Args:
            origin_lat: Origin latitude
            origin_lng: Origin longitude
            dest_lat: Destination latitude
            dest_lng: Destination longitude
            mode: Travel mode (driving, walking, bicycling, transit)
            departure_time: When to depart (for traffic predictions)
        
        Returns:
            Dict with distance_km, duration_minutes, duration_in_traffic_minutes
        """
        if not self.google_api_key:
            return None
        
        url = "https://maps.googleapis.com/maps/api/distancematrix/json"
        
        params = {
            "origins": f"{origin_lat},{origin_lng}",
            "destinations": f"{dest_lat},{dest_lng}",
            "mode": mode,
            "key": self.google_api_key,
            "units": "metric"
        }
        
        # Add departure time for traffic-aware routing
        if departure_time:
            timestamp = int(departure_time.timestamp())
            params["departure_time"] = timestamp
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data["status"] != "OK":
                print(f"⚠️  Google Maps API error: {data['status']}")
                return None
            
            element = data["rows"][0]["elements"][0]
            
            if element["status"] != "OK":
                print(f"⚠️  Route not found: {element['status']}")
                return None
            
            result = {
                "distance_km": element["distance"]["value"] / 1000,  # meters to km
                "distance_text": element["distance"]["text"],
                "duration_minutes": element["duration"]["value"] / 60,  # seconds to minutes
                "duration_text": element["duration"]["text"],
                "origin": {"lat": origin_lat, "lng": origin_lng},
                "destination": {"lat": dest_lat, "lng": dest_lng}
            }
            
            # Add traffic duration if available
            if "duration_in_traffic" in element:
                result["duration_in_traffic_minutes"] = element["duration_in_traffic"]["value"] / 60
                result["duration_in_traffic_text"] = element["duration_in_traffic"]["text"]
            
            return result
            
        except Exception as e:
            print(f"❌ Google Maps API error: {e}")
            return None
    
    def estimate_travel_time_haversine(
        self,
        origin_lat: float,
        origin_lng: float,
        dest_lat: float,
        dest_lng: float,
        avg_speed_kmh: float = 50.0
    ) -> Dict:
        """
        Estimate travel time using Haversine distance formula
        Fallback when Google Maps API is not available
        
        Args:
            origin_lat: Origin latitude
            origin_lng: Origin longitude
            dest_lat: Destination latitude
            dest_lng: Destination longitude
            avg_speed_kmh: Average speed in km/h (default 50 km/h for city driving)
        
        Returns:
            Dict with distance_km and estimated duration_minutes
        """
        from math import radians, sin, cos, sqrt, atan2
        
        # Earth radius in kilometers
        R = 6371.0
        
        # Convert to radians
        lat1 = radians(origin_lat)
        lon1 = radians(origin_lng)
        lat2 = radians(dest_lat)
        lon2 = radians(dest_lng)
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        distance_km = R * c
        
        # Add 30% for city routing (not straight line)
        distance_km *= 1.3
        
        # Calculate duration
        duration_minutes = (distance_km / avg_speed_kmh) * 60
        
        return {
            "distance_km": round(distance_km, 2),
            "distance_text": f"{distance_km:.1f} km",
            "duration_minutes": round(duration_minutes, 1),
            "duration_text": f"{int(duration_minutes)} mins",
            "origin": {"lat": origin_lat, "lng": origin_lng},
            "destination": {"lat": dest_lat, "lng": dest_lng},
            "method": "haversine_estimate"
        }
    
    def estimate_travel_time_gemini(
        self,
        origin_lat: float,
        origin_lng: float,
        dest_lat: float,
        dest_lng: float,
        hour: int,
        day_of_week: int
    ) -> Dict:
        """
        Use Gemini AI to estimate travel time based on location context
        
        Args:
            origin_lat, origin_lng: Origin coordinates
            dest_lat, dest_lng: Destination coordinates
            hour: Hour of day (0-23)
            day_of_week: Day of week (0=Monday, 6=Sunday)
        
        Returns:
            Dict with estimated travel time and distance
        """
        if not self.gemini_model:
            return None
        
        prompt = f"""Given these coordinates in the Dallas-Fort Worth area:
- Origin: ({origin_lat}, {origin_lng})
- Destination: ({dest_lat}, {dest_lng})
- Time: Hour {hour}, Day {day_of_week} (0=Monday)

Estimate the driving travel time and distance. Consider:
1. Typical traffic patterns for this time
2. Urban vs highway routes
3. Rush hour effects if applicable

Respond in JSON format:
{{
    "distance_km": <number>,
    "duration_minutes": <number>,
    "reasoning": "<brief explanation>"
}}"""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            text = response.text.strip()
            
            # Extract JSON from response
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            
            data = json.loads(text)
            
            return {
                "distance_km": data.get("distance_km", 10.0),
                "distance_text": f"{data.get('distance_km', 10):.1f} km",
                "duration_minutes": data.get("duration_minutes", 20.0),
                "duration_text": f"{int(data.get('duration_minutes', 20))} mins",
                "origin": {"lat": origin_lat, "lng": origin_lng},
                "destination": {"lat": dest_lat, "lng": dest_lng},
                "method": "gemini_ai_estimate",
                "reasoning": data.get("reasoning", "AI estimation")
            }
            
        except Exception as e:
            print(f"❌ Gemini estimation error: {e}")
            return None
    
    def get_travel_time(
        self,
        origin_lat: float,
        origin_lng: float,
        dest_lat: float,
        dest_lng: float,
        hour: int,
        day_of_week: int,
        departure_time: Optional[datetime] = None,
        avg_speed_kmh: float = 72.0
    ) -> Dict:
        """
        Get travel time using best available method:
        1. Try Google Maps API (most accurate)
        2. Try Gemini AI estimation (context-aware)
        3. Fallback to Haversine formula (basic estimation)
        
        Args:
            origin_lat, origin_lng: Origin coordinates
            dest_lat, dest_lng: Destination coordinates
            hour: Hour of day (0-23)
            day_of_week: Day of week (0=Monday, 6=Sunday)
            departure_time: When to depart (for traffic predictions)
            avg_speed_kmh: Average speed in km/h (default 72.0 = 45 mph)
        
        Returns:
            Dict with distance and travel time information
        """
        # Try Google Maps first
        result = self.get_travel_time_google(
            origin_lat, origin_lng, dest_lat, dest_lng,
            departure_time=departure_time
        )
        
        if result:
            result["method"] = "google_maps"
            print(f"✓ Google Maps: {result['duration_text']} ({result['distance_text']})")
            return result
        
        # Try Gemini AI
        result = self.estimate_travel_time_gemini(
            origin_lat, origin_lng, dest_lat, dest_lng, hour, day_of_week
        )
        
        if result:
            print(f"✓ Gemini AI: {result['duration_text']} ({result['distance_text']})")
            return result
        
        # Fallback to Haversine (uses user-specified speed)
        result = self.estimate_travel_time_haversine(
            origin_lat, origin_lng, dest_lat, dest_lng,
            avg_speed_kmh=avg_speed_kmh
        )
        print(f"✓ Haversine ({avg_speed_kmh:.0f} km/h): {result['duration_text']} ({result['distance_text']})")
        return result


# Singleton instance
_distance_service = None

def get_distance_service() -> DistanceService:
    """Get or create distance service singleton"""
    global _distance_service
    if _distance_service is None:
        _distance_service = DistanceService()
    return _distance_service
