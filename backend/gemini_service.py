"""
Google Gemini API Integration Service
Handles NLU for prompt parsing and text summarization
"""

import os
import json
from typing import Dict, Optional
import google.generativeai as genai
from PIL import Image


class GeminiService:
    """Service class for interacting with Google Gemini API"""
    
    def __init__(self):
        """Initialize Gemini service with API key"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not configured")
        
        genai.configure(api_key=api_key)
        
        # Use Gemini 2.0 Flash for fast responses
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # System prompts
        self.nlu_system_prompt = """You are an expert urban planning assistant. Your task is to analyze a user's request and extract key simulation parameters.

The user will provide a natural language description of a change they want to simulate in a city's traffic network.

You must return ONLY a valid JSON object with the extracted parameters. Do not include any other text.

Possible actions are:
- CLOSE_ROAD: Temporarily or permanently close a road
- ADD_LANE: Add a new lane to a road
- MODIFY_LANE_COUNT: Change the number of lanes
- CHANGE_SPEED_LIMIT: Modify speed limit
- ADD_TRAFFIC_LIGHT: Add a new traffic signal
- REMOVE_TRAFFIC_LIGHT: Remove a traffic signal
- ADD_PUBLIC_TRANSIT: Add bus/train route
- SIMULATE_CONGESTION: Test rush hour scenarios

Return format:
{
  "action": "ACTION_TYPE",
  "parameters": {
    "street_name": "Street Name",
    "additional_params": "value"
  },
  "time_window": {
    "start": "HH:MM",
    "end": "HH:MM"
  },
  "description": "Brief summary of what will be simulated"
}

Examples:

User: "Close Cooper Street"
Output:
{
  "action": "CLOSE_ROAD",
  "parameters": {
    "street_name": "Cooper Street"
  },
  "description": "Simulate closure of Cooper Street"
}

User: "Reduce Division Street to 2 lanes from 7 AM to 10 AM"
Output:
{
  "action": "MODIFY_LANE_COUNT",
  "parameters": {
    "street_name": "Division Street",
    "lane_count": 2
  },
  "time_window": {
    "start": "07:00",
    "end": "10:00"
  },
  "description": "Reduce Division Street to 2 lanes during morning rush hour"
}"""
    
    def parse_prompt(self, user_prompt: str) -> Dict:
        """
        Parse natural language prompt into structured simulation parameters
        
        Args:
            user_prompt: User's natural language description
            
        Returns:
            Dict containing structured parameters for simulation
        """
        try:
            # Create the full prompt
            full_prompt = f"{self.nlu_system_prompt}\n\nUser Prompt: {user_prompt}\n\nOutput:"
            
            # Generate response
            response = self.model.generate_content(full_prompt)
            
            # Extract JSON from response
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON
            parsed_data = json.loads(response_text)
            
            print(f"✓ Gemini parsed prompt: {parsed_data}")
            return parsed_data
            
        except json.JSONDecodeError as e:
            print(f"✗ Failed to parse Gemini response as JSON: {e}")
            print(f"Response was: {response_text}")
            
            # Return a fallback structure
            return {
                "action": "UNKNOWN",
                "parameters": {
                    "raw_prompt": user_prompt
                },
                "description": user_prompt,
                "error": "Failed to parse prompt"
            }
        except Exception as e:
            print(f"✗ Gemini API error: {e}")
            raise
    
    def analyze_satellite_image(self, image_path: str) -> Dict:
        """
        Analyze satellite imagery to extract urban patterns
        
        Args:
            image_path: Path to satellite image
            
        Returns:
            Dict containing analysis of urban patterns
        """
        try:
            img = Image.open(image_path)
            
            prompt = """Analyze this satellite image and extract urban planning information:

1. Road network patterns (grid, radial, organic)
2. Building density zones (high, medium, low)
3. Green spaces and parks
4. Major intersections and traffic nodes
5. Commercial vs residential areas

Return as JSON with coordinates and classifications."""
            
            response = self.model.generate_content([prompt, img])
            
            # Parse response
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            
            return json.loads(response_text)
            
        except Exception as e:
            print(f"✗ Image analysis error: {e}")
            return {"error": str(e)}
    
    def summarize_simulation_results(self, simulation_data: Dict) -> str:
        """
        Generate human-readable summary of simulation results
        
        Args:
            simulation_data: Raw simulation output from MATLAB
            
        Returns:
            str: Natural language summary
        """
        try:
            prompt = f"""Analyze these traffic simulation results and provide a concise, professional summary suitable for urban planners:

Simulation Data:
{json.dumps(simulation_data, indent=2)}

Provide a summary that includes:
1. Overall traffic impact (improvement/degradation)
2. Key metrics (travel time changes, congestion levels)
3. Affected areas
4. Recommendations

Keep the summary conversational and under 150 words, suitable for text-to-speech narration."""
            
            response = self.model.generate_content(prompt)
            summary = response.text.strip()
            
            print(f"✓ Generated summary: {summary[:100]}...")
            return summary
            
        except Exception as e:
            print(f"✗ Summarization error: {e}")
            return f"Simulation completed. Error generating summary: {str(e)}"
    
    def generate_city_layout(self, user_description: str, constraints: Dict) -> Dict:
        """
        Generate procedural city layout from text description
        
        Args:
            user_description: Natural language city description
            constraints: Dict with population, area, budget etc.
            
        Returns:
            Dict containing city layout (roads, buildings, zones)
        """
        try:
            prompt = f"""Generate a realistic city layout based on this description:

Description: {user_description}

Constraints:
- Population: {constraints.get('population', 'medium')}
- Area: {constraints.get('area', '10')} sq km
- Budget: ${constraints.get('budget', '100')}M

Return JSON with:
- Road network (grid of coordinates, road types)
- Building placements (residential, commercial, industrial zones)
- Traffic signal locations
- Public transit routes (if applicable)

Use realistic urban planning principles."""
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            
            return json.loads(response_text)
            
        except Exception as e:
            print(f"✗ City generation error: {e}")
            return {"error": str(e)}
    
    def predict_rural_traffic(
        self,
        latitude: float,
        longitude: float,
        hour: int,
        day_of_week: int,
        is_holiday: bool = False
    ) -> Dict:
        """
        Use Gemini to predict traffic for rural areas where we don't have training data
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            hour: Hour of day (0-23)
            day_of_week: Day of week (0=Monday, 6=Sunday)
            is_holiday: Whether it's a holiday
            
        Returns:
            Dict with congestion_level, travel_time_index, average_speed
        """
    def predict_rural_traffic(
        self,
        latitude: float,
        longitude: float,
        hour: int,
        day_of_week: int,
        is_holiday: bool = False,
        nearest_city: str = None,
        distance_from_city: float = None
    ) -> Dict:
        """
        Use Gemini to predict traffic for any Texas location
        
        Args:
            latitude: Location latitude
            longitude: Location longitude
            hour: Hour of day (0-23)
            day_of_week: Day of week (0=Monday, 6=Sunday)
            is_holiday: Whether it's a holiday
            nearest_city: Name of nearest major city (e.g., "Austin", "Dallas")
            distance_from_city: Distance in km from nearest city
            
        Returns:
            Dict with congestion_level, travel_time_index, average_speed
        """
        try:
            # Map day_of_week to name
            day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            day_name = day_names[day_of_week]
            
            # Determine if weekend
            is_weekend = day_of_week >= 5
            
            # Determine location type and expected congestion range
            if distance_from_city is not None:
                if distance_from_city < 5:
                    location_desc = f"in {nearest_city} city center"
                    congestion_hint = "Urban center - expect moderate to heavy congestion during rush hours (30-70%), light at night (5-20%)"
                elif distance_from_city < 15:
                    location_desc = f"in {nearest_city} suburbs ({distance_from_city:.1f}km from center)"
                    congestion_hint = "Suburban area - expect light to moderate congestion during rush hours (15-40%), very light at night (5-15%)"
                else:
                    location_desc = f"rural area ({distance_from_city:.1f}km from {nearest_city})"
                    congestion_hint = "Rural area - expect very light congestion (5-15%), minimal at night (2-8%)"
            else:
                location_desc = "Texas location"
                congestion_hint = "General Texas area - assess based on time and day"
            
            # Rush hour logic
            is_morning_rush = (hour >= 7 and hour <= 9) and not is_weekend
            is_evening_rush = (hour >= 16 and hour <= 18) and not is_weekend
            is_night = (hour >= 22 or hour <= 5)
            
            prompt = f"""You are a traffic analysis AI. Analyze traffic conditions for this Texas location.

Location: {location_desc} (Lat: {latitude}, Lng: {longitude})
Time: {hour}:00 on a {day_name}
Is Holiday: {is_holiday}
Is Weekend: {is_weekend}
Morning Rush Hour: {is_morning_rush}
Evening Rush Hour: {is_evening_rush}
Late Night/Early Morning: {is_night}

{congestion_hint}

Provide a REALISTIC traffic assessment considering:
1. Time of day is CRITICAL - 2 AM should be VERY low congestion (5-10%), 8 AM weekday should be high (40-70%)
2. Day of week matters - weekends typically 20-40% lower congestion
3. Rush hours (7-9 AM, 4-6 PM weekdays) have highest congestion
4. Late night/early morning (10 PM - 6 AM) has lowest congestion
5. Holidays reduce weekday traffic

Return ONLY a JSON object with these exact fields:
{{
  "congestion_level": 0.45,  // 0.0 to 1.0 (percentage as decimal) - MUST vary by time!
  "travel_time_index": 1.0,  // 1.0 = normal, 2.0 = twice as long
  "average_speed_mph": 45,   // typical speed in mph
  "location_type": "urban",  // urban, suburban, or rural
  "reasoning": "brief explanation mentioning time of day"
}}

IMPORTANT: Adjust congestion based on TIME OF DAY! Don't give the same value for 2 AM and 8 AM!

Return ONLY valid JSON, no other text."""

            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Clean up markdown if present
            if result_text.startswith("```json"):
                result_text = result_text.replace("```json", "").replace("```", "").strip()
            elif result_text.startswith("```"):
                result_text = result_text.replace("```", "").strip()
            
            # Parse JSON
            result = json.loads(result_text)
            
            # Ensure values are in valid ranges
            result["congestion_level"] = max(0.0, min(1.0, float(result.get("congestion_level", 0.05))))
            result["travel_time_index"] = max(1.0, min(3.0, float(result.get("travel_time_index", 1.0))))
            result["average_speed_mph"] = max(5, min(75, int(result.get("average_speed_mph", 55))))
            
            print(f"🤖 Gemini prediction for {location_desc} at {hour}:00: {result['congestion_level']*100:.1f}% congestion - {result.get('reasoning', 'N/A')}")
            
            return result
            
        except Exception as e:
            print(f"⚠️ Gemini rural prediction failed: {e}")
            # Return conservative rural estimates as fallback
            return {
                "congestion_level": 0.05,  # 5% congestion for rural
                "travel_time_index": 1.0,
                "average_speed_mph": 55,
                "location_type": "rural",
                "reasoning": "Default rural estimates (Gemini unavailable)"
            }


# Testing
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    service = GeminiService()
    
    # Test prompt parsing
    test_prompts = [
        "Close I-30 westbound at Cooper Street from 7 AM to 10 AM",
        "Reduce Division Street to 2 lanes",
        "What happens if we add a subway line on Abram Street?"
    ]
    
    for prompt in test_prompts:
        print(f"\nTesting: {prompt}")
        result = service.parse_prompt(prompt)
        print(f"Result: {json.dumps(result, indent=2)}")
