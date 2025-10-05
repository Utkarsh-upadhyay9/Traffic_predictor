"""
Location-Based ML Prediction Service
Provides traffic predictions based on latitude, longitude, time
Uses Deep Learning (PyTorch) model by default,         # Replace date's hour with provided hour
        date = date.replace(hour=hour, minute=0, second=0, microsecond=0)
        
        # Check if it's a holidayandom Forest fallback
"""
import joblib
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta
from calendar_service import get_calendar_service

try:
    from deep_learning_service import get_deep_learning_service
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("⚠️  Deep learning service not available, using Random Forest models")

class LocationPredictionService:
    """Service for location-based traffic predictions"""
    
    def __init__(self, models_dir: str = "ml/models"):
        """Initialize the service with trained models"""
        # Handle path whether running from project root or backend directory
        models_path = Path(models_dir)
        if not models_path.exists():
            # Try parent directory if running from backend
            models_path = Path("..") / models_dir
        
        self.models_dir = models_path
        self.models = {}
        self.feature_info = {}
        self.location_metadata = {}
        self.calendar_service = get_calendar_service()
        
        # Try to initialize deep learning service
        self.use_deep_learning = False
        self.deep_learning_service = None
        
        if DEEP_LEARNING_AVAILABLE:
            try:
                self.deep_learning_service = get_deep_learning_service()
                if self.deep_learning_service.is_ready():
                    self.use_deep_learning = True
                    print("✅ Using Deep Learning (PyTorch) model for predictions")
                else:
                    print("⚠️  Deep learning model not ready, falling back to Random Forest")
            except Exception as e:
                print(f"⚠️  Could not initialize deep learning: {e}")
        
        # Load Random Forest models as fallback
        if not self.use_deep_learning:
            self._load_models()
        
        self._load_feature_info()
        self._load_location_metadata()
    
    def _load_models(self):
        """Load all trained models"""
        model_files = {
            'congestion_simple': 'congestion_simple_location_model.pkl',
            'travel_time_simple': 'travel_time_simple_location_model.pkl',
            'vehicle_count_simple': 'vehicle_count_simple_location_model.pkl',
        }
        
        for model_name, filename in model_files.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)
                print(f"✅ Loaded {model_name}")
            else:
                print(f"⚠️  Model not found: {filename}")
    
    def _load_feature_info(self):
        """Load feature configuration"""
        feature_path = self.models_dir / "location_features.json"
        if feature_path.exists():
            with open(feature_path, 'r') as f:
                self.feature_info = json.load(f)
            print(f"✅ Loaded feature info (version {self.feature_info.get('version', 'unknown')})")
    
    def _load_location_metadata(self):
        """Load location metadata"""
        metadata_path = self.models_dir.parent / "location_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.location_metadata = json.load(f)
            print(f"✅ Loaded location metadata")
    
    def _calculate_location_features(self, lat: float, lon: float) -> Dict:
        """Calculate location-based features"""
        # UT Arlington center
        uta_center = self.location_metadata.get('center', {'lat': 32.7357, 'lon': -97.1081})
        
        # Distance from campus
        lat_diff = lat - uta_center['lat']
        lon_diff = lon - uta_center['lon']
        distance_from_campus = np.sqrt(lat_diff**2 + lon_diff**2) * 111  # Convert to km
        
        # Determine location type based on distance and position
        if distance_from_campus < 1:
            location_type_encoded = 2  # campus
        elif distance_from_campus < 2:
            location_type_encoded = 1  # commercial
        elif distance_from_campus < 3:
            location_type_encoded = 4  # major_intersection
        else:
            location_type_encoded = 0  # residential
        
        return {
            'distance_from_campus_km': distance_from_campus,
            'location_type_encoded': location_type_encoded
        }
    
    def predict_simple(
        self,
        latitude: float,
        longitude: float,
        hour: int,
        day_of_week: int,
        date: Optional[datetime] = None,
    ) -> Dict:
        """
        Make prediction using only location and time
        Uses Deep Learning model if available, otherwise Random Forest
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            hour: Hour of day (0-23)
            day_of_week: Day of week (0=Monday, 6=Sunday)
            date: Optional date for holiday checking (defaults to today)
        
        Returns:
            Dictionary with predictions
        """
        # Validate inputs
        if not (-90 <= latitude <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        if not (-180 <= longitude <= 180):
            raise ValueError("Longitude must be between -180 and 180")
        if not (0 <= hour <= 23):
            raise ValueError("Hour must be between 0 and 23")
        if not (0 <= day_of_week <= 6):
            raise ValueError("Day of week must be between 0 and 6")
        
        # Use provided date or calculate from today
        if date is None:
            # Calculate date based on day_of_week offset from today
            today = datetime.now()
            days_ahead = (day_of_week - today.weekday()) % 7
            date = today + timedelta(days=days_ahead)
        
        # Replace date's hour with provided hour
        date = date.replace(hour=hour, minute=0, second=0, microsecond=0)
        
        # Check if it's a holiday
        is_holiday = self.calendar_service.is_holiday(date)
        holiday_name = self.calendar_service.get_holiday_name(date)
        traffic_factor = self.calendar_service.get_traffic_impact_factor(date, hour)
        
        # GOOGLE MAPS-STYLE TRAFFIC PATTERNS
        # Based on real-world urban traffic observations
        base_congestion = 0.15  # Base 15% congestion at all times in urban areas
        
        if day_of_week < 5:  # Monday-Friday (Weekdays)
            if hour >= 6 and hour <= 10:  # Morning rush (6 AM - 10 AM)
                # Google Maps shows RED (heavy) during peak morning
                if hour == 7 or hour == 8:  # Peak morning rush
                    base_congestion = 0.75  # 75% congestion (RED on Google Maps)
                elif hour == 6 or hour == 9:  # Building up / winding down
                    base_congestion = 0.55  # 55% congestion (ORANGE-RED)
                else:  # hour == 10
                    base_congestion = 0.40  # 40% congestion (ORANGE)
            
            elif hour >= 15 and hour <= 19:  # Evening rush (3 PM - 7 PM)
                # Google Maps shows DARK RED (heaviest) during evening peak
                if hour >= 16 and hour <= 18:  # Peak evening rush (4-6 PM)
                    base_congestion = 0.85  # 85% congestion (DARK RED)
                elif hour == 15 or hour == 19:  # Building up / winding down
                    base_congestion = 0.60  # 60% congestion (ORANGE-RED)
            
            elif hour >= 11 and hour <= 14:  # Lunch/Midday (11 AM - 2 PM)
                base_congestion = 0.35  # 35% congestion (YELLOW-ORANGE)
            
            elif hour >= 20 and hour <= 22:  # Late evening
                base_congestion = 0.30  # 30% congestion (YELLOW)
            
            elif hour >= 23 or hour <= 5:  # Late night/Early morning
                base_congestion = 0.10  # 10% congestion (GREEN)
        
        else:  # Saturday-Sunday (Weekends)
            if hour >= 10 and hour <= 20:  # Weekend daytime
                base_congestion = 0.35  # 35% congestion (lighter than weekdays)
            elif hour >= 21 or hour <= 9:  # Weekend night/morning
                base_congestion = 0.15  # 15% congestion
        
        # Holidays have even lighter traffic (like Sunday)
        if is_holiday:
            base_congestion *= 0.6  # 40% reduction on holidays
        
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
            dist = np.sqrt((latitude - lat)**2 + (longitude - lng)**2) * 111
            distances.append((city, dist))
        
        # Find nearest city
        nearest_city, distance_from_urban = min(distances, key=lambda x: x[1])
        
        # Apply rural area adjustment based on distance from nearest major city
        is_rural = distance_from_urban > 50
        rural_factor = 1.0
        if distance_from_urban > 50:
            rural_factor = 0.05  # Rural areas have 5% of urban congestion
        elif distance_from_urban > 30:
            rural_factor = 0.3   # Suburban areas have 30% of urban congestion
        elif distance_from_urban > 15:
            rural_factor = 0.6   # Near-urban areas have 60% of urban congestion
        
        print(f"📍 Location: {distance_from_urban:.1f}km from {nearest_city} (rural_factor: {rural_factor})")
        print(f"🔍 DEBUG: hour={hour}, day={day_of_week}, is_holiday={is_holiday}, traffic_factor={traffic_factor:.2f}")
        
        # Use Deep Learning model if available
        if self.use_deep_learning and self.deep_learning_service:
            try:
                dl_predictions = self.deep_learning_service.predict(
                    latitude=latitude,
                    longitude=longitude,
                    hour=hour,
                    day_of_week=day_of_week,
                    is_holiday=is_holiday,
                    speed_limit=45,
                    date=date
                )
                print(f"🧠 DEBUG: DL raw output: {dl_predictions}")
                
                # DEBUG: Show what DL model returned
                print(f"  📊 DL Model returned: congestion={dl_predictions['congestion_level']:.3f}")
                print(f"  📐 Google Maps Pattern: base_congestion={base_congestion:.2f}, rural_factor={rural_factor:.2f}")
                
                # Use Google Maps-style base congestion (override model for realism)
                # Only use model as minor adjustment factor
                model_adjustment = (dl_predictions['congestion_level'] - 0.5) * 0.2  # ±10% from model
                congestion_level = base_congestion + model_adjustment
                congestion_level = float(np.clip(congestion_level * rural_factor, 0, 1))
                
                print(f"  ✅ Final congestion: {congestion_level:.3f} (Google Maps-style pattern)")
                
                # Log the adjustment for debugging
                if rural_factor < 1.0:
                    print(f"  🌾 Rural adjustment applied: {dl_predictions['congestion_level']:.2f} → {congestion_level:.2f}")
                
                # Calculate derived metrics
                predictions = {
                    'congestion_level': congestion_level,
                    'travel_time_min': dl_predictions.get('travel_time_index', 1.0) * 30,  # Base 30 min
                    'vehicle_count': int(congestion_level * 2000),  # Estimated
                    'average_speed_mph': dl_predictions.get('average_speed_mph', 35),
                    'model_type': dl_predictions.get('model_type', 'deep_learning'),
                    'model_name': dl_predictions.get('model_name', 'LightweightTrafficNet')
                }
                
            except Exception as e:
                print(f"⚠️  Deep learning prediction failed: {e}, falling back to Random Forest")
                # Fall through to Random Forest
                self.use_deep_learning = False
        
        # Use Random Forest models (fallback or primary)
        if not self.use_deep_learning:
            # Prepare features
            features = {
                'latitude': latitude,
                'longitude': longitude,
                'hour': hour,
                'day_of_week': day_of_week,
            }
            
            # Add engineered features
            features['is_weekend'] = 1 if day_of_week >= 5 else 0
            features['is_rush_hour'] = 1 if (6 <= hour <= 9) or (15 <= hour <= 18) else 0
            features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            
            # Create feature array (must match training order)
            simple_features = self.feature_info.get('simple_features', [
                'latitude', 'longitude', 'hour', 'day_of_week',
                'is_weekend', 'is_rush_hour', 'hour_sin', 'hour_cos'
            ])
            
            X = np.array([[features[f] for f in simple_features]])
            
            # Make predictions
            predictions = {'model_type': 'random_forest', 'model_name': 'Random Forest Ensemble'}
            
            if 'congestion_simple' in self.models:
                model_congestion = self.models['congestion_simple'].predict(X)[0]
                # Use Google Maps-style base congestion (override model for realism)
                # Only use model as minor adjustment factor
                model_adjustment = (model_congestion - 0.5) * 0.2  # ±10% from model
                congestion = base_congestion + model_adjustment
                predictions['congestion_level'] = float(np.clip(congestion * rural_factor, 0, 1))
                
                # Log the adjustment for debugging
                if rural_factor < 1.0:
                    print(f"  🌾 Rural adjustment applied: {congestion:.2f} → {predictions['congestion_level']:.2f}")
            
            if 'travel_time_simple' in self.models:
                travel_time = self.models['travel_time_simple'].predict(X)[0]
                # Apply holiday traffic factor (inverse for travel time)
                if traffic_factor < 1.0:
                    travel_time = travel_time * (2 - traffic_factor)  # Less traffic = faster
                else:
                    travel_time = travel_time * traffic_factor  # More traffic = slower
                predictions['travel_time_min'] = float(max(1, travel_time))
            
            if 'vehicle_count_simple' in self.models:
                vehicle_count = self.models['vehicle_count_simple'].predict(X)[0]
                # Apply holiday traffic factor
                vehicle_count = vehicle_count * traffic_factor
                predictions['vehicle_count'] = int(max(0, vehicle_count))
        
        # Add metadata
        predictions['latitude'] = latitude
        predictions['longitude'] = longitude
        predictions['hour'] = hour
        predictions['day_of_week'] = day_of_week
        predictions['date'] = date.strftime('%Y-%m-%d')
        predictions['is_holiday'] = is_holiday
        
        if holiday_name:
            predictions['holiday_name'] = holiday_name
            predictions['traffic_factor'] = round(traffic_factor, 2)
        
        # Determine confidence based on congestion level
        congestion_level = predictions.get('congestion_level', 0.5)
        if congestion_level < 0.3:
            predictions['confidence'] = 'high'
            predictions['status'] = 'free_flow'
        elif congestion_level < 0.7:
            predictions['confidence'] = 'medium'
            predictions['status'] = 'moderate_traffic'
        else:
            predictions['confidence'] = 'high'
            predictions['status'] = 'heavy_congestion'
        
        # Add location description
        location_features = self._calculate_location_features(latitude, longitude)
        distance_km = location_features['distance_from_campus_km']
        
        if distance_km < 1:
            predictions['area'] = 'Campus Area'
        elif distance_km < 2:
            predictions['area'] = 'Near Campus'
        elif distance_km < 3:
            predictions['area'] = 'City Center'
        else:
            predictions['area'] = 'Surrounding Area'
        
        predictions['distance_from_campus_km'] = round(distance_km, 2)
        
        return predictions
    
    def get_location_info(self) -> Dict:
        """Get location metadata for map initialization"""
        return self.location_metadata
    
    def is_ready(self) -> bool:
        """Check if service is ready"""
        # If using deep learning, check DL service
        if self.use_deep_learning and self.deep_learning_service:
            return self.deep_learning_service.is_ready()
        # Otherwise check Random Forest models
        required_models = ['congestion_simple', 'travel_time_simple', 'vehicle_count_simple']
        return all(model in self.models for model in required_models)

# Global service instance
_service = None

def get_location_service() -> LocationPredictionService:
    """Get or create location prediction service"""
    global _service
    if _service is None:
        _service = LocationPredictionService()
    return _service
