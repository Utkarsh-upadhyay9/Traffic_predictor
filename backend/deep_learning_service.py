"""
Deep Learning Prediction Service
Uses PyTorch neural network model for traffic predictions
"""
# Try to import PyTorch - gracefully handle if not installed
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️  PyTorch not installed - deep learning predictions will be disabled")
    print("   App will use temporal traffic patterns and Random Forest models instead")
    TORCH_AVAILABLE = False
    torch = None
    nn = None

import numpy as np
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

class TrafficNet(nn.Module):
    """Neural network for traffic prediction"""
    def __init__(self, input_size=14, hidden_sizes=[2048, 2048, 2048], output_size=3):
        super(TrafficNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class DeepLearningPredictionService:
    """Service for deep learning-based traffic predictions"""
    
    def __init__(self, models_dir: str = "ml/models"):
        """Initialize the service with trained model"""
        # Check if PyTorch is available
        if not TORCH_AVAILABLE:
            print("⚠️  PyTorch not available - skipping deep learning model loading")
            self.model = None
            self.device = None
            return
            
        models_path = Path(models_dir)
        if not models_path.exists():
            models_path = Path("..") / models_dir
        
        self.models_dir = models_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained PyTorch model"""
        if not TORCH_AVAILABLE:
            return
            
        # Try lightweight model first (new, properly trained)
        lightweight_path = self.models_dir / "lightweight_traffic_model.pth"
        old_model_path = self.models_dir / "deep_traffic_best.pth"
        
        print(f"🔍 Looking for models in: {self.models_dir.absolute()}")
        print(f"   Lightweight model exists: {lightweight_path.exists()}")
        print(f"   Old model exists: {old_model_path.exists()}")
        
        model_path = lightweight_path if lightweight_path.exists() else old_model_path
        
        print(f"📦 Loading model from: {model_path}")
        
        if not model_path.exists():
            print(f"⚠️  Deep learning model not found: {model_path}")
            print(f"🔧 Run 'python ml/train_lightweight_model.py' to train a new model")
            return
        
        try:
            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model state dict and metadata
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                input_size = checkpoint.get('input_size', 8)
                hidden_size = checkpoint.get('hidden_size', 128)
                output_size = checkpoint.get('output_size', 3)
                
                print(f"📊 Lightweight model: input={input_size}, hidden={hidden_size}, output={output_size}")
                
                # Use LightweightTrafficNet architecture
                from torch import nn
                
                class LightweightTrafficNet(nn.Module):
                    def __init__(self, input_size=8, hidden_size=128, output_size=3):
                        super(LightweightTrafficNet, self).__init__()
                        self.net = nn.Sequential(
                            nn.Linear(input_size, hidden_size),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(hidden_size, hidden_size),
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(hidden_size, hidden_size // 2),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Linear(hidden_size // 2, output_size)
                        )
                        self.sigmoid = nn.Sigmoid()
                        self.relu = nn.ReLU()
                    
                    def forward(self, x):
                        out = self.net(x)
                        congestion = self.sigmoid(out[:, 0:1])
                        travel_time = 1.0 + self.relu(out[:, 1:2]) * 2.0
                        speed = 5.0 + self.relu(out[:, 2:3]) * 70.0
                        return torch.cat([congestion, travel_time, speed], dim=1)
                
                self.model = LightweightTrafficNet(input_size, hidden_size, output_size)
                self.model.load_state_dict(state_dict)
                self.model_type = "lightweight"
                
            else:
                # Fallback to old model format
                state_dict = checkpoint
                input_size = state_dict['net.0.weight'].shape[1]
                hidden_sizes = []
                layer_idx = 0
                while f'net.{layer_idx}.weight' in state_dict:
                    layer_shape = state_dict[f'net.{layer_idx}.weight'].shape
                    hidden_sizes.append(layer_shape[0])
                    layer_idx += 2
                
                output_size = hidden_sizes[-1]
                hidden_sizes = hidden_sizes[:-1]
                
                print(f"📊 Old model: input={input_size}, hidden={hidden_sizes}, output={output_size}")
                
                self.model = TrafficNet(
                    input_size=input_size,
                    hidden_sizes=hidden_sizes,
                    output_size=output_size
                )
                self.model.load_state_dict(state_dict)
                self.model_type = "legacy"
            
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Loaded {self.model_type} model on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading deep learning model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def is_ready(self) -> bool:
        """Check if the model is loaded and ready"""
        return self.model is not None
    
    def prepare_features(
        self,
        latitude: float,
        longitude: float,
        hour: int,
        day_of_week: int,
        is_holiday: bool = False,
        speed_limit: int = 45,
        distance_from_center: float = None,
        population_density: float = 1.0
    ) -> torch.Tensor:
        """
        Prepare input features for the lightweight model (8 features)
        
        Features for lightweight model:
        1. latitude
        2. longitude
        3. hour (0-23)
        4. day_of_week (0-6)
        5. is_weekend (0 or 1)
        6. is_holiday (0 or 1)
        7. distance_from_center (km)
        8. population_density (relative scale)
        """
        # Calculate distance from center if not provided
        if distance_from_center is None:
            # UT Arlington center
            center_lat, center_lon = 32.7357, -97.1081
            distance_from_center = np.sqrt((latitude - center_lat)**2 + (longitude - center_lon)**2) * 111  # km
        
        # Prepare features
        is_weekend = 1 if day_of_week >= 5 else 0
        
        features = [
            latitude,
            longitude,
            float(hour),
            float(day_of_week),
            float(is_weekend),
            float(is_holiday),
            distance_from_center,
            population_density
        ]
        
        return torch.FloatTensor(features).unsqueeze(0)
        
        # Binary features
        is_weekend = 1 if day_of_week >= 5 else 0
        is_rush_hour = 1 if (6 <= hour <= 9) or (16 <= hour <= 19) else 0
        is_morning = 1 if 6 <= hour < 12 else 0
        is_afternoon = 1 if 12 <= hour < 18 else 0
        is_evening = 1 if 18 <= hour < 24 else 0
        
        # Normalize features
        features = np.array([
            latitude / 90.0,  # 1. Normalize latitude
            longitude / 180.0,  # 2. Normalize longitude
            hour / 24.0,  # 3. Normalize hour
            day_of_week / 7.0,  # 4. Normalize day_of_week
            is_weekend,  # 5. Binary: is weekend
            1 if is_holiday else 0,  # 6. Binary: is holiday
            hour_sin,  # 7. Cyclic hour (sin)
            hour_cos,  # 8. Cyclic hour (cos)
            speed_limit / 100.0,  # 9. Normalize speed limit
            distance / 50.0,  # 10. Normalize distance (max 50km)
            is_rush_hour,  # 11. Binary: is rush hour
            is_morning,  # 12. Binary: is morning
            is_afternoon,  # 13. Binary: is afternoon
            is_evening  # 14. Binary: is evening
        ], dtype=np.float32)
        
        return torch.from_numpy(features).unsqueeze(0).to(self.device)
    
    def predict(
        self,
        latitude: float,
        longitude: float,
        hour: int,
        day_of_week: int,
        is_holiday: bool = False,
        speed_limit: int = 45,
        date: Optional[datetime] = None
    ) -> Dict:
        """
        Make prediction using deep learning model
        
        Returns:
            dict with congestion_level, travel_time_index, average_speed
        """
        if not self.is_ready():
            raise RuntimeError("Deep learning model not loaded")
        
        # Prepare features
        features = self.prepare_features(
            latitude=latitude,
            longitude=longitude,
            hour=hour,
            day_of_week=day_of_week,
            is_holiday=is_holiday,
            speed_limit=speed_limit
        )
        
        # Make prediction
        with torch.no_grad():
            output = self.model(features)
            predictions = output.cpu().numpy()[0]
        
        # DEBUG: Log raw model output
        print(f"🧠 DL Model Raw Output: {predictions}")
        print(f"   Input features: {features.cpu().numpy()[0]}")
        print(f"   Parsed: hour={hour}, day={day_of_week}, weekend={day_of_week>=5}, holiday={is_holiday}, lat={latitude:.4f}, lng={longitude:.4f}")
        
        # Interpret predictions
        # Assuming output: [congestion_level, travel_time_index, average_speed]
        congestion_level = float(predictions[0])
        travel_time_index = float(predictions[1]) if len(predictions) > 1 else congestion_level
        average_speed = float(predictions[2]) if len(predictions) > 2 else speed_limit * (1 - congestion_level)
        
        print(f"   Parsed: congestion={congestion_level:.3f}, travel_time={travel_time_index:.3f}, speed={average_speed:.1f}")
        
        # Ensure values are in reasonable ranges
        congestion_level = np.clip(congestion_level, 0, 1)
        travel_time_index = np.clip(travel_time_index, 0, 2)
        average_speed = np.clip(average_speed, 5, speed_limit)
        
        return {
            "congestion_level": round(congestion_level, 3),
            "travel_time_index": round(travel_time_index, 3),
            "average_speed_mph": round(average_speed, 1),
            "congestion_category": self._get_congestion_category(congestion_level),
            "model_type": "deep_learning",
            "model_name": "LightweightTrafficNet (PyTorch)" if self.model_type == "lightweight" else "TrafficNet (PyTorch)",
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_congestion_category(self, level: float) -> str:
        """Convert congestion level to category"""
        if level < 0.3:
            return "low"
        elif level < 0.6:
            return "medium"
        else:
            return "high"


# Singleton instance
_deep_learning_service = None

def get_deep_learning_service() -> DeepLearningPredictionService:
    """Get the singleton deep learning service instance"""
    global _deep_learning_service
    if _deep_learning_service is None:
        _deep_learning_service = DeepLearningPredictionService()
    return _deep_learning_service
