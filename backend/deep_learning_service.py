"""
Deep Learning Prediction Service
Uses PyTorch neural network model for traffic predictions
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

class TrafficNet(nn.Module):
    """Neural network for traffic prediction"""
    def __init__(self, input_size=10, hidden_sizes=[128, 64, 32], output_size=3):
        super(TrafficNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class DeepLearningPredictionService:
    """Service for deep learning-based traffic predictions"""
    
    def __init__(self, models_dir: str = "ml/models"):
        """Initialize the service with trained model"""
        models_path = Path(models_dir)
        if not models_path.exists():
            models_path = Path("..") / models_dir
        
        self.models_dir = models_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._load_model()
    
    def _load_model(self):
        """Load the trained PyTorch model"""
        model_path = self.models_dir / "deep_traffic_best.pth"
        
        if not model_path.exists():
            print(f"⚠️  Deep learning model not found: {model_path}")
            return
        
        try:
            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Determine model architecture from checkpoint
            if isinstance(checkpoint, dict):
                # Extract model parameters
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                
                # Infer input/output sizes from state dict
                first_layer_key = list(state_dict.keys())[0]
                if 'network.0.weight' in state_dict:
                    input_size = state_dict['network.0.weight'].shape[1]
                    hidden_sizes = []
                    
                    # Find hidden layer sizes
                    layer_idx = 0
                    while f'network.{layer_idx}.weight' in state_dict:
                        if layer_idx > 0:
                            hidden_sizes.append(state_dict[f'network.{layer_idx}.weight'].shape[0])
                        layer_idx += 3  # Skip ReLU and Dropout
                    
                    # Get output size from last layer
                    last_layer_keys = [k for k in state_dict.keys() if k.startswith('network.')]
                    last_layer_idx = max([int(k.split('.')[1]) for k in last_layer_keys if k.split('.')[1].isdigit()])
                    output_size = state_dict[f'network.{last_layer_idx}.weight'].shape[0]
                    
                    # Create model
                    self.model = TrafficNet(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size)
                    self.model.load_state_dict(state_dict)
                else:
                    # Try default architecture
                    self.model = TrafficNet()
                    self.model.load_state_dict(state_dict)
            else:
                # Direct state dict
                self.model = TrafficNet()
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Loaded deep learning model on {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading deep learning model: {e}")
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
        speed_limit: int = 45
    ) -> torch.Tensor:
        """
        Prepare input features for the model
        
        Features:
        - latitude
        - longitude
        - hour (normalized 0-23)
        - day_of_week (normalized 0-6)
        - is_weekend (binary)
        - is_holiday (binary)
        - hour_sin, hour_cos (cyclic encoding)
        - speed_limit (normalized)
        - distance_from_center
        """
        # UT Arlington center
        center_lat, center_lon = 32.7357, -97.1081
        
        # Calculate distance from center
        distance = np.sqrt((latitude - center_lat)**2 + (longitude - center_lon)**2) * 111  # km
        
        # Cyclic encoding for hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        
        # Is weekend
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Normalize features
        features = np.array([
            latitude / 90.0,  # Normalize latitude
            longitude / 180.0,  # Normalize longitude
            hour / 24.0,
            day_of_week / 7.0,
            is_weekend,
            1 if is_holiday else 0,
            hour_sin,
            hour_cos,
            speed_limit / 100.0,  # Normalize speed limit
            distance / 50.0  # Normalize distance (assume max 50km)
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
        
        # Interpret predictions
        # Assuming output: [congestion_level, travel_time_index, average_speed]
        congestion_level = float(predictions[0])
        travel_time_index = float(predictions[1]) if len(predictions) > 1 else congestion_level
        average_speed = float(predictions[2]) if len(predictions) > 2 else speed_limit * (1 - congestion_level)
        
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
            "model_name": "TrafficNet",
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
