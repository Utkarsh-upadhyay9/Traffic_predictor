"""
Train a lightweight deep learning model for traffic prediction
This model can be trained on CPU and learns time-based patterns
Uses a simple feed-forward neural network with proper training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class TrafficDataset(Dataset):
    """Generate synthetic traffic data based on realistic patterns"""
    
    def __init__(self, num_samples=10000):
        self.num_samples = num_samples
        self.data = []
        
        print("📊 Generating synthetic traffic data...")
        
        # Major Texas cities with realistic traffic patterns
        cities = [
            {"name": "Dallas", "lat": 32.7767, "lng": -96.7970, "population": 1.3},
            {"name": "Austin", "lat": 30.2672, "lng": -97.7431, "population": 1.0},
            {"name": "Houston", "lat": 29.7604, "lng": -95.3698, "population": 2.3},
            {"name": "San Antonio", "lat": 29.4241, "lng": -98.4936, "population": 1.5},
        ]
        
        for _ in range(num_samples):
            # Random city
            city = random.choice(cities)
            
            # Add some spatial variation around city center
            lat = city["lat"] + random.gauss(0, 0.1)
            lng = city["lng"] + random.gauss(0, 0.1)
            distance_from_center = abs(random.gauss(0, 5))  # km
            
            # Random time
            hour = random.randint(0, 23)
            day_of_week = random.randint(0, 6)
            is_weekend = day_of_week >= 5
            is_holiday = random.random() < 0.05
            
            # Calculate realistic congestion based on time patterns
            base_congestion = self._calculate_realistic_congestion(
                hour, day_of_week, is_weekend, is_holiday, 
                city["population"], distance_from_center
            )
            
            # Add some noise
            congestion = np.clip(base_congestion + random.gauss(0, 0.05), 0, 1)
            
            # Travel time index (1.0 = normal, higher = slower)
            travel_time_index = 1.0 + congestion * 1.5
            
            # Average speed (inversely related to congestion)
            base_speed = 60 if distance_from_center > 3 else 45
            avg_speed = base_speed * (1 - congestion * 0.6)
            
            self.data.append({
                'features': [lat, lng, hour, day_of_week, int(is_weekend), 
                           int(is_holiday), distance_from_center, city["population"]],
                'targets': [congestion, travel_time_index, avg_speed]
            })
    
    def _calculate_realistic_congestion(self, hour, day_of_week, is_weekend, 
                                       is_holiday, population, distance):
        """Calculate realistic congestion based on known patterns"""
        
        # Base congestion increases with population
        base = 0.15 * population
        
        # Distance from center reduces congestion
        distance_factor = max(0.3, 1.0 - distance / 20)
        
        # Time of day patterns (most important!)
        if 7 <= hour <= 9 or 16 <= hour <= 18:  # Rush hours
            time_factor = 1.8
        elif 10 <= hour <= 15:  # Midday
            time_factor = 1.2
        elif 22 <= hour or hour <= 5:  # Night
            time_factor = 0.2
        else:  # Other times
            time_factor = 0.8
        
        # Weekend reduction
        if is_weekend:
            time_factor *= 0.6
        
        # Holiday reduction (weekday traffic)
        if is_holiday and not is_weekend:
            time_factor *= 0.5
        
        congestion = base * distance_factor * time_factor
        return np.clip(congestion, 0.02, 0.95)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.FloatTensor(item['features'])
        targets = torch.FloatTensor(item['targets'])
        return features, targets


class LightweightTrafficNet(nn.Module):
    """Lightweight neural network for traffic prediction"""
    
    def __init__(self, input_size=8, hidden_size=128, output_size=3):
        super(LightweightTrafficNet, self).__init__()
        
        # Simple but effective architecture
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
        
        # Output activation for each prediction
        self.sigmoid = nn.Sigmoid()  # For congestion (0-1)
        self.relu = nn.ReLU()  # For travel time and speed (positive)
    
    def forward(self, x):
        out = self.net(x)
        
        # Apply appropriate activations
        congestion = self.sigmoid(out[:, 0:1])
        travel_time = 1.0 + self.relu(out[:, 1:2]) * 2.0  # 1.0 to 3.0
        speed = 5.0 + self.relu(out[:, 2:3]) * 70.0  # 5 to 75 mph
        
        return torch.cat([congestion, travel_time, speed], dim=1)


def train_model(epochs=50, batch_size=64, learning_rate=0.001):
    """Train the lightweight traffic model"""
    
    print("🚀 Starting lightweight model training...")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    
    # Create dataset
    train_dataset = TrafficDataset(num_samples=8000)
    val_dataset = TrafficDataset(num_samples=2000)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Using device: {device}")
    
    model = LightweightTrafficNet()
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            models_dir = Path(__file__).parent / "models"
            models_dir.mkdir(exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': 8,
                'hidden_size': 128,
                'output_size': 3,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'timestamp': datetime.now().isoformat()
            }, models_dir / "lightweight_traffic_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"⏹️  Early stopping at epoch {epoch+1}")
                break
    
    print(f"✅ Training complete! Best validation loss: {best_val_loss:.4f}")
    return model


def test_model():
    """Test the trained model with realistic scenarios"""
    
    print("\n🧪 Testing trained model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightweightTrafficNet()
    
    models_dir = Path(__file__).parent / "models"
    checkpoint = torch.load(models_dir / "lightweight_traffic_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Test scenarios
    scenarios = [
        {"name": "Austin 2 AM Sunday", "lat": 30.2672, "lng": -97.7431, "hour": 2, "day": 6, "dist": 2},
        {"name": "Austin 8 AM Monday", "lat": 30.2672, "lng": -97.7431, "hour": 8, "day": 0, "dist": 2},
        {"name": "Austin 12 PM Saturday", "lat": 30.2672, "lng": -97.7431, "hour": 12, "day": 5, "dist": 2},
        {"name": "Austin 5 PM Friday", "lat": 30.2672, "lng": -97.7431, "hour": 17, "day": 4, "dist": 2},
        {"name": "Rural Texas 8 AM", "lat": 31.0, "lng": -98.0, "hour": 8, "day": 0, "dist": 50},
    ]
    
    print("\n" + "="*70)
    print("Scenario".ljust(30) + "Congestion".ljust(15) + "Speed (mph)")
    print("="*70)
    
    with torch.no_grad():
        for scenario in scenarios:
            features = torch.FloatTensor([[
                scenario["lat"], scenario["lng"], scenario["hour"], 
                scenario["day"], int(scenario["day"] >= 5), 0,  # is_weekend, is_holiday
                scenario["dist"], 1.0  # distance from center, population
            ]]).to(device)
            
            output = model(features)
            congestion = output[0, 0].item()
            speed = output[0, 2].item()
            
            print(f"{scenario['name']:<30}{congestion*100:>6.1f}%{speed:>18.1f}")
    
    print("="*70)


if __name__ == "__main__":
    # Train the model
    model = train_model(epochs=50, batch_size=64, learning_rate=0.001)
    
    # Test it
    test_model()
    
    print("\n✅ Model saved to ml/models/lightweight_traffic_model.pth")
    print("🔄 Update deep_learning_service.py to use this new model file!")
