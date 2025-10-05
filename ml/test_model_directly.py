import torch
from torch import nn
import numpy as np

# Define the model architecture
class LightweightTrafficNet(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, output_size=3):
        super().__init__()
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

# Load the model
checkpoint = torch.load('models/lightweight_traffic_model.pth', map_location='cpu')
model = LightweightTrafficNet()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Testing lightweight model predictions:\n")

# Test cases
test_cases = [
    {"name": "Austin 2 AM Sunday", "features": [30.2672, -97.7431, 2.0, 6.0, 1.0, 0.0, 5.0, 1.0]},
    {"name": "Austin 8 AM Monday", "features": [30.2672, -97.7431, 8.0, 0.0, 0.0, 0.0, 5.0, 1.0]},
    {"name": "Austin 5 PM Friday", "features": [30.2672, -97.7431, 17.0, 4.0, 0.0, 0.0, 5.0, 1.0]},
    {"name": "Austin 12 PM Saturday", "features": [30.2672, -97.7431, 12.0, 5.0, 1.0, 0.0, 5.0, 1.0]},
]

for test in test_cases:
    features = torch.FloatTensor([test["features"]])
    with torch.no_grad():
        output = model(features)
        pred = output.numpy()[0]
    
    print(f"{test['name']}:")
    print(f"  Features: {test['features']}")
    print(f"  Raw output: {pred}")
    print(f"  Congestion: {pred[0]:.1%}, Travel Time Index: {pred[1]:.2f}, Speed: {pred[2]:.1f} mph")
    print()
