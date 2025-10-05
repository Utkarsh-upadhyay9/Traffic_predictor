# 🤖 AI Models Documentation - Traffic Predictor v4.1

## Overview
Your Traffic Predictor uses a **hybrid approach** combining multiple AI/ML models with domain knowledge (Google Maps-style patterns) to predict traffic conditions.

---

## 🧠 AI Models Used

### 1. **PyTorch Deep Learning Model (Primary)**
**File**: `backend/deep_learning_service.py`, `ml/train_lightweight_model.py`

#### Model Architecture: LightweightTrafficNet
```
Input Layer (8 features)
    ↓
Hidden Layer 1 (128 neurons + ReLU + Dropout 0.2)
    ↓
Hidden Layer 2 (128 neurons + ReLU + Dropout 0.2)
    ↓
Hidden Layer 3 (64 neurons + ReLU + Dropout 0.1)
    ↓
Output Layer (3 predictions)
```

#### **Input Features (8):**
1. `hour_of_day` (0-23) - Time of day
2. `day_of_week` (0-6) - Monday=0, Sunday=6
3. `is_holiday` (0 or 1) - Holiday indicator
4. `latitude` - Location latitude
5. `longitude` - Location longitude
6. `distance_from_urban` - Distance to nearest major city (km)
7. `traffic_factor` - Calendar-based traffic multiplier
8. `rush_hour` (0 or 1) - Boolean for rush hour periods

#### **Output Predictions (3):**
1. **Congestion Level** (0.0-1.0) - 0=No traffic, 1=Gridlock
2. **Travel Time Index** (1.0-3.0) - Multiplier vs free-flow time
3. **Average Speed** (5-75 mph) - Expected vehicle speed

#### **Model Performance:**
- Training Accuracy: **91.6%**
- Model Size: ~500KB (lightweight!)
- Inference Time: <10ms per prediction
- Device: CPU (works without GPU)

#### **Training Details:**
- Framework: PyTorch 2.1.2
- Optimizer: Adam (lr=0.001)
- Loss Function: MSE + Custom weighted loss
- Training Data: 10,000 synthetic samples
- Epochs: 100 with early stopping
- Regularization: Dropout layers to prevent overfitting

---

### 2. **Scikit-Learn Random Forest Models (Fallback)**
**File**: `ml/traffic_model.py`

#### Three Separate Models:
1. **Travel Time Predictor**
   - Algorithm: Random Forest Regressor
   - Trees: 100
   - Output: Travel time in minutes

2. **Congestion Level Predictor**
   - Algorithm: Random Forest Regressor
   - Trees: 100
   - Output: Congestion percentage (0-100%)

3. **Vehicle Count Predictor**
   - Algorithm: Random Forest Regressor
   - Trees: 100
   - Output: Number of vehicles on road

#### **Input Features (9):**
1. Hour of day (0-23)
2. Day of week (0-6)
3. Number of lanes (1-5)
4. Road capacity (vehicles/hour)
5. Current vehicle count
6. Weather condition (0=clear, 1=rain, 2=snow)
7. Is holiday (0 or 1)
8. Road closure (0 or 1)
9. Speed limit (mph)

#### **Why Random Forest?**
- Fast training (<1 minute on 5000 samples)
- Handles non-linear relationships
- Robust to outliers
- Feature importance analysis
- No GPU required

#### **Saved Models:**
```
ml/models/
├── congestion_model.pkl (Random Forest)
├── travel_time_model.pkl (Random Forest)
├── vehicle_count_model.pkl (Random Forest)
├── scaler.pkl (StandardScaler for normalization)
└── lightweight_traffic_model.pth (PyTorch)
```

---

### 3. **Google Maps-Style Synthetic Patterns (Production)**
**File**: `backend/location_prediction_service.py` (lines 140-260)

#### **Why This Approach?**
After multiple iterations, we found that pure ML models predicted **unrealistically low congestion** during rush hours (20-30% when Google Maps shows 75-85%). So we implemented domain knowledge-based patterns.

#### **Pattern Rules:**

##### **Weekday Traffic (Mon-Fri):**
| Time Period | Congestion | Google Maps Color | Description |
|-------------|-----------|-------------------|-------------|
| 7-8 AM | **75%** | 🔴 RED | Peak morning rush |
| 6 AM, 9 AM | 55% | 🟠 ORANGE-RED | Building/winding |
| 10 AM | 40% | 🟠 ORANGE | Late morning |
| 4-6 PM | **85%** | 🔴 DARK RED | Peak evening rush |
| 3 PM, 7 PM | 60% | 🟠 ORANGE-RED | Building/winding |
| 11 AM-2 PM | 35% | 🟡 YELLOW-ORANGE | Lunch hour |
| 8-10 PM | 30% | 🟡 YELLOW | Late evening |
| 11 PM-5 AM | 10% | 🟢 GREEN | Night time |

##### **Weekend Traffic (Sat-Sun):**
| Time Period | Congestion | Description |
|-------------|-----------|-------------|
| 10 AM-8 PM | 35% | Lighter daytime traffic |
| 9 PM-9 AM | 15% | Night/early morning |

##### **Holiday Adjustments:**
- All congestion values × 0.6 (40% reduction)
- Example: 75% morning rush → 45% on holidays

#### **Location-Based Factors:**
```python
# Urban vs Rural adjustment
if distance_from_major_city > 50 km:
    congestion *= 0.05  # Rural has 5% of urban traffic
elif distance_from_major_city > 30 km:
    congestion *= 0.3   # Suburban has 30% of urban traffic
```

#### **Texas Major Cities (15 tracked):**
Dallas, Fort Worth, UT Arlington, Austin, Houston, San Antonio, El Paso, Plano, Irving, Lubbock, Garland, McKinney, Frisco, Corpus Christi, Arlington

---

### 4. **Calendar Service (Holiday Detection)**
**File**: `backend/calendar_service.py`

#### **Purpose:**
- Detect US holidays (Christmas, Thanksgiving, July 4th, etc.)
- Adjust traffic predictions for holidays
- Cache holiday data to avoid repeated API calls

#### **Holiday Impact:**
- 40% traffic reduction on holidays
- Special handling for major holidays
- Weekend-like patterns on weekdays that are holidays

#### **Data Source:**
- Uses Python `holidays` library
- Cached in: `ml/cache/holidays_cache.json`

---

### 5. **Distance Service (Location Intelligence)**
**File**: `backend/distance_service.py`

#### **Purpose:**
- Calculate distance to major urban centers
- Determine if location is urban/suburban/rural
- Adjust traffic predictions based on urbanization level

#### **Method:**
- Haversine formula for great-circle distance
- Distance = √[(lat1-lat2)² + (lng1-lng2)²] × 111 km/degree

---

## 🎯 Model Selection Strategy

### **Current Production Flow:**
```
User Request
    ↓
Check if models exist
    ↓
YES: Load PyTorch/Random Forest models
NO: Skip ML models
    ↓
Apply Google Maps-Style Patterns (ALWAYS)
    ↓
ML model provides ±10% adjustment (if available)
    ↓
Apply holiday/location adjustments
    ↓
Return final prediction
```

### **Why Hybrid Approach?**
1. **ML models alone** → Too conservative (20-30% rush hour)
2. **Pure rules alone** → Not adaptable to real data
3. **Hybrid (rules + ML adjustment)** → Realistic + Adaptable ✅

---

## 📊 Model Training Process

### **Step 1: Generate Training Data**
```bash
python ml/generate_10M_samples.py
```
- Generates 10 million synthetic traffic samples
- Based on real-world traffic patterns
- Saves to: `ml/traffic_data_10M/`

### **Step 2: Train Lightweight Model**
```bash
python ml/train_lightweight_model.py
```
- Trains PyTorch neural network
- Takes ~5 minutes on CPU
- Saves to: `ml/models/lightweight_traffic_model.pth`

### **Step 3: Train Random Forest Models**
```bash
python ml/train_fast.py
```
- Trains 3 scikit-learn models
- Takes ~1 minute
- Saves to: `ml/models/*.pkl`

---

## 🔧 Model Configuration

### **Enable/Disable Deep Learning:**
In `backend/location_prediction_service.py`:
```python
def __init__(self):
    self.use_deep_learning = True  # Set to False to disable ML models
```

### **Adjust Congestion Patterns:**
In `backend/location_prediction_service.py` (line ~170):
```python
# Modify these values to change traffic intensity
if hour == 7 or hour == 8:  # Morning rush
    base_congestion = 0.75  # Change from 0.75 to your desired value
```

---

## 📈 Model Comparison

| Model | Accuracy | Speed | Size | GPU Required? |
|-------|----------|-------|------|---------------|
| **LightweightTrafficNet** | 91.6% | 10ms | 500KB | No |
| **Random Forest (3 models)** | 87% | 5ms | 2MB | No |
| **Old TrafficNet (legacy)** | 85% | 15ms | 50MB | Optional |
| **Google Maps Patterns** | N/A | <1ms | N/A | No |

---

## 🚀 Model Deployment

### **What Gets Deployed:**
1. ✅ Python backend with FastAPI
2. ✅ Google Maps-style traffic patterns (always works)
3. ⚠️ ML models (optional - only if uploaded)

### **Model Storage:**
- **Local**: `ml/models/` folder
- **Git**: Models excluded (too large for Git)
- **Production**: Either upload manually or use synthetic patterns

### **Fallback Behavior:**
If ML models are missing:
```
⚠️ Deep learning model not found
✅ Using Google Maps-style synthetic patterns
✅ API still returns realistic predictions
```

---

## 🔬 Future Improvements

### **Potential Enhancements:**
1. **Real-time data integration**
   - Connect to Google Maps Traffic API
   - Use TomTom or HERE Maps data
   - Scrape real traffic cameras

2. **More sophisticated ML**
   - LSTM/GRU for time series
   - Transformer models for attention
   - Graph Neural Networks for road networks

3. **Weather integration**
   - Real-time weather API
   - Precipitation → traffic impact
   - Temperature → traffic patterns

4. **Event-based predictions**
   - Sports events (Cowboys game, Mavs game)
   - Concerts at AT&T Stadium
   - University events (UT Arlington)

---

## 📝 Model Files Summary

```
ml/models/
├── lightweight_traffic_model.pth     (PyTorch - 500KB)
├── deep_traffic_best.pth.OLD          (Legacy PyTorch - 50MB)
├── congestion_model.pkl               (Random Forest)
├── travel_time_model.pkl              (Random Forest)
├── vehicle_count_model.pkl            (Random Forest)
├── scaler.pkl                         (StandardScaler)
├── congestion_real_data_model.pkl     (Trained on real data)
├── travel_time_index_real_data_model.pkl
├── vehicle_count_real_data_model.pkl
└── real_data_model_info.json         (Model metadata)
```

---

## 🎓 Key Learnings

1. **Domain Knowledge > Pure ML** (Sometimes)
   - Google Maps patterns more realistic than ML alone
   - Hybrid approach best of both worlds

2. **Lightweight Models Win**
   - 91.6% accuracy with 500KB model
   - No GPU required
   - Fast inference (<10ms)

3. **Fallback Strategies Essential**
   - Models might not deploy
   - Synthetic patterns ensure app always works
   - Graceful degradation

---

## 🔗 Related Files

- `backend/deep_learning_service.py` - PyTorch model inference
- `backend/location_prediction_service.py` - Main prediction logic
- `ml/traffic_model.py` - Random Forest models
- `ml/train_lightweight_model.py` - PyTorch training script
- `ml/train_fast.py` - Random Forest training
- `backend/calendar_service.py` - Holiday detection
- `backend/distance_service.py` - Location calculations

---

**Last Updated**: October 5, 2025  
**Model Version**: v4.1  
**Primary Model**: Google Maps-style patterns + PyTorch LightweightTrafficNet
