# 🚀 Speed & Units Customization - v3.2 Upgrade

## 📋 What's New

**User Request:** "give users an option to add speed and use miles instead of KM and shit"

**Implementation Date:** January 2025  
**Version:** 3.2.0 (upgrade from 3.1)

---

## 🎯 Problem Solved

In v3.1, we fixed the "meaningless travel time" issue by implementing proper origin-destination routing. However, the system was still limited:

1. **No speed customization** - Used hardcoded 50 km/h average speed
2. **No unit preferences** - All distances shown in kilometers only
3. **Limited flexibility** - Users couldn't adjust for different road types

**User feedback was clear**: People need to specify their expected speed (residential vs highway) and prefer their local unit system (miles vs kilometers).

---

## ✨ New Features

### 1. **Speed Selection** 🏎️

Users can now select average speed based on road type:

| Speed Option | MPH | KM/H | Use Case |
|--------------|-----|------|----------|
| Residential  | 25  | 40   | Neighborhood streets |
| City Streets | 35  | 56   | Local roads |
| Main Roads   | 45  | 72   | **Default** - Typical driving |
| Highway      | 55  | 88   | State highways |
| Expressway   | 65  | 105  | Express lanes |
| Interstate   | 75  | 120  | Interstate highways |

**Why this matters:**
- More accurate time estimates
- Route-specific speed assumptions
- Real-world driving conditions

### 2. **Unit System Toggle** 📏

Users can switch between Imperial (miles) and Metric (kilometers):

**Imperial (US)**
- Distance: Miles (mi)
- Speed: Miles per hour (mph)
- Default selected

**Metric (International)**
- Distance: Kilometers (km)
- Speed: Kilometers per hour (km/h)

**Conversions:**
- 1 mile = 1.60934 km
- 1 km = 0.621371 miles

### 3. **Smart Display Updates** 📊

All distance and speed displays now respect unit preferences:
- Distance metric card
- Average speed display
- Units update automatically
- No page reload needed

---

## 🔧 Technical Implementation

### Frontend Changes (`index.html`)

#### 1. New Form Fields

```html
<!-- Speed Selection Dropdown -->
<div class="form-group">
    <label for="speedLimit">Average Speed</label>
    <select id="speedLimit" required>
        <option value="25">Residential (25 mph / 40 km/h)</option>
        <option value="35">City Streets (35 mph / 56 km/h)</option>
        <option value="45" selected>Main Roads (45 mph / 72 km/h)</option>
        <option value="55">Highway (55 mph / 88 km/h)</option>
        <option value="65">Expressway (65 mph / 105 km/h)</option>
        <option value="75">Interstate (75 mph / 120 km/h)</option>
    </select>
</div>

<!-- Unit System Toggle -->
<div class="form-group">
    <label for="unitSystem">Distance Units</label>
    <select id="unitSystem" required>
        <option value="imperial" selected>Miles (Imperial)</option>
        <option value="metric">Kilometers (Metric)</option>
    </select>
</div>
```

#### 2. Updated Results Display

```html
<!-- Distance Card -->
<div class="metric-card">
    <div class="metric-label">Distance</div>
    <div class="metric-value" id="distance">--</div>
    <div class="metric-unit" id="distanceUnit">mi</div>
</div>

<!-- Speed Card (NEW) -->
<div class="metric-card">
    <div class="metric-label">Avg Speed</div>
    <div class="metric-value" id="avgSpeed">--</div>
    <div class="metric-unit" id="speedUnit">mph</div>
</div>
```

#### 3. JavaScript Conversion Logic

```javascript
// Get user preferences
const speedLimit = document.getElementById('speedLimit').value;  // In MPH
const unitSystem = document.getElementById('unitSystem').value;  // 'imperial' or 'metric'

// Convert units based on preference
let distance = prediction.distance_km;  // Backend returns km
let avgSpeed = parseFloat(speedLimit);   // User selected mph
let distanceUnit = 'km';
let speedUnit = 'km/h';

if (unitSystem === 'imperial') {
    // Convert distance to miles
    distance = distance * 0.621371;  // km to miles
    distanceUnit = 'mi';
    speedUnit = 'mph';
} else {
    // Convert speed to km/h for display
    avgSpeed = avgSpeed * 1.60934;  // mph to km/h
    speedUnit = 'km/h';
}

// Update display
document.getElementById('distance').textContent = distance.toFixed(1);
document.getElementById('distanceUnit').textContent = distanceUnit;
document.getElementById('avgSpeed').textContent = avgSpeed.toFixed(0);
document.getElementById('speedUnit').textContent = speedUnit;
```

### Backend Changes (`backend/main.py`)

#### 1. Updated API Endpoint

```python
@app.post("/api/predict-location")
async def predict_by_location(
    dest_latitude: float,
    dest_longitude: float,
    origin_latitude: Optional[float] = None,
    origin_longitude: Optional[float] = None,
    hour: int = 8,
    day_of_week: int = 0,
    date: Optional[str] = None,
    speed_limit: int = 45  # NEW PARAMETER
):
```

#### 2. Speed Conversion for Distance Service

```python
# Convert speed_limit from mph to km/h for distance service
speed_kmh = speed_limit * 1.60934

# Pass to distance service
travel_info = distance_service.get_travel_time(
    origin_lat=origin_latitude,
    origin_lng=origin_longitude,
    dest_lat=dest_latitude,
    dest_lng=dest_longitude,
    hour=hour,
    day_of_week=day_of_week,
    departure_time=date_obj,
    avg_speed_kmh=speed_kmh  # NEW PARAMETER
)
```

#### 3. Response Enhancement

```python
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
    "speed_limit_mph": speed_limit,          # NEW FIELD
    "speed_limit_kmh": round(speed_kmh, 1)   # NEW FIELD
}

return {
    "success": True,
    "prediction": prediction,
    "model_version": "3.1.0",
}
```

---

## 🎨 UI Updates

### Form Layout (4-Column Grid)

```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ Hour of Day │    Date     │ Avg Speed   │ Dist Units  │
│             │             │             │             │
│    0-23     │ Date Picker │  Dropdown   │   Toggle    │
└─────────────┴─────────────┴─────────────┴─────────────┘
┌───────────────────────────┬───────────────────────────┐
│      Origin (Green)       │   Destination (Blue)      │
│   Right-click on map      │   Left-click on map       │
└───────────────────────────┴───────────────────────────┘
```

### Results Display (Updated)

```
┌──────────┬──────────┬──────────────┬──────────────┐
│ Distance │ Avg Speed│ Baseline Time│ With Traffic │
│  12.5 mi │  45 mph  │   16.7 mins  │   24.2 mins  │
└──────────┴──────────┴──────────────┴──────────────┘
```

**Key Changes:**
- Added "Avg Speed" card between Distance and Baseline Time
- Unit labels dynamically update (mi/km, mph/km/h)
- All values rounded appropriately

---

## 📡 API Changes

### Request Example

**Before (v3.1):**
```
POST /api/predict-location?
  origin_latitude=32.7357&
  origin_longitude=-97.1081&
  dest_latitude=32.7500&
  dest_longitude=-97.1200&
  hour=8&
  day_of_week=1&
  date=2025-10-05
```

**After (v3.2):**
```
POST /api/predict-location?
  origin_latitude=32.7357&
  origin_longitude=-97.1081&
  dest_latitude=32.7500&
  dest_longitude=-97.1200&
  hour=8&
  day_of_week=1&
  date=2025-10-05&
  speed_limit=55              ← NEW
```

### Response Example

**Before (v3.1):**
```json
{
  "success": true,
  "prediction": {
    "distance_km": 5.2,
    "baseline_travel_time_min": 15.5,
    "adjusted_travel_time_min": 22.3,
    ...
  },
  "model_version": "3.0.0"
}
```

**After (v3.2):**
```json
{
  "success": true,
  "prediction": {
    "distance_km": 5.2,
    "baseline_travel_time_min": 15.5,
    "adjusted_travel_time_min": 22.3,
    "speed_limit_mph": 55,        ← NEW
    "speed_limit_kmh": 88.5,      ← NEW
    ...
  },
  "model_version": "3.1.0"
}
```

---

## 🚀 How to Use

### Step 1: Start the Backend

```powershell
# Use the v3.1 startup script (compatible with v3.2)
.\START_V3.1.bat
```

Or manually:
```powershell
cd backend
python main.py
```

### Step 2: Open Frontend

Navigate to:
```
http://localhost:8001
```

Or open `index.html` in your browser.

### Step 3: Set Your Preferences

1. **Select Speed**: Choose road type that matches your route
   - Residential (25 mph) for neighborhood
   - Highway (55 mph) for state routes
   - Interstate (75 mph) for long trips

2. **Choose Units**: Pick your preferred measurement system
   - Imperial (miles/mph) - Default for US
   - Metric (km/km/h) - International standard

3. **Select Route**:
   - Right-click map → Set origin (green marker)
   - Left-click map → Set destination (blue marker)

4. **Click "Calculate Route & Traffic"**

### Step 4: View Results

Results now show:
- **Distance** in your chosen units (mi or km)
- **Avg Speed** converted to your units (mph or km/h)
- **Travel Times** calculated using your speed preference
- **Traffic Impact** with congestion adjustments

---

## 🔍 Behind the Scenes

### Speed Impact on Travel Time

The speed selection affects the **baseline travel time calculation** in the Haversine fallback method:

```python
# distance_service.py (Haversine method)
def estimate_travel_time_haversine(
    origin_lat, origin_lng, 
    dest_lat, dest_lng, 
    avg_speed_kmh=50.0  # Now user-customizable!
):
    distance_km = haversine_distance(...)
    city_routing_multiplier = 1.3  # Account for non-straight routes
    actual_distance = distance_km * city_routing_multiplier
    
    # Time = Distance / Speed
    travel_time_hours = actual_distance / avg_speed_kmh
    return travel_time_hours * 60  # Convert to minutes
```

**Impact:**
- Higher speed → Shorter baseline time
- Lower speed → Longer baseline time
- More realistic estimates for route type

### Unit Conversion Strategy

**Why convert on frontend, not backend?**

1. **Backend stays in metric** (international standard)
2. **Google Maps API uses metric** (easier integration)
3. **ML models trained on metric data** (consistency)
4. **Frontend converts for display only** (user preference)

This approach:
- ✅ Keeps backend simple and consistent
- ✅ Supports multiple clients with different preferences
- ✅ Easy to add more unit systems later
- ✅ No data corruption from multiple conversions

---

## 📊 Example Scenarios

### Scenario 1: Morning Commute (Imperial)

**Settings:**
- Speed: 45 mph (Main Roads)
- Units: Imperial (miles)
- Route: Home → Office (8.5 km)
- Time: 8:00 AM, Monday

**Results:**
- Distance: **5.3 mi**
- Avg Speed: **45 mph**
- Baseline Time: 11.3 mins
- With Traffic: 16.9 mins (+49% delay)

### Scenario 2: Highway Trip (Metric)

**Settings:**
- Speed: 75 mph (Interstate)
- Units: Metric (kilometers)
- Route: City A → City B (120 km)
- Time: 2:00 PM, Saturday

**Results:**
- Distance: **120.0 km**
- Avg Speed: **121 km/h**
- Baseline Time: 59.5 mins
- With Traffic: 65.4 mins (+10% delay)

### Scenario 3: Residential Delivery (Imperial)

**Settings:**
- Speed: 25 mph (Residential)
- Units: Imperial (miles)
- Route: Store → Customer (2.5 km)
- Time: 5:00 PM, Friday

**Results:**
- Distance: **1.6 mi**
- Avg Speed: **25 mph**
- Baseline Time: 3.8 mins
- With Traffic: 5.5 mins (+45% delay)

---

## 🎯 Benefits

### For Users
1. **Personalized Experience**
   - Choose speed that matches driving style
   - See distances in familiar units
   - More accurate time estimates

2. **Better Planning**
   - Adjust for road conditions
   - Account for speed limits
   - Compare scenarios easily

3. **International Support**
   - Works for US drivers (miles/mph)
   - Works for international (km/km/h)
   - Easy to switch between systems

### For Developers
1. **Clean Architecture**
   - Backend stays in metric (consistency)
   - Frontend handles conversion (flexibility)
   - Easy to extend with more units

2. **Better API Design**
   - User preferences passed as parameters
   - Responses include both units
   - Backward compatible with v3.1

3. **Improved Accuracy**
   - Speed affects baseline calculation
   - More realistic time estimates
   - User-controlled assumptions

---

## 🔄 Version History

### v3.2 (Current) - Speed & Units
- ✅ Speed selection (25-75 mph)
- ✅ Unit system toggle (miles/km)
- ✅ Smart display updates
- ✅ Backend speed integration

### v3.1 - Route Calculation
- ✅ Origin + destination routing
- ✅ Google Maps API integration
- ✅ Gemini AI fallback
- ✅ Real travel times

### v3.0 - Enhanced Models & UI
- ✅ 100K training dataset
- ✅ 6 improved ML models
- ✅ Date picker (30 days)
- ✅ Dark glassmorphism UI

### v2.1 - Calendar Integration
- ✅ Google Calendar sync
- ✅ 30-day holiday lookup
- ✅ Holiday impact on traffic

### v2.0 - Initial Release
- ✅ Map-based prediction
- ✅ Pin functionality
- ✅ Basic traffic model

---

## 🛠️ Configuration

### Default Values

```javascript
// Frontend defaults (index.html)
const DEFAULT_SPEED = 45;           // mph
const DEFAULT_UNITS = 'imperial';   // miles/mph
const DEFAULT_ORIGIN = {            // UT Arlington
    lat: 32.7357,
    lng: -97.1081
};
```

### Backend defaults

```python
# Backend defaults (main.py)
speed_limit: int = 45  # mph
```

### Customization Options

To change defaults, edit:

**Speed options** (add more if needed):
```javascript
<option value="40">Custom (40 mph / 64 km/h)</option>
```

**Unit systems** (add more if needed):
```javascript
<option value="nautical">Nautical (nautical miles)</option>
```

---

## 📝 Migration Guide

### From v3.1 to v3.2

**No breaking changes!** v3.2 is backward compatible.

**Files changed:**
1. `index.html` - Added speed/unit fields, conversion logic
2. `backend/main.py` - Added speed_limit parameter

**What stays the same:**
- All v3.1 route calculation features
- Google Maps / Gemini API integration
- Distance service interface
- ML prediction models
- Calendar integration

**New optional parameters:**
- `speed_limit` (API) - defaults to 45 mph if not provided
- `unitSystem` (UI) - defaults to 'imperial' if not selected

---

## 🐛 Known Limitations

1. **Speed affects Haversine only**
   - Google Maps uses real-time traffic data (ignores speed_limit)
   - Gemini AI uses its own estimation
   - Speed parameter only affects Haversine fallback

2. **Unit conversion is display-only**
   - Backend always returns metric (km)
   - Frontend converts for display
   - This is intentional (keeps backend simple)

3. **No custom speed input**
   - Users pick from dropdown (25, 35, 45, 55, 65, 75 mph)
   - Can't type arbitrary speed like "42 mph"
   - Easy to add if needed

---

## 🚀 Future Enhancements

**Potential v3.3 features:**

1. **Save Preferences**
   - Remember unit selection
   - Store favorite speed
   - Save recent routes

2. **Custom Speed Input**
   - Allow typing any speed (e.g., "42 mph")
   - Validate range (5-100 mph)
   - Show in dropdown for quick access

3. **More Unit Systems**
   - Nautical miles
   - Meters/kilometers
   - Regional preferences

4. **Speed Zones**
   - Detect road types automatically
   - Suggest speed based on location
   - Show speed limit from maps

---

## 📚 Documentation

- **Main Guide:** `README.md`
- **Route Calculation:** `ROUTE_CALCULATION_V3.1.md`
- **This Document:** `SPEED_AND_UNITS_V3.2.md`

## 🙋 Support

**Issues?** Check:
1. Backend running on port 8001?
2. Selected both origin and destination?
3. Speed and units selected in form?

**Questions?** Refer to:
- v3.1 documentation for route calculation basics
- This document for speed/unit features

---

## ✅ Summary

**v3.2 gives users control over:**
- ✅ **Speed assumptions** (25-75 mph based on road type)
- ✅ **Display units** (miles/mph vs km/km/h)
- ✅ **Better accuracy** (realistic time estimates)
- ✅ **Personal preference** (US vs International units)

**All while maintaining:**
- ✅ v3.1 route calculation features
- ✅ Origin + destination routing
- ✅ Google Maps / Gemini integration
- ✅ ML traffic predictions
- ✅ Dark glassmorphism UI

**Next steps:** Use the app with your preferred speed and units! 🎉
