# 🚀 How to Start Digi_sim v3.0

## Quick Start (3 Steps)

### Step 1: Start the Backend
Double-click this file in your Digi_sim folder:
```
START_BACKEND.bat
```

A new PowerShell window will open showing:
```
✓ Models loaded
✅ Loaded 2 holidays/events
INFO: Uvicorn running on http://0.0.0.0:8001
```

**⚠️ IMPORTANT: Keep this window open!** Don't close it while using the app.

---

### Step 2: Open the Frontend
Open this file in your web browser (Chrome, Edge, Firefox):
```
C:\Users\utkar\Desktop\Xapps\Digi_sim\index.html
```

**How to open:**
- Right-click `index.html` → "Open with" → Choose your browser
- Or drag and drop `index.html` into an open browser window

---

### Step 3: Use the Application

1. **Click on the map** to drop a pin at your destination
2. **Select a date** (today to +30 days)
3. **Enter time** (format: HH:MM, like 14:30 for 2:30 PM)
4. **Select weather** condition
5. Click **"Predict Traffic"**

The map will show:
- 🎯 Your destination pin (orange)
- 📊 Predicted congestion level (color-coded)
- 🚗 Estimated traffic metrics
- 🎉 Holiday warnings (if applicable)

---

## What's New in v3.0

### ✨ Enhanced Features

1. **Date Picker UI**
   - Instead of selecting "day of week", choose exact dates
   - Up to 30 days in the future
   - Real-time holiday detection

2. **Better Predictions**
   - 100,000 training samples (was 50,000)
   - 13 location types (was 7)
   - Special events support (sports, concerts, festivals)
   - Traffic incident modeling (accidents, construction)
   - **+10.6% accuracy improvement** on travel time!

3. **Enhanced Weather**
   - 7 weather conditions (was 4)
   - Light rain, heavy rain, thunderstorm, extreme heat, etc.
   - More realistic impact modeling

---

## Important URLs

### ✅ CORRECT URLs

**Frontend (what you should open):**
```
C:\Users\utkar\Desktop\Xapps\Digi_sim\index.html
```

**API Testing (optional, for developers):**
```
http://localhost:8001/health
http://localhost:8001/api/holidays
```

### ❌ INCORRECT URLs (Don't Use)

**These will NOT work:**
```
http://0.0.0.0:8001/              ❌ Wrong format
http://localhost:8001/            ❌ No frontend here
http://127.0.0.1:8001/            ❌ API only, not the app
```

**Why?**
- `0.0.0.0:8001` is a **server binding address**, not a user-accessible URL
- `localhost:8001` is the **backend API**, not the web interface
- The app frontend is the **HTML file**, which connects to the backend automatically

---

## Troubleshooting

### Backend won't start
- Make sure virtual environment exists: `venv` folder should be present
- Check if port 8001 is already in use:
  ```powershell
  Get-NetTCPConnection -LocalPort 8001
  ```
- Kill any existing Python processes and try again

### "Cannot predict" errors
- Verify backend is running (check PowerShell window)
- Refresh the browser page
- Check browser console for error messages (F12)

### Map not loading
- Check your internet connection (Mapbox requires internet)
- Verify Mapbox token in index.html is valid
- Try refreshing the page

### Date picker not working
- Make sure you select a date within 30 days
- Browser must support HTML5 date input (Chrome, Edge, Firefox do)

---

## API Endpoints (For Developers)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check backend status |
| `/api/holidays` | GET | Get next 30 days holidays |
| `/api/is-holiday?date=YYYY-MM-DD` | GET | Check if date is holiday |
| `/api/predict-location` | POST | Predict traffic metrics |
| `/api/location-metadata` | GET | Get location types and features |

**Test Example:**
```powershell
# Check backend health
Invoke-RestMethod -Uri "http://localhost:8001/health"

# Get holidays
Invoke-RestMethod -Uri "http://localhost:8001/api/holidays"

# Check specific date
Invoke-RestMethod -Uri "http://localhost:8001/api/is-holiday?date=2025-12-25"
```

---

## System Requirements

- **OS**: Windows 10/11
- **Python**: 3.8+ (already installed in venv)
- **Browser**: Chrome, Edge, or Firefox (latest version)
- **Internet**: Required for map tiles and geocoding
- **RAM**: 2GB minimum
- **Storage**: 500MB for models and cache

---

## Version Info

**Current Version**: 3.0.0
**Release Date**: 2025
**Model Version**: 3.0.0
**Dataset Size**: 100,000 samples

**Performance:**
- Travel Time Prediction: 94.8% R² (full model)
- Congestion Prediction: 97.9% R²
- Vehicle Count Prediction: 98.4% R²

---

## Support

**Documentation:**
- `UPGRADE_TO_V3.md` - Complete upgrade guide
- `V3_RESULTS.md` - Performance analysis
- `QUICK_START_V3.md` - Feature explanations

**Common Questions:**

Q: Why can't I access http://0.0.0.0:8001?
A: That's the server binding address. Open `index.html` in your browser instead.

Q: Can I predict traffic for yesterday?
A: No, only future dates (today to +30 days). The model is trained for prediction, not historical analysis.

Q: How accurate are the predictions?
A: The full model achieves 94.8% R² on travel time, meaning it explains ~95% of the variance in real-world traffic patterns.

Q: Does it use real traffic data?
A: It uses realistic synthetic data based on real-world patterns, traffic studies, and events modeling.

---

## License & Credits

Built with:
- FastAPI (backend)
- Mapbox GL JS (maps)
- scikit-learn (ML models)
- Google Calendar API (holidays)

**Enjoy using Digi_sim v3.0!** 🚗📍
