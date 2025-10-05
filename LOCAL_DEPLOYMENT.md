# 🚀 Local Deployment Guide

## Quick Start - Run Locally

### Option 1: Simple HTTP Server (No Backend - Frontend Only)

If you just want to test the UI without predictions:

```powershell
# Open index.html directly in browser
start index.html
```

**Note**: API calls will fail, but you can see the map interface.

---

### Option 2: Full Stack (Backend + Frontend)

#### Step 1: Install Dependencies
```powershell
# Install backend dependencies
cd backend
pip install -r requirements.txt
```

#### Step 2: Start Backend
```powershell
# From project root
cd backend
python main.py
```

**Expected Output:**
```
✅ Loaded holidays from cache
✅ Loaded lightweight model on cpu
INFO:     Uvicorn running on http://0.0.0.0:8000
```

#### Step 3: Open Frontend
```powershell
# In a new terminal or just open in browser
start ../index.html
```

---

## ✅ YOUR APP IS NOW RUNNING LOCALLY!

- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Frontend**: Open `index.html` in your browser

---

## 🔧 Troubleshooting

### Problem: "Port 8000 already in use"
```powershell
# Find what's using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with the number from above)
taskkill /PID <PID> /F
```

### Problem: "Module not found"
```powershell
# Make sure you're in the backend directory
cd backend
pip install -r requirements.txt
```

### Problem: " Gemini import takes forever"
The `google-generativeai` package can be slow to import on Windows. Just wait 10-15 seconds.

### Problem: "KeyboardInterrupt on startup"
This is normal if imports take time. Just let it finish loading (10-15 seconds).

---

## 📊 What's Running?

### Backend Services:
- ✅ FastAPI web server (port 8000)
- ✅ PyTorch deep learning model (lightweight_traffic_model.pth)
- ✅ Google Maps-style traffic patterns
- ✅ Holiday calendar service
- ✅ Distance calculation service
- ✅ Gemini AI (optional, for enhanced predictions)

### Frontend:
- ✅ Interactive Mapbox map
- ✅ Location search with autocomplete
- ✅ Time/date picker (15-min increments)
- ✅ Real-time traffic predictions

---

## 🎯 Testing the App

### Test Backend Health:
Open in browser: http://localhost:8000/health

Should return:
```json
{
  "status": "healthy",
  "service": "traffic-predictor-api",
  "version": "4.1"
}
```

### Test Prediction:
1. Open `index.html` in browser
2. Click on the map to select a location
3. Choose time: **8:00 AM** (morning rush)
4. Choose day: **Monday** (weekday)
5. Click "Predict Traffic"
6. Should show: **~75% congestion** (RED)

### Test API Docs:
Open: http://localhost:8000/docs
- Interactive Swagger UI
- Test endpoints directly
- See request/response formats

---

## 🛑 Stopping the Server

```powershell
# In the terminal running the server, press:
Ctrl + C

# Or kill all Python processes:
taskkill /F /IM python.exe
```

---

## 📝 Configuration

### Change API URL (for deployment):
Edit `index.html` line 688:
```javascript
// For local development:
const API_URL = 'http://localhost:8000';

// For production (Render.com):
const API_URL = 'https://traffic-predictor-api.onrender.com';
```

### Change Server Port:
Edit `backend/main.py` line ~510:
```python
uvicorn.run(
    "main:app",
    host="0.0.0.0",
    port=8000,  # Change this number
    reload=False
)
```

---

## 🎉 Success!

Your traffic predictor is now running locally!

**Next Steps:**
1. Test different locations (Dallas, Austin, Houston)
2. Try different times (morning vs evening rush)
3. Test weekday vs weekend traffic
4. Check holiday predictions

**For Production Deployment:**
See `DEPLOYMENT_GUIDE.md` for hosting on Render.com, Railway, or other platforms.

---

## 🐛 Still Having Issues?

Check these files for details:
- `AI_MODELS_DOCUMENTATION.md` - All about the AI models
- `DEPLOYMENT_GUIDE.md` - Deployment options
- `QUICK_START_V4.md` - Quick start guide
- `backend/main.py` - Backend entry point
- `index.html` - Frontend code

---

**Last Updated**: October 5, 2025
**Version**: 4.1
**Status**: Ready for local development! 🚀
