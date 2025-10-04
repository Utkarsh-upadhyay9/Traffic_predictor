# 🚀 Quick Start Guide - SimCity AI

## ✅ **CURRENT STATUS**

Your application is **ALREADY RUNNING**:
- ✅ Backend: http://localhost:8000 
- ✅ Frontend: http://localhost:3000
- ✅ API Docs: http://localhost:8000/docs

**All API keys configured!** ✅

---

## 🎯 **ONE-COMMAND STARTUP**

### **Method 1: Using run.ps1 (RECOMMENDED)**

```powershell
.\run.ps1
```

**What it does**:
- Opens Backend in a **separate window** (minimized)
- Shows Frontend logs in **current terminal**
- Auto-opens browser when ready
- Easy to monitor and debug

### **Method 2: Using start.bat**

```cmd
start.bat
```

**What it does**:
- Starts Backend in background
- Starts Frontend in current terminal
- Simple and fast

---

## 🛑 **HOW TO STOP**

### If using run.ps1:
1. **Press Ctrl+C** in the terminal (stops frontend)
2. **Close the backend window** (stops backend)

### If using start.bat:
1. **Press Ctrl+C** in the terminal (stops frontend)
2. Find backend window and close it

### Nuclear option (kill all):
```powershell
# Kill all Python processes (backend)
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

# Kill all Node processes (frontend)
Get-Process node -ErrorAction SilentlyContinue | Stop-Process -Force
```

---

## 🔄 **RESTART SERVERS**

### Full restart:
```powershell
# Stop everything first
Get-Process python,node -ErrorAction SilentlyContinue | Stop-Process -Force

# Wait a moment
Start-Sleep -Seconds 2

# Start fresh
.\run.ps1
```

### Restart just backend:
```powershell
# In backend terminal:
# Press Ctrl+C, then:
python backend/main.py
```

### Restart just frontend:
```powershell
# In frontend terminal:
# Press Ctrl+C, then:
cd frontend
npm start
```

---

## 📁 **AVAILABLE STARTUP SCRIPTS**

| Script | Description | Best For |
|--------|-------------|----------|
| **run.ps1** | Backend in separate window, frontend logs visible | **Development** ⭐ |
| **start.bat** | Simple batch startup | Quick testing |
| **start.ps1** | Original with separate windows | Alternative |
| **start-all.ps1** | Advanced with job management | Complex setups |

---

## ✅ **VERIFY IT'S WORKING**

### 1. Check Backend Health
```powershell
curl http://localhost:8000/health
```

**Expected**:
```json
{
  "status": "healthy",
  "services": {
    "gemini": "configured",
    "auth": "bypass_mode"
  }
}
```

### 2. Test ML Prediction
```powershell
curl "http://localhost:8000/api/predict?hour=8&road_closure=true&current_vehicle_count=1500"
```

**Expected**: JSON with travel time, congestion, vehicle count

### 3. Open Frontend
```
http://localhost:3000
```

**Expected**: See SimCity AI interface with:
- ✅ Prediction form
- ✅ Interactive map (Mapbox)
- ✅ Two modes: Single/Compare

### 4. Open API Docs
```
http://localhost:8000/docs
```

**Expected**: Swagger UI with all endpoints

---

## 🐛 **TROUBLESHOOTING**

### "Port 3000 already in use"

**Solution 1** - Kill the old process:
```powershell
Get-Process node -ErrorAction SilentlyContinue | Stop-Process -Force
```

**Solution 2** - Use different port:
When prompted, type `y` to run on different port (3001, 3002, etc.)

### "Port 8000 already in use"

```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
```

### "Virtual environment not found"

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

### "node_modules not found"

```powershell
cd frontend
npm install
cd ..
```

### "API keys not working"

1. Check `backend/.env` has your keys
2. Restart backend: `python backend/main.py`
3. Verify: `curl http://localhost:8000/health`

---

## 🎬 **DEMO WORKFLOW**

### Perfect Demo Startup:

1. **Clean slate**:
```powershell
Get-Process python,node -ErrorAction SilentlyContinue | Stop-Process -Force
```

2. **Start application**:
```powershell
.\run.ps1
```

3. **Wait for "Compiled successfully!"** message

4. **Open browser**: http://localhost:3000

5. **Demo flow**:
   - Show ML prediction (single mode)
   - Show scenario comparison
   - Show interactive map
   - Show API docs (Swagger)

---

## 💡 **PRO TIPS**

### Keep logs visible:
- Run `.\run.ps1` - frontend logs show in main terminal
- Backend window shows backend logs

### Quick testing:
```powershell
# Test prediction
curl "http://localhost:8000/api/predict?hour=17&num_lanes=3&current_vehicle_count=1500"

# Test comparison
curl -X POST "http://localhost:8000/api/compare" -H "Content-Type: application/json" -d '{\"baseline\":{\"hour\":17,\"num_lanes\":3},\"modified\":{\"hour\":17,\"num_lanes\":2}}'
```

### API Keys configured:
- ✅ Gemini: AIzaSy...bYk
- ✅ Mapbox: pk.eyJ...lfQLQ
- ✅ Auth0: 7ZP3Ku...Qr6k
- ✅ ElevenLabs: sk_94a4...

All in `backend/.env`!

---

## 📊 **WHAT'S RUNNING**

When you start with `.\run.ps1`:

**Backend (Separate Window)**:
- FastAPI server on :8000
- ML models loaded (3 models)
- Gemini AI connected
- Auth bypassed (dev mode)
- CORS enabled

**Frontend (Current Terminal)**:
- React dev server on :3000
- Webpack hot reload
- Mapbox GL maps
- Axios API client
- Auto-opens browser

---

## 🎉 **YOU'RE READY!**

**Everything is configured and working!**

**To start:**
```powershell
.\run.ps1
```

**To test:**
```
http://localhost:3000
```

**To present:**
1. Start with run.ps1
2. Wait for "Compiled successfully!"
3. Demo the features
4. Win HackUTA 7! 🏆

---

## 📚 **MORE HELP**

- `API_KEYS_STATUS.md` - API key configuration
- `API_KEYS_GUIDE.md` - Where to get keys
- `APP_RUNNING.md` - Detailed usage guide
- `FULL_APP_GUIDE.md` - Complete documentation

---

**Current Status**: ✅ **READY TO DEMO**
**Last Updated**: October 4, 2025
