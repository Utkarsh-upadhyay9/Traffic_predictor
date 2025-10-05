# 🚀 Deployment Guide - Traffic Predictor v4.1

## Hosting Your Traffic Predictor on the Web

---

## 🌐 Option 1: GitHub Pages (Frontend Only - Static Demo)

### Best for: Showcasing the UI and map

**Steps:**
```bash
# 1. Create a gh-pages branch
git checkout -b gh-pages

# 2. Modify index.html to use a public API or mock data
# (GitHub Pages only serves static files, no Python backend)

# 3. Push to GitHub
git push origin gh-pages

# 4. Enable GitHub Pages
# Go to: Settings → Pages → Source: gh-pages branch
```

**Access at**: `https://utkarsh-upadhyay9.github.io/Traffic_predictor/`

**Limitations**: No backend, would need to use mock data or connect to a hosted API

---

## 🐳 Option 2: Render.com (Full Stack - RECOMMENDED)

### Best for: Free hosting with Python backend + static frontend

**Steps:**

### Backend (Python/FastAPI):
1. Go to [render.com](https://render.com)
2. Sign up with GitHub
3. Click "New +" → "Web Service"
4. Connect your `Traffic_predictor` repo
5. Configure:
   ```
   Name: traffic-predictor-api
   Runtime: Python 3
   Build Command: pip install -r backend/requirements.txt
   Start Command: cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT
   ```
6. Click "Create Web Service"

**Backend URL**: `https://traffic-predictor-api.onrender.com`

### Frontend (Static Site):
1. Click "New +" → "Static Site"
2. Connect same repo
3. Configure:
   ```
   Name: traffic-predictor
   Build Command: echo "No build needed"
   Publish Directory: .
   ```
4. Update `index.html` line ~591:
   ```javascript
   const API_URL = 'https://traffic-predictor-api.onrender.com';
   ```

**Frontend URL**: `https://traffic-predictor.onrender.com`

**Cost**: FREE (with cold starts)

---

## ⚡ Option 3: Railway.app (Full Stack - Fast & Easy)

### Best for: Quick deployment, no configuration

**Steps:**
1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select `Traffic_predictor`
5. Railway auto-detects Python and deploys!
6. Add environment variables if needed
7. Update frontend API_URL to Railway backend URL

**Cost**: $5/month (500 hours free trial)

---

## 🔥 Option 4: Vercel (Frontend) + Railway (Backend)

### Best for: Production-grade performance

### Backend on Railway:
```bash
# Same as Option 3 for backend
```

### Frontend on Vercel:
1. Go to [vercel.com](https://vercel.com)
2. Import `Traffic_predictor` repo
3. Configure:
   ```
   Framework Preset: Other
   Build Command: (leave empty)
   Output Directory: .
   ```
4. Add environment variable:
   ```
   NEXT_PUBLIC_API_URL=https://your-railway-backend-url
   ```
5. Update `index.html` to use `process.env.NEXT_PUBLIC_API_URL`

**URLs**:
- Frontend: `https://traffic-predictor.vercel.app`
- Backend: `https://your-backend.railway.app`

---

## 🐋 Option 5: Docker + Any Cloud (AWS, Azure, GCP)

### Best for: Professional deployment, scalability

**Create `Dockerfile` in project root:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY backend/ ./backend/
COPY ml/ ./ml/
COPY index.html .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and deploy:**
```bash
# Build
docker build -t traffic-predictor .

# Run locally
docker run -p 8000:8000 traffic-predictor

# Deploy to cloud
# AWS: Elastic Beanstalk, ECS, or App Runner
# Azure: Container Instances or App Service
# GCP: Cloud Run
```

---

## 📊 Recommended Approach (FREE)

### For Hackathon Demo: **Render.com**

1. **Backend on Render**: Free Python web service
2. **Frontend on Render Static**: Free static hosting
3. **Total Cost**: $0/month
4. **Setup Time**: 10 minutes

**Trade-offs**:
- Cold starts (15-30 seconds on first request)
- Sleep after 15 min inactivity
- Perfect for demos and portfolios!

---

## 🔧 Quick Setup for Render

### Step 1: Create `render.yaml` (Optional but recommended)
```yaml
services:
  - type: web
    name: traffic-predictor-api
    runtime: python
    buildCommand: pip install -r backend/requirements.txt
    startCommand: cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: GEMINI_API_KEY
        value: AIzaSyDQBIe6HG58TMe_HYdW6F9AShugLYWYbYk

  - type: web
    name: traffic-predictor-frontend
    runtime: static
    buildCommand: echo "Static site"
    staticPublishPath: .
```

### Step 2: Update `index.html`
Change line ~591 from:
```javascript
const API_URL = 'http://localhost:8000';
```
To:
```javascript
const API_URL = 'https://traffic-predictor-api.onrender.com';
```

### Step 3: Push to GitHub
```bash
git add .
git commit -m "Add Render deployment configuration"
git push origin main
```

### Step 4: Deploy on Render
1. Go to render.com
2. Connect GitHub repo
3. Click "Apply" on the render.yaml
4. Wait 5-10 minutes for deployment

**Done!** Your app is live! 🎉

---

## 🎯 Post-Deployment Checklist

- [ ] Backend health check works: `https://your-api/health`
- [ ] Frontend loads map correctly
- [ ] API calls succeed (check browser console)
- [ ] Predictions return realistic congestion values
- [ ] Test on mobile device
- [ ] Share link with hackathon judges! 🏆

---

## 📝 Environment Variables Needed

```env
# Backend (.env file)
GEMINI_API_KEY=your_key_here  # Optional (for Gemini features)
GOOGLE_MAPS_API_KEY=your_key  # Optional (for accurate distances)
MAPBOX_TOKEN=your_token       # Already in index.html (public token OK)
```

---

## 🆘 Troubleshooting

### "Backend not responding"
- Check if backend service is running on Render
- Look at logs in Render dashboard
- Verify API_URL in index.html matches backend URL

### "Map not loading"
- Check Mapbox token in index.html (line ~595)
- Verify CORS is enabled in backend (already set to allow all origins)

### "Models not loading"
- Models are excluded from Git (too large)
- Either:
  1. Train models locally and commit to Git LFS
  2. OR use the sklearn fallback models (automatically used)

---

## 🎉 Success!

Once deployed, share your live app:

**Live Demo**: `https://your-app.onrender.com`  
**API Docs**: `https://your-api.onrender.com/docs`  
**GitHub**: `https://github.com/Utkarsh-upadhyay9/Traffic_predictor`

Happy hosting! 🚀
