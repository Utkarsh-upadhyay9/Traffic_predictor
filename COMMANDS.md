# SimCity AI - Quick Reference Commands

## 🚀 Most Common Commands

### Setup
```powershell
# Quick start (Windows)
.\quick-start.ps1

# Manual setup
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
copy backend\.env.example backend\.env
```

### Testing
```powershell
# Test all services
python test_setup.py

# Test individual services
cd backend
python gemini_service.py
python osm_data_service.py
python matlab_service.py
python auth_service.py

# Test agents
cd agents
python orchestrator_agent.py
python simulation_agent.py
python reporting_agent.py
```

### Running Backend
```powershell
# Start backend server
cd backend
python main.py

# Or with uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Access API docs
# Open: http://localhost:8000/docs
```

### Agentuity Deployment
```powershell
# Install CLI
pip install agentuity-cli

# Login
agentuity login

# Deploy agents
cd agents
agentuity create simcity-ai
agentuity deploy orchestrator_agent.py --name OrchestratorAgent
agentuity deploy simulation_agent.py --name SimulationAgent
agentuity deploy reporting_agent.py --name ReportingAgent

# View logs
agentuity logs OrchestratorAgent --follow
agentuity logs SimulationAgent --follow
agentuity logs ReportingAgent --follow

# List deployments
agentuity list
```

### Docker Commands
```powershell
# Build for Arm
cd docker
docker buildx build --platform linux/arm64 -t simcity-api -f Dockerfile.backend ../backend

# Run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop containers
docker-compose down
```

### Git Commands
```powershell
# Initialize repo (if not done)
git init
git add .
git commit -m "Initial commit: Complete SimCity AI backend"

# Push to GitHub
git remote add origin https://github.com/yourusername/simcity-ai.git
git branch -M main
git push -u origin main

# Create release
git tag -a v1.0.0 -m "HackUTA 7 submission"
git push origin v1.0.0
```

---

## 🔑 API Endpoints Quick Reference

### Health Check
```powershell
curl http://localhost:8000/health
```

### Start Simulation (With Auth)
```powershell
# Get token from Auth0 first, then:
curl -X POST http://localhost:8000/api/simulation `
  -H "Authorization: Bearer YOUR_TOKEN_HERE" `
  -H "Content-Type: application/json" `
  -d '{"prompt": "Close Cooper Street from 7 AM to 10 AM"}'
```

### Start Simulation (Dev Mode - No Auth)
```powershell
# If SKIP_AUTH_VERIFICATION=true in .env
curl -X POST http://localhost:8000/api/simulation `
  -H "Authorization: Bearer fake_token" `
  -H "Content-Type: application/json" `
  -d '{"prompt": "Close Cooper Street from 7 AM to 10 AM"}'
```

### Get Simulation Results
```powershell
curl -X GET http://localhost:8000/api/simulation/SIMULATION_ID `
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

---

## 🐛 Troubleshooting Commands

### Check Python Version
```powershell
python --version
# Should be 3.10 or higher
```

### Check Installed Packages
```powershell
pip list
pip show fastapi
pip show google-generativeai
```

### Reinstall Dependencies
```powershell
pip install --upgrade -r backend\requirements.txt
```

### Check MATLAB Engine
```powershell
python -c "import matlab.engine; print('MATLAB Engine installed')"
```

### Test Gemini API
```powershell
python -c "import google.generativeai as genai; print('Gemini SDK installed')"
```

### Check Port Availability
```powershell
netstat -ano | findstr :8000
```

### Kill Process on Port 8000 (if stuck)
```powershell
# Find PID from netstat command above, then:
taskkill /PID <PID> /F
```

---

## 📊 Environment Variables Quick Reference

### Required (Add to backend\.env)
```env
GEMINI_API_KEY=your_key_here
AUTH0_DOMAIN=your-tenant.us.auth0.com
AUTH0_API_AUDIENCE=https://simcity-ai-api
ELEVENLABS_API_KEY=your_key_here
```

### Optional
```env
AGENTUITY_API_KEY=your_key_here
AGENTUITY_WEBHOOK_URL=https://your-endpoint.com/webhook
MATLAB_SERVICE_URL=http://localhost:8001
```

### Development Mode
```env
SKIP_AUTH_VERIFICATION=true
DEBUG=True
ENVIRONMENT=development
```

---

## 🎬 Demo Commands

### Quick Demo Flow
```powershell
# 1. Start backend
cd backend
python main.py

# 2. In another terminal, test API
curl http://localhost:8000/health

# 3. Test simulation (Swagger UI is easier)
# Open: http://localhost:8000/docs
# Click "POST /api/simulation"
# Click "Try it out"
# Enter prompt: "Close Cooper Street"
# Execute
```

### Show Gemini Parsing
```powershell
cd backend
python
>>> from gemini_service import GeminiService
>>> service = GeminiService()
>>> result = service.parse_prompt("Close Cooper Street from 7 AM to 10 AM")
>>> print(result)
```

### Show MATLAB Simulation
```powershell
cd backend
python
>>> from matlab_service import MATLABSimulationService
>>> service = MATLABSimulationService()
>>> result = service.run_traffic_simulation(
...     {"nodes": [], "edges": []},
...     {"action": "CLOSE_ROAD", "parameters": {"street_name": "Cooper St"}}
... )
>>> print(result)
```

### Show Agent Flow
```powershell
cd agents
python orchestrator_agent.py
# Shows full workflow simulation
```

---

## 📸 Screenshot Locations

### For Documentation
```powershell
# API Docs
http://localhost:8000/docs

# Health Check
http://localhost:8000/health

# Agentuity Dashboard
https://app.agentuity.com/

# Auth0 Dashboard
https://manage.auth0.com/

# AWS Console (for Arm deployment)
https://console.aws.amazon.com/
```

---

## 🎥 Demo Video Script

### Recording Setup
```powershell
# Use OBS Studio or Windows Game Bar
# Windows + G to start recording
# Or: OBS Studio (free download)
```

### Demo Sequence (3 minutes)
1. **[0:00-0:15]** Show landing page, click login (Auth0)
2. **[0:15-0:30]** Enter prompt: "What if we close Cooper Street during construction?"
3. **[0:30-0:45]** Show Gemini parsing in console/logs
4. **[0:45-1:15]** Show MATLAB simulation running (or animation)
5. **[1:15-1:45]** Show Agentuity dashboard with agent handoffs
6. **[1:45-2:15]** Display results: map, metrics, heatmap
7. **[2:15-2:45]** Play ElevenLabs audio narration
8. **[2:45-3:00]** Show tech stack slide, call to action

---

## 💾 Backup Commands

### Save Your Work
```powershell
# Backup to USB
xcopy /E /I /Y c:\Users\utkar\Desktop\Xapps\Digi_sim D:\backup\Digi_sim

# Create zip
Compress-Archive -Path . -DestinationPath simcity-ai-backup.zip

# Export environment
pip freeze > requirements-full.txt
```

### Restore If Needed
```powershell
# From git
git clone https://github.com/yourusername/simcity-ai.git
cd simcity-ai
.\quick-start.ps1

# From zip
Expand-Archive simcity-ai-backup.zip
cd simcity-ai
.\quick-start.ps1
```

---

## 🏆 Submission Checklist Commands

### Pre-Submission Verification
```powershell
# 1. Run all tests
python test_setup.py

# 2. Check git status
git status
git log --oneline -5

# 3. Verify .env not committed
git ls-files | findstr .env
# Should return nothing

# 4. Check all files are committed
git add .
git commit -m "Final submission"
git push

# 5. Create release tag
git tag -a v1.0.0 -m "HackUTA 7 Final Submission"
git push origin v1.0.0
```

---

## 🚨 Emergency Commands

### If Server Won't Start
```powershell
# Kill all Python processes
taskkill /F /IM python.exe

# Clear Python cache
Get-ChildItem -Include __pycache__ -Recurse -Force | Remove-Item -Force -Recurse

# Restart from scratch
.\quick-start.ps1
```

### If Git Issues
```powershell
# Reset to last commit
git reset --hard HEAD

# Force push (use carefully!)
git push --force origin main
```

### If Dependencies Broken
```powershell
# Remove and recreate venv
Remove-Item -Recurse -Force venv
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

---

## 📱 Contact & Resources

### Quick Links
- **Project Repo**: [Add GitHub URL]
- **Demo Video**: [Add YouTube URL]
- **Devpost**: [Add Devpost URL]
- **Live Demo**: [Add deployment URL]

### Documentation
- `README.md` - Start here
- `SETUP.md` - Setup instructions
- `ARCHITECTURE.md` - Technical details
- `PRIZES.md` - Prize integrations
- `CHECKLIST.md` - Task tracker
- `STATUS.md` - Current status

---

**Pro Tip**: Keep this file open in a tab during the hackathon for quick reference!
