# SimCity AI - Setup Guide

## 🚀 Quick Start (Development Mode)

This guide will help you get SimCity AI running on your local machine for the hackathon.

## Prerequisites

### Required Software
- Python 3.10 or higher
- Node.js 18 or higher
- Git
- (Optional) MATLAB R2023b+ with Automated Driving Toolbox

### Required API Keys
You'll need accounts and API keys for:
1. **Google Gemini API** - https://makersuite.google.com/app/apikey
2. **Auth0** - https://auth0.com/ (Free tier)
3. **Agentuity** - https://agentuity.com/
4. **ElevenLabs** - https://elevenlabs.io/
5. (Optional) **AWS Account** for Arm deployment

## Step 1: Clone and Setup Backend

```powershell
# Navigate to project directory
cd c:\Users\utkar\Desktop\Xapps\Digi_sim

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install Python dependencies
cd backend
pip install -r requirements.txt

# Copy environment template
copy .env.example .env

# Edit .env file with your API keys
notepad .env
```

### Configure .env File

Open `.env` and add your API keys:

```env
# Auth0 - Create app at https://manage.auth0.com/
AUTH0_DOMAIN=your-tenant.us.auth0.com
AUTH0_API_AUDIENCE=https://simcity-ai-api
AUTH0_ISSUER=https://your-tenant.us.auth0.com/
AUTH0_ALGORITHMS=RS256

# For development, you can temporarily skip auth verification
SKIP_AUTH_VERIFICATION=true

# Gemini API - Get key at https://makersuite.google.com/
GEMINI_API_KEY=your_gemini_api_key_here

# Agentuity - Get from https://agentuity.com/
AGENTUITY_API_KEY=your_agentuity_key_here
AGENTUITY_WEBHOOK_URL=https://your-agentuity-endpoint.com/webhook

# ElevenLabs - Get from https://elevenlabs.io/
ELEVENLABS_API_KEY=your_elevenlabs_key_here
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM

# MATLAB Service
MATLAB_SERVICE_URL=http://localhost:8001

# Development settings
ENVIRONMENT=development
DEBUG=True
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

## Step 2: Test Backend Services

```powershell
# Test individual services
cd backend

# Test Gemini integration
python gemini_service.py

# Test OSM data fetching
python osm_data_service.py

# Test MATLAB service (will use mock if MATLAB not installed)
python matlab_service.py

# Test Auth0 (will show configuration)
python auth_service.py
```

## Step 3: Run Backend API

```powershell
cd backend

# Run the FastAPI server
python main.py

# Or using uvicorn directly
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Step 4: Test Agents Locally

```powershell
cd agents

# Test OrchestratorAgent
python orchestrator_agent.py

# Test SimulationAgent
python simulation_agent.py

# Test ReportingAgent
python reporting_agent.py
```

## Step 5: Deploy to Agentuity (Optional)

### Install Agentuity CLI

```powershell
pip install agentuity-cli
```

### Deploy Agents

```powershell
cd agents

# Login to Agentuity
agentuity login

# Create new project
agentuity create simcity-ai

# Deploy each agent
agentuity deploy orchestrator_agent.py --name OrchestratorAgent
agentuity deploy simulation_agent.py --name SimulationAgent
agentuity deploy reporting_agent.py --name ReportingAgent

# View logs
agentuity logs OrchestratorAgent --follow
```

## Step 6: MATLAB Setup (Optional but Recommended)

### If you have MATLAB installed:

1. **Install MATLAB Engine for Python**:
   ```powershell
   cd "C:\Program Files\MATLAB\R2023b\extern\engines\python"
   python setup.py install
   ```

2. **Test MATLAB Engine**:
   ```powershell
   python
   >>> import matlab.engine
   >>> eng = matlab.engine.start_matlab()
   >>> eng.sqrt(4.0)
   2.0
   >>> eng.quit()
   ```

3. **Add MATLAB scripts to path**:
   The `matlab_service.py` automatically adds the `matlab/` directory to MATLAB's path.

### If you don't have MATLAB:

The system will automatically use **mock mode** which generates realistic fake data. This is perfectly fine for the hackathon demo!

## Step 7: Frontend Setup (Coming Soon)

The frontend will be created in the next phase. For now, you can test the API using:

1. **Swagger UI**: http://localhost:8000/docs
2. **cURL**:
   ```powershell
   curl -X POST http://localhost:8000/api/simulation `
     -H "Content-Type: application/json" `
     -H "Authorization: Bearer fake_token_for_dev" `
     -d '{"prompt": "Close Cooper Street from 7 AM to 10 AM"}'
   ```

## Troubleshooting

### Python Import Errors

If you see "Import X could not be resolved":
```powershell
# Make sure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

### MATLAB Engine Not Found

This is normal if MATLAB isn't installed. The system will use mock mode automatically.

### Auth0 Errors

For development, set `SKIP_AUTH_VERIFICATION=true` in `.env` to bypass authentication temporarily.

### Port Already in Use

If port 8000 is busy:
```powershell
# Use a different port
uvicorn main:app --reload --port 8001
```

## Development Workflow

### Recommended Development Order:

1. ✅ **Backend API** - Get the core API running (YOU ARE HERE)
2. **Frontend** - Build React UI with map visualization
3. **Agent Integration** - Deploy to Agentuity and test workflow
4. **MATLAB Enhancement** - Add real traffic simulation
5. **Polish** - Add features, improve UI, test end-to-end

### Testing Strategy:

1. **Unit Tests**: Test each service independently
2. **Integration Tests**: Test agent workflow
3. **End-to-End**: Test full user journey from prompt to results

## Next Steps

1. ✅ Verify all services are working
2. Create frontend application
3. Deploy agents to Agentuity
4. Integrate ElevenLabs audio
5. Add real-time visualization
6. Prepare demo and presentation

## Resources

- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Gemini API**: https://ai.google.dev/
- **Auth0 Python**: https://auth0.com/docs/quickstart/backend/python
- **Agentuity Docs**: https://docs.agentuity.com/
- **MATLAB Automated Driving**: https://www.mathworks.com/help/driving/
- **OSMnx**: https://osmnx.readthedocs.io/

## Support

If you encounter issues:
1. Check the console output for error messages
2. Verify all API keys are correctly configured
3. Check that services are running on correct ports
4. Review the logs in the terminal

Good luck with the hackathon! 🚀
