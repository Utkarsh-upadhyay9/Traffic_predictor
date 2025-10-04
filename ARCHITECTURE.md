# SimCity AI - Architecture Documentation

## System Overview

SimCity AI is a distributed, agentic system for urban traffic simulation. It leverages multiple AI services orchestrated through Agentuity to provide an intuitive natural language interface for city planning.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend                            │
│                   (React + Mapbox GL)                       │
│  - Auth0 Login UI                                           │
│  - Interactive Map Visualization                            │
│  - Simulation Dashboard                                     │
│  - Audio Player (ElevenLabs)                                │
└────────────┬────────────────────────────────────────────────┘
             │ HTTPS/WebSocket
             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend API (FastAPI)                    │
│                     Running on Arm                          │
│  - JWT Validation (Auth0)                                   │
│  - Request Router                                           │
│  - WebSocket Manager                                        │
└────────────┬────────────────────────────────────────────────┘
             │ HTTPS Webhook
             ▼
┌─────────────────────────────────────────────────────────────┐
│                   Agentuity Cloud                           │
│                                                             │
│  ┌───────────────────────────────────────────────────┐     │
│  │        OrchestratorAgent (Entry Point)            │     │
│  │  - Receives user prompt                           │     │
│  │  - Calls Gemini API for NLU                       │     │
│  │  - Fetches OSM road network                       │     │
│  └───────────────┬───────────────────────────────────┘     │
│                  │ resp.handoff()                          │
│                  ▼                                          │
│  ┌───────────────────────────────────────────────────┐     │
│  │           SimulationAgent (Compute)               │     │
│  │  - Receives structured parameters                 │     │
│  │  - Calls MATLAB Service via HTTP                  │     │
│  │  - Waits for long-running simulation              │     │
│  └───────────────┬───────────────────────────────────┘     │
│                  │ resp.handoff()                          │
│                  ▼                                          │
│  ┌───────────────────────────────────────────────────┐     │
│  │         ReportingAgent (Finalization)             │     │
│  │  - Calls Gemini for summarization                 │     │
│  │  - Generates audio via ElevenLabs                 │     │
│  │  - Returns complete results package               │     │
│  └───────────────────────────────────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘

External Services:
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Gemini API  │  │ ElevenLabs   │  │ OpenStreetMap│
│  (Google)    │  │    (TTS)     │  │   (OSMnx)    │
└──────────────┘  └──────────────┘  └──────────────┘

Compute Backend:
┌─────────────────────────────────────────────────────────────┐
│              MATLAB Simulation Service                      │
│              (Docker Container on Arm)                      │
│  - MATLAB Engine API (Python)                               │
│  - Automated Driving Toolbox                                │
│  - Optimization Toolbox                                     │
│  - Traffic simulation algorithms                            │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. User Request Flow

```
User types prompt → Frontend → Backend API → Agentuity Webhook
                                                    ↓
                                          OrchestratorAgent
```

### 2. Orchestration Flow

```
OrchestratorAgent
    ├─→ Gemini API (Parse prompt to JSON)
    ├─→ OSMnx (Fetch road network)
    └─→ handoff to SimulationAgent
```

### 3. Simulation Flow

```
SimulationAgent
    ├─→ HTTP POST to MATLAB Service
    ├─→ Wait for simulation (async)
    ├─→ Receive results
    └─→ handoff to ReportingAgent
```

### 4. Reporting Flow

```
ReportingAgent
    ├─→ Gemini API (Summarize results)
    ├─→ ElevenLabs API (Generate audio)
    ├─→ Package final results
    └─→ Return to user (via WebSocket or polling)
```

## Technology Integration Details

### 1. Gemini API Integration

**Purpose**: Natural Language Understanding and Content Generation

**Integration Points**:
- `OrchestratorAgent`: Parse user prompts into structured JSON
- `ReportingAgent`: Generate natural language summaries

**Implementation**:
```python
from gemini_service import GeminiService

service = GeminiService()
parsed = service.parse_prompt("Close Cooper Street from 7-10 AM")
# Returns: {"action": "CLOSE_ROAD", "parameters": {...}}

summary = service.summarize_simulation_results(sim_data)
# Returns: Natural language summary text
```

**Why This Wins**: 
- Multimodal capabilities (text + images for satellite analysis)
- Complex reasoning for urban planning
- JSON mode for structured outputs

### 2. Auth0 Integration

**Purpose**: Secure Authentication and User Management

**Integration Points**:
- Frontend: Login UI and JWT acquisition
- Backend: JWT validation middleware

**Implementation**:
```python
from auth_service import verify_token

@app.post("/api/simulation")
async def run_simulation(token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)  # Validates JWT
    user_id = payload.get("sub")
    # ... proceed with authenticated request
```

**Why This Wins**:
- Professional authentication flow
- User-specific simulation history
- Role-based access control ready

### 3. Agentuity Integration

**Purpose**: Orchestrate complex, asynchronous workflows

**Integration Points**:
- All three agents (Orchestrator, Simulation, Reporting)
- Agent-to-agent handoffs using `resp.handoff()`

**Implementation**:
```python
from agentuity import Agent, agent, resp

@agent("OrchestratorAgent")
class OrchestratorAgent(Agent):
    async def run(self, data):
        # ... process data
        return resp.handoff(
            name="SimulationAgent",
            data=processed_data
        )
```

**Why This Wins**:
- Demonstrates true agentic architecture
- Handles long-running tasks elegantly
- Cloud deployment with monitoring

### 4. MATLAB Integration

**Purpose**: High-performance traffic simulation engine

**Integration Points**:
- `SimulationAgent` calls MATLAB Service
- Python ↔ MATLAB via Engine API

**Implementation**:
```python
import matlab.engine

eng = matlab.engine.start_matlab()
results = eng.runTrafficSimulation(
    road_network,
    scenario_params,
    nargout=1
)
```

**MATLAB Functions**:
- `runTrafficSimulation.m`: Main simulation loop
- `optimizeSignals.m`: Signal timing optimization

**Why This Wins**:
- Uses Automated Driving Toolbox (specialized tool)
- Optimization Toolbox for signal timing
- Professional computational approach

### 5. ElevenLabs Integration

**Purpose**: Generate natural voice narration of results

**Integration Points**:
- `ReportingAgent` generates audio from summary text

**Implementation**:
```python
from elevenlabs import ElevenLabs

client = ElevenLabs(api_key=api_key)
audio = client.text_to_speech.convert(
    voice_id=voice_id,
    text=summary_text,
    model_id="eleven_turbo_v2"
)
```

**Why This Wins**:
- Real-time narration of simulation events
- Accessibility feature for planners
- Professional polish for demo

### 6. Arm Architecture

**Purpose**: Deploy on efficient, cost-effective infrastructure

**Integration**:
- All backend services containerized for `linux/arm64`
- Deploy to AWS Graviton instances

**Implementation**:
```dockerfile
FROM --platform=linux/arm64 python:3.10-slim
# ... build application
```

```powershell
docker buildx build --platform linux/arm64 -t simcity-api .
```

**Why This Wins**:
- Demonstrates modern cloud-native architecture
- Better performance/cost ratio
- Professional deployment strategy

## API Endpoints

### Backend API (FastAPI)

| Endpoint | Method | Purpose | Auth Required |
|----------|--------|---------|---------------|
| `/` | GET | API info | No |
| `/health` | GET | Health check | No |
| `/api/simulation` | POST | Start simulation | Yes (JWT) |
| `/api/simulation/{id}` | GET | Get simulation results | Yes (JWT) |
| `/api/user/simulations` | GET | List user's simulations | Yes (JWT) |

### Request/Response Examples

**Start Simulation**:
```json
POST /api/simulation
Authorization: Bearer <jwt_token>

{
  "prompt": "Close Cooper Street from 7 AM to 10 AM",
  "user_location": {
    "lat": 32.7299,
    "lng": -97.1161
  }
}

Response:
{
  "status": "simulation_started",
  "simulation_id": "abc-123-def",
  "message": "Your simulation is being processed..."
}
```

**Get Results**:
```json
GET /api/simulation/abc-123-def
Authorization: Bearer <jwt_token>

Response:
{
  "status": "completed",
  "simulation_id": "abc-123-def",
  "summary": {
    "text": "Closing Cooper Street increased...",
    "audio_url": "/audio/narration_abc-123.mp3"
  },
  "metrics": {
    "baseline_travel_time_min": 15.0,
    "new_travel_time_min": 21.0,
    "travel_time_change_pct": -40.0
  }
}
```

## Data Models

### Simulation Request
```python
class SimulationRequest(BaseModel):
    prompt: str
    user_location: Optional[dict] = None
```

### Parsed Parameters (from Gemini)
```json
{
  "action": "CLOSE_ROAD",
  "parameters": {
    "street_name": "Cooper Street"
  },
  "time_window": {
    "start": "07:00",
    "end": "10:00"
  },
  "description": "Simulate closure of Cooper Street"
}
```

### Road Network (from OSM)
```json
{
  "nodes": [
    {
      "id": "123",
      "lat": 32.7299,
      "lng": -97.1161,
      "street_count": 3
    }
  ],
  "edges": [
    {
      "from": "123",
      "to": "456",
      "length": 220,
      "highway": "residential",
      "name": "Cooper Street",
      "lanes": "2"
    }
  ]
}
```

### Simulation Results
```json
{
  "status": "completed",
  "metrics": {
    "baseline_travel_time_min": 15.0,
    "new_travel_time_min": 21.0,
    "travel_time_change_pct": -40.0,
    "congestion_change_pct": -28.0,
    "affected_vehicles": 1500
  },
  "recommendations": [
    "Consider alternative routes",
    "Monitor peak hours"
  ],
  "congestion_heatmap": [...],
  "vehicle_trajectories": [...]
}
```

## Deployment Architecture

### Development Environment
- Backend: `uvicorn` on localhost:8000
- MATLAB: Local MATLAB installation or mock mode
- Agents: Run locally for testing

### Production Environment
- **Backend API**: AWS Graviton EC2 instance (Arm)
- **MATLAB Service**: Docker container on Arm
- **Agents**: Deployed to Agentuity Cloud
- **Frontend**: Vercel or Netlify
- **Database**: PostgreSQL on AWS RDS (for storing simulation history)

### Scaling Considerations
- **Horizontal scaling**: Multiple backend API instances behind load balancer
- **Queue system**: Add Redis for job queuing if MATLAB simulations take too long
- **Caching**: Cache OSM network data to reduce API calls
- **CDN**: Serve audio files from S3 + CloudFront

## Security

### Authentication Flow
1. User logs in via Auth0 (frontend)
2. Auth0 returns JWT token
3. Frontend includes token in `Authorization: Bearer <token>` header
4. Backend validates JWT signature and claims
5. Requests with valid tokens proceed

### Data Protection
- All API communication over HTTPS
- JWT tokens expire after 24 hours
- Sensitive data (API keys) in environment variables
- User simulations isolated by user_id

## Monitoring & Logging

### What to Monitor
- API response times
- MATLAB simulation duration
- Agentuity agent execution status
- Error rates
- API quota usage (Gemini, ElevenLabs)

### Logging Strategy
- Structured JSON logs
- Log levels: INFO, WARNING, ERROR
- Include correlation IDs (simulation_id) for tracing

## Future Enhancements

### Phase 2 Features
- Real-time WebSocket updates during simulation
- Comparison mode (A/B scenarios)
- Historical data analysis
- Satellite image analysis with Gemini Vision

### Phase 3 Features
- Machine learning predictions
- Multi-city support
- Collaborative planning (multiple users)
- Export reports (PDF, PPT)

## Troubleshooting Guide

See [SETUP.md](./SETUP.md) for detailed troubleshooting steps.

## Contributing

This is a hackathon project. For questions, contact the team.
