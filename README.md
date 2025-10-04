# SimCity AI: Generative Urban Digital Twin

## 🏆 HackUTA 7 Project

**MetropolisAI** is an AI-powered urban simulation platform that generates realistic cities and optimizes traffic flow in real-time using cutting-edge AI and simulation technologies.

## 🎯 Project Overview

SimCity AI empowers urban planners with an intuitive, AI-driven tool for data-backed decision-making. Using natural language prompts, planners can simulate traffic scenarios, test infrastructure changes, and receive detailed insights before spending millions on real-world implementations.

## 🏗️ Architecture

```
Frontend (React + Mapbox) 
    ↓
Backend API (FastAPI on Arm) 
    ↓
Agentuity Orchestration Layer
    ↓
├─ OrchestratorAgent → Gemini API (NLU)
├─ SimulationAgent → MATLAB Engine (Traffic Sim)
└─ ReportingAgent → Gemini (Summary) + ElevenLabs (Audio)
```

## 🛠️ Technology Stack

| Technology | Role | Integration |
|------------|------|-------------|
| **Gemini API** | Natural Language Understanding & Content Generation | Parse user prompts, generate summaries |
| **Auth0** | User Authentication & Authorization | Secure login, user profiles, JWT validation |
| **Agentuity** | Agentic Workflow Orchestration | Manage async simulation pipeline |
| **MATLAB** | Core Traffic Simulation Engine | Automated Driving Toolbox simulations |
| **ElevenLabs** | Audio Reporting | Voice narration of simulation results |
| **Arm** | Deployment Platform | Run containers on Arm-based cloud (AWS Graviton) |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker with buildx
- MATLAB R2023b+ with Automated Driving Toolbox
- API Keys for: Gemini, Auth0, Agentuity, ElevenLabs

### Installation

```bash
# Clone the repository
git clone https://github.com/Utkarsh-upadhyay9/simcity-ai.git
cd simcity-ai

# Backend setup
cd backend
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys

# Frontend setup
cd ../frontend
npm install
cp .env.example .env.local
# Edit .env.local with your keys

# Run locally
cd ../backend && uvicorn main:app --reload &
cd ../frontend && npm run dev
```

## 📖 Usage

1. **Login**: Authenticate using Auth0
2. **Select Area**: View the interactive map of your city
3. **Describe Scenario**: Type in natural language: "Close I-30 westbound at Cooper Street from 7-10 AM"
4. **Wait for Simulation**: AI agents orchestrate the MATLAB simulation
5. **View Results**: See animated traffic flow, metrics dashboard, and listen to audio report

## 🎬 Demo Video

[Link to demo video]

## 📊 Project Structure

```
simcity-ai/
├── backend/          # FastAPI backend
├── frontend/         # React frontend
├── agents/           # Agentuity agent definitions
├── matlab/           # MATLAB simulation scripts
├── docker/           # Docker configurations
└── docs/             # Documentation
```

## 🏅 Prize Categories

- ✅ Best Use of Gemini API
- ✅ Best Use of Auth0
- ✅ Best Use of Agentuity
- ✅ Best Use of MATLAB
- ✅ Best Use of ElevenLabs
- ✅ Best Use of Arm

## 👥 Team

**Utkarsh Upadhyay**
- GitHub: [@Utkarsh-upadhyay9](https://github.com/Utkarsh-upadhyay9)
- Email: utkars95@gmail.com

## 📄 License

MIT License

## 🙏 Acknowledgments

Built at HackUTA 7 - October 2025  
Created by Utkarsh Upadhyay
