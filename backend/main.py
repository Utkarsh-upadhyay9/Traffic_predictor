"""
SimCity AI - Main FastAPI Application
Backend API for urban simulation digital twin
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

from auth_service import verify_token
from gemini_service import GeminiService
from agentuity_client import trigger_simulation_workflow

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="SimCity AI API",
    description="Urban simulation and traffic flow digital twin",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2 scheme for Auth0
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize services
gemini_service = GeminiService()


# Pydantic models
class SimulationRequest(BaseModel):
    """Request model for simulation endpoint"""
    prompt: str
    user_location: Optional[dict] = None
    

class SimulationResponse(BaseModel):
    """Response model for simulation endpoint"""
    status: str
    simulation_id: str
    message: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    services: dict


# Routes
@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "SimCity AI API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify all services"""
    services_status = {
        "auth0": "configured" if os.getenv("AUTH0_DOMAIN") else "not_configured",
        "gemini": "configured" if os.getenv("GEMINI_API_KEY") else "not_configured",
        "agentuity": "configured" if os.getenv("AGENTUITY_API_KEY") else "not_configured",
        "elevenlabs": "configured" if os.getenv("ELEVENLABS_API_KEY") else "not_configured",
        "matlab": "configured" if os.getenv("MATLAB_SERVICE_URL") else "not_configured",
    }
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": services_status
    }


@app.post("/api/simulation", response_model=SimulationResponse)
async def run_simulation(
    request: SimulationRequest,
    token: str = Depends(oauth2_scheme)
):
    """
    Main endpoint to trigger urban traffic simulation
    
    Requires Auth0 JWT token for authentication.
    Triggers Agentuity workflow that orchestrates:
    1. Gemini for NLU (parse prompt)
    2. MATLAB for simulation
    3. Gemini for summarization
    4. ElevenLabs for audio generation
    """
    try:
        # Verify Auth0 token
        payload = verify_token(token)
        user_id = payload.get("sub")
        
        print(f"User {user_id} requested simulation: {request.prompt}")
        
        # Trigger the Agentuity workflow
        simulation_id = await trigger_simulation_workflow(
            prompt=request.prompt,
            user_id=user_id,
            location=request.user_location
        )
        
        return {
            "status": "simulation_started",
            "simulation_id": simulation_id,
            "message": "Your simulation is being processed. Check back shortly for results."
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation failed: {str(e)}"
        )


@app.get("/api/simulation/{simulation_id}")
async def get_simulation_results(
    simulation_id: str,
    token: str = Depends(oauth2_scheme)
):
    """
    Get results of a completed simulation
    """
    try:
        # Verify Auth0 token
        payload = verify_token(token)
        user_id = payload.get("sub")
        
        # TODO: Implement result retrieval from database or Agentuity
        # For now, return placeholder
        return {
            "simulation_id": simulation_id,
            "status": "processing",
            "message": "Results will be available when simulation completes"
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {str(e)}",
        )


@app.get("/api/user/simulations")
async def get_user_simulations(token: str = Depends(oauth2_scheme)):
    """
    Get all simulations for the authenticated user
    """
    try:
        payload = verify_token(token)
        user_id = payload.get("sub")
        
        # TODO: Retrieve from database
        return {
            "user_id": user_id,
            "simulations": []
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
