"""
Agentuity Client
Handles communication with Agentuity workflow orchestration platform
"""

import os
import httpx
import uuid
from typing import Dict, Optional


async def trigger_simulation_workflow(
    prompt: str,
    user_id: str,
    location: Optional[Dict] = None
) -> str:
    """
    Trigger the Agentuity workflow for urban simulation
    
    This sends a request to Agentuity which will orchestrate:
    1. OrchestratorAgent: Parse prompt with Gemini
    2. SimulationAgent: Run MATLAB simulation
    3. ReportingAgent: Generate summary and audio
    
    Args:
        prompt: User's natural language prompt
        user_id: Auth0 user ID
        location: Optional dict with lat/lng
        
    Returns:
        str: Simulation ID for tracking
    """
    
    # Generate unique simulation ID
    simulation_id = str(uuid.uuid4())
    
    # Get Agentuity configuration
    agentuity_url = os.getenv("AGENTUITY_WEBHOOK_URL")
    agentuity_key = os.getenv("AGENTUITY_API_KEY")
    
    if not agentuity_url or not agentuity_key:
        print("⚠️  Agentuity not configured, running in mock mode")
        return simulation_id
    
    # Prepare payload
    payload = {
        "simulation_id": simulation_id,
        "user_id": user_id,
        "prompt": prompt,
        "location": location or {"lat": 32.7299, "lng": -97.1161},  # Default to Arlington, TX
        "timestamp": None  # Will be set by Agentuity
    }
    
    try:
        # Send to Agentuity webhook
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                agentuity_url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {agentuity_key}",
                    "Content-Type": "application/json"
                }
            )
            
            response.raise_for_status()
            
            print(f"✓ Triggered Agentuity workflow for simulation {simulation_id}")
            return simulation_id
            
    except httpx.HTTPError as e:
        print(f"✗ Failed to trigger Agentuity workflow: {e}")
        # Return the ID anyway - can still track it
        return simulation_id
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return simulation_id


async def get_simulation_status(simulation_id: str) -> Dict:
    """
    Check status of a simulation in Agentuity
    
    Args:
        simulation_id: The simulation ID to check
        
    Returns:
        Dict with status information
    """
    # TODO: Implement once Agentuity agents are deployed
    return {
        "simulation_id": simulation_id,
        "status": "processing",
        "progress": 50,
        "message": "Simulation in progress"
    }


# Mock functions for development without Agentuity
def mock_simulation_workflow(prompt: str, user_id: str) -> str:
    """
    Mock workflow for testing without Agentuity
    Returns a simulation ID immediately
    """
    simulation_id = str(uuid.uuid4())
    print(f"🔧 MOCK: Started simulation {simulation_id}")
    print(f"   User: {user_id}")
    print(f"   Prompt: {prompt}")
    return simulation_id


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Test the trigger function
    async def test():
        sim_id = await trigger_simulation_workflow(
            prompt="Close Cooper Street from 7 AM to 10 AM",
            user_id="test_user_123",
            location={"lat": 32.7299, "lng": -97.1161}
        )
        print(f"Created simulation: {sim_id}")
    
    asyncio.run(test())
