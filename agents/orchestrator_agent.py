"""
SimCity AI - Orchestrator Agent
First agent in the workflow - handles NLU and routing
"""

from typing import Dict, Any
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from agentuity import Agent, agent, resp
    AGENTUITY_AVAILABLE = True
except ImportError:
    print("⚠️  Agentuity not installed. This is a template for deployment.")
    AGENTUITY_AVAILABLE = False
    # Define dummy decorators for development
    def agent(name):
        def decorator(cls):
            return cls
        return decorator
    
    class Agent:
        pass
    
    class resp:
        @staticmethod
        def handoff(name, data):
            return {"handoff": name, "data": data}


# Import services (will use mocks if dependencies not installed)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from gemini_service import GeminiService
from osm_data_service import OSMDataService


@agent("OrchestratorAgent")
class OrchestratorAgent(Agent):
    """
    Orchestrator Agent - Entry point for simulation workflow
    
    Responsibilities:
    1. Receive user's natural language prompt
    2. Use Gemini API to parse prompt into structured parameters
    3. Fetch relevant road network data from OpenStreetMap
    4. Hand off to SimulationAgent
    """
    
    async def run(self, data: Dict[str, Any]):
        """
        Main execution method
        
        Args:
            data: Dict containing:
                - simulation_id: Unique ID for this simulation
                - user_id: Auth0 user ID
                - prompt: Natural language prompt from user
                - location: Dict with lat/lng
        """
        print("=" * 60)
        print("ORCHESTRATOR AGENT: Starting workflow")
        print("=" * 60)
        
        simulation_id = data.get("simulation_id")
        user_id = data.get("user_id")
        prompt = data.get("prompt")
        location = data.get("location", {"lat": 32.7299, "lng": -97.1161})
        
        print(f"Simulation ID: {simulation_id}")
        print(f"User: {user_id}")
        print(f"Prompt: {prompt}")
        
        try:
            # Step 1: Use Gemini to parse the natural language prompt
            print("\n[Step 1/3] Parsing prompt with Gemini API...")
            gemini_service = GeminiService()
            parsed_params = gemini_service.parse_prompt(prompt)
            
            print(f"✓ Parsed parameters:")
            print(f"  Action: {parsed_params.get('action')}")
            print(f"  Street: {parsed_params.get('parameters', {}).get('street_name', 'N/A')}")
            
            # Step 2: Fetch road network data from OpenStreetMap
            print("\n[Step 2/3] Fetching road network data...")
            osm_service = OSMDataService()
            
            # Fetch network for UT Arlington area (demo default)
            road_network = osm_service.get_ut_arlington_network()
            
            print(f"✓ Road network loaded:")
            print(f"  Nodes: {road_network.get('node_count')}")
            print(f"  Edges: {road_network.get('edge_count')}")
            
            # Step 3: Prepare data for SimulationAgent
            print("\n[Step 3/3] Handing off to SimulationAgent...")
            
            simulation_data = {
                "simulation_id": simulation_id,
                "user_id": user_id,
                "original_prompt": prompt,
                "parsed_params": parsed_params,
                "road_network": road_network,
                "location": location
            }
            
            # Hand off to SimulationAgent
            return resp.handoff(
                name="SimulationAgent",
                data=simulation_data
            )
            
        except Exception as e:
            print(f"\n✗ ORCHESTRATOR ERROR: {e}")
            
            # Return error response
            return {
                "status": "error",
                "simulation_id": simulation_id,
                "error": str(e),
                "stage": "orchestration"
            }


# For local testing
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Create mock agent instance
    orchestrator = OrchestratorAgent()
    
    # Test data
    test_data = {
        "simulation_id": "test-123",
        "user_id": "user-456",
        "prompt": "What happens if we close Cooper Street from 7 AM to 10 AM?",
        "location": {"lat": 32.7299, "lng": -97.1161}
    }
    
    # Run the agent
    async def test():
        result = await orchestrator.run(test_data)
        print("\n" + "=" * 60)
        print("ORCHESTRATOR RESULT:")
        print("=" * 60)
        print(result)
    
    asyncio.run(test())
