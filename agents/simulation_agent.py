"""
SimCity AI - Simulation Agent
Executes MATLAB traffic simulations
"""

from typing import Dict, Any
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from agentuity import Agent, agent, resp
    AGENTUITY_AVAILABLE = True
except ImportError:
    print("⚠️  Agentuity not installed. This is a template for deployment.")
    AGENTUITY_AVAILABLE = False
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

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from matlab_service import MATLABSimulationService


@agent("SimulationAgent")
class SimulationAgent(Agent):
    """
    Simulation Agent - Executes traffic simulations
    
    Responsibilities:
    1. Receive parsed parameters and road network from OrchestratorAgent
    2. Run MATLAB traffic simulation
    3. Process and structure results
    4. Hand off to ReportingAgent
    """
    
    async def run(self, data: Dict[str, Any]):
        """
        Main execution method
        
        Args:
            data: Dict containing:
                - simulation_id: Unique ID
                - parsed_params: Structured simulation parameters
                - road_network: Road network data from OSM
        """
        print("=" * 60)
        print("SIMULATION AGENT: Running traffic simulation")
        print("=" * 60)
        
        simulation_id = data.get("simulation_id")
        parsed_params = data.get("parsed_params")
        road_network = data.get("road_network")
        
        print(f"Simulation ID: {simulation_id}")
        print(f"Action: {parsed_params.get('action')}")
        
        try:
            # Initialize MATLAB service
            print("\n[Step 1/2] Initializing MATLAB simulation engine...")
            matlab_service = MATLABSimulationService()
            
            # Run the simulation
            print("\n[Step 2/2] Running traffic simulation...")
            print("(This may take 1-3 minutes for complex scenarios)")
            
            simulation_results = matlab_service.run_traffic_simulation(
                road_network=road_network,
                scenario_params=parsed_params
            )
            
            print(f"\n✓ Simulation completed!")
            print(f"  Status: {simulation_results.get('status')}")
            print(f"  Travel time change: {simulation_results.get('metrics', {}).get('travel_time_change_pct', 0):.1f}%")
            print(f"  Congestion change: {simulation_results.get('metrics', {}).get('congestion_change_pct', 0):.1f}%")
            
            # Prepare data for ReportingAgent
            reporting_data = {
                "simulation_id": simulation_id,
                "user_id": data.get("user_id"),
                "original_prompt": data.get("original_prompt"),
                "parsed_params": parsed_params,
                "simulation_results": simulation_results
            }
            
            # Hand off to ReportingAgent
            print("\nHanding off to ReportingAgent...")
            return resp.handoff(
                name="ReportingAgent",
                data=reporting_data
            )
            
        except Exception as e:
            print(f"\n✗ SIMULATION ERROR: {e}")
            
            return {
                "status": "error",
                "simulation_id": simulation_id,
                "error": str(e),
                "stage": "simulation"
            }


# For local testing
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    
    load_dotenv()
    
    simulation_agent = SimulationAgent()
    
    # Test data (would come from OrchestratorAgent)
    test_data = {
        "simulation_id": "test-123",
        "user_id": "user-456",
        "original_prompt": "Close Cooper Street",
        "parsed_params": {
            "action": "CLOSE_ROAD",
            "parameters": {
                "street_name": "Cooper Street"
            },
            "description": "Simulate closure of Cooper Street"
        },
        "road_network": {
            "nodes": [{"id": "1", "lat": 32.73, "lng": -97.11}],
            "edges": [{"from": "1", "to": "2", "length": 1000}],
            "node_count": 1,
            "edge_count": 1
        }
    }
    
    async def test():
        result = await simulation_agent.run(test_data)
        print("\n" + "=" * 60)
        print("SIMULATION RESULT:")
        print("=" * 60)
        import json
        print(json.dumps(result, indent=2))
    
    asyncio.run(test())
