"""
SimCity AI - Quick Test Script
Tests all services independently to verify setup
"""

import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def print_result(service, success, message=""):
    """Print test result"""
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"{status} - {service}")
    if message:
        print(f"       {message}")

def test_environment():
    """Test that environment variables are configured"""
    print_header("Testing Environment Configuration")
    
    required_vars = [
        "GEMINI_API_KEY",
        "AUTH0_DOMAIN",
        "ELEVENLABS_API_KEY"
    ]
    
    optional_vars = [
        "AGENTUITY_API_KEY",
        "MATLAB_SERVICE_URL"
    ]
    
    all_configured = True
    
    for var in required_vars:
        value = os.getenv(var)
        configured = value is not None and value != "" and "your_" not in value.lower()
        print_result(var, configured, 
                    "Configured" if configured else "Not configured - REQUIRED")
        if not configured:
            all_configured = False
    
    for var in optional_vars:
        value = os.getenv(var)
        configured = value is not None and value != ""
        print_result(var, configured, 
                    "Configured" if configured else "Not configured - Optional")
    
    return all_configured

def test_gemini_service():
    """Test Gemini API integration"""
    print_header("Testing Gemini Service")
    
    try:
        from gemini_service import GeminiService
        
        service = GeminiService()
        print_result("Gemini Service Import", True)
        
        # Test prompt parsing
        test_prompt = "Close Cooper Street from 7 AM to 10 AM"
        result = service.parse_prompt(test_prompt)
        
        success = result is not None and "action" in result
        print_result("Gemini Prompt Parsing", success, 
                    f"Action: {result.get('action', 'N/A')}")
        
        return success
        
    except Exception as e:
        print_result("Gemini Service", False, str(e))
        return False

def test_osm_service():
    """Test OpenStreetMap data service"""
    print_header("Testing OpenStreetMap Service")
    
    try:
        from osm_data_service import OSMDataService
        
        service = OSMDataService()
        print_result("OSM Service Import", True)
        
        # Test network fetching (will use mock if OSMnx not installed)
        network = service.get_ut_arlington_network()
        
        success = network is not None and "nodes" in network
        print_result("OSM Network Fetch", success,
                    f"Nodes: {network.get('node_count', 0)}, "
                    f"Edges: {network.get('edge_count', 0)}")
        
        return success
        
    except Exception as e:
        print_result("OSM Service", False, str(e))
        return False

def test_matlab_service():
    """Test MATLAB simulation service"""
    print_header("Testing MATLAB Service")
    
    try:
        from matlab_service import MATLABSimulationService
        
        service = MATLABSimulationService()
        print_result("MATLAB Service Import", True)
        
        # Test simulation (will use mock if MATLAB not installed)
        test_network = {
            "nodes": [{"id": "1", "lat": 32.73, "lng": -97.11}],
            "edges": [{"from": "1", "to": "2", "length": 1000}]
        }
        
        test_params = {
            "action": "CLOSE_ROAD",
            "parameters": {"street_name": "Cooper Street"}
        }
        
        result = service.run_traffic_simulation(test_network, test_params)
        
        success = result is not None and result.get("status") == "completed"
        print_result("MATLAB Simulation", success,
                    f"Travel time change: {result.get('metrics', {}).get('travel_time_change_pct', 0):.1f}%")
        
        if service.matlab_available:
            print("       ℹ Using real MATLAB engine")
        else:
            print("       ℹ Using mock simulation (MATLAB not installed)")
        
        return success
        
    except Exception as e:
        print_result("MATLAB Service", False, str(e))
        return False

def test_auth_service():
    """Test Auth0 configuration"""
    print_header("Testing Auth0 Service")
    
    try:
        from auth_service import verify_token
        
        print_result("Auth Service Import", True)
        
        # Check if Auth0 is configured
        domain = os.getenv("AUTH0_DOMAIN")
        audience = os.getenv("AUTH0_API_AUDIENCE")
        
        if domain and "your-tenant" not in domain:
            print_result("Auth0 Configuration", True, f"Domain: {domain}")
        else:
            print_result("Auth0 Configuration", False, 
                        "Update AUTH0_DOMAIN in .env file")
        
        # Check if skip verification is enabled (dev mode)
        skip_auth = os.getenv("SKIP_AUTH_VERIFICATION") == "true"
        if skip_auth:
            print("       ℹ Auth verification is disabled (development mode)")
        
        return True
        
    except Exception as e:
        print_result("Auth Service", False, str(e))
        return False

def test_agents():
    """Test agent scripts"""
    print_header("Testing Agentuity Agents")
    
    try:
        # Test imports
        sys.path.append("agents")
        
        from orchestrator_agent import OrchestratorAgent
        print_result("OrchestratorAgent Import", True)
        
        from simulation_agent import SimulationAgent
        print_result("SimulationAgent Import", True)
        
        from reporting_agent import ReportingAgent
        print_result("ReportingAgent Import", True)
        
        print("\n       ℹ Agents are ready for deployment to Agentuity")
        print("       ℹ Test locally with: python agents/orchestrator_agent.py")
        
        return True
        
    except Exception as e:
        print_result("Agent Imports", False, str(e))
        return False

def test_api_server():
    """Test that API server can be imported"""
    print_header("Testing API Server")
    
    try:
        from main import app
        print_result("FastAPI App Import", True)
        
        print("\n       ℹ To start the server, run:")
        print("         cd backend")
        print("         python main.py")
        print("\n       ℹ Or:")
        print("         uvicorn main:app --reload")
        
        return True
        
    except Exception as e:
        print_result("API Server", False, str(e))
        return False

def main():
    """Run all tests"""
    print("\n" + "#" * 60)
    print("#")
    print("#  SimCity AI - System Test Suite")
    print("#")
    print("#" * 60)
    
    results = {}
    
    # Run tests
    results["Environment"] = test_environment()
    results["Gemini"] = test_gemini_service()
    results["OSM"] = test_osm_service()
    results["MATLAB"] = test_matlab_service()
    results["Auth0"] = test_auth_service()
    results["Agents"] = test_agents()
    results["API"] = test_api_server()
    
    # Summary
    print_header("Test Summary")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    for service, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {service}")
    
    print(f"\nTotal: {passed}/{total} passed, {failed}/{total} failed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your system is ready.")
        print("\nNext steps:")
        print("  1. Start the backend: cd backend && python main.py")
        print("  2. Test the API at: http://localhost:8000/docs")
        print("  3. Build the frontend")
        print("  4. Deploy agents to Agentuity")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        print("\nCommon issues:")
        print("  - Missing API keys in .env file")
        print("  - Missing Python packages (run: pip install -r requirements.txt)")
        print("  - MATLAB not installed (OK to use mock mode for now)")
    
    return passed == total

if __name__ == "__main__":
    # Change to backend directory for imports
    backend_dir = os.path.join(os.path.dirname(__file__), "backend")
    os.chdir(backend_dir)
    sys.path.insert(0, backend_dir)
    
    success = main()
    sys.exit(0 if success else 1)
