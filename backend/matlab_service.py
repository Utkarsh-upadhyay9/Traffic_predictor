"""
MATLAB Traffic Simulation - Python Bridge
Uses MATLAB Engine API to execute traffic simulations
"""

import os
import json
from typing import Dict, Optional
import numpy as np


# Try to import MATLAB engine - it may not be installed yet
try:
    import matlab.engine
    MATLAB_AVAILABLE = True
except ImportError:
    print("⚠️  MATLAB Engine API not installed. Running in mock mode.")
    print("   Install from your MATLAB installation:")
    print("   cd 'matlabroot\\extern\\engines\\python'")
    print("   python setup.py install")
    MATLAB_AVAILABLE = False


class MATLABSimulationService:
    """Service for running traffic simulations via MATLAB"""
    
    def __init__(self):
        """Initialize MATLAB engine connection"""
        self.engine = None
        self.matlab_available = MATLAB_AVAILABLE
        
        if self.matlab_available:
            try:
                print("Starting MATLAB Engine...")
                self.engine = matlab.engine.start_matlab()
                
                # Add MATLAB scripts directory to path
                matlab_dir = os.path.join(os.path.dirname(__file__), '..', 'matlab')
                if os.path.exists(matlab_dir):
                    self.engine.addpath(matlab_dir)
                
                print("✓ MATLAB Engine started successfully")
            except Exception as e:
                print(f"✗ Failed to start MATLAB Engine: {e}")
                self.matlab_available = False
    
    def __del__(self):
        """Clean up MATLAB engine on deletion"""
        if self.engine:
            try:
                self.engine.quit()
                print("MATLAB Engine stopped")
            except:
                pass
    
    def run_traffic_simulation(
        self,
        road_network: Dict,
        scenario_params: Dict
    ) -> Dict:
        """
        Execute traffic simulation in MATLAB
        
        Args:
            road_network: Dict containing road network data (nodes, edges)
            scenario_params: Dict with simulation parameters (closures, modifications, etc.)
            
        Returns:
            Dict containing simulation results (congestion, travel times, etc.)
        """
        
        if not self.matlab_available or not self.engine:
            return self._mock_simulation(road_network, scenario_params)
        
        try:
            print("Running MATLAB traffic simulation...")
            
            # Call MATLAB function
            # Assumes a function 'runTrafficSimulation.m' exists in matlab/ directory
            results = self.engine.runTrafficSimulation(
                road_network,
                scenario_params,
                nargout=1
            )
            
            print("✓ MATLAB simulation completed")
            
            # Convert MATLAB results to Python dict
            return self._matlab_to_dict(results)
            
        except Exception as e:
            print(f"✗ MATLAB simulation error: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def optimize_traffic_signals(
        self,
        current_signals: Dict,
        congestion_data: Dict
    ) -> Dict:
        """
        Use MATLAB's optimization toolbox to optimize traffic signal timing
        
        Args:
            current_signals: Current signal timings
            congestion_data: Current congestion metrics
            
        Returns:
            Dict with optimized signal timings
        """
        
        if not self.matlab_available or not self.engine:
            return self._mock_optimization(current_signals, congestion_data)
        
        try:
            print("Running MATLAB signal optimization...")
            
            optimized = self.engine.optimizeSignals(
                current_signals,
                congestion_data,
                nargout=1
            )
            
            print("✓ Signal optimization completed")
            return self._matlab_to_dict(optimized)
            
        except Exception as e:
            print(f"✗ Optimization error: {e}")
            return {"error": str(e)}
    
    def _matlab_to_dict(self, matlab_data) -> Dict:
        """Convert MATLAB data structure to Python dict"""
        # This is a placeholder - actual conversion depends on MATLAB output format
        try:
            return dict(matlab_data)
        except:
            return {"data": str(matlab_data)}
    
    def _mock_simulation(self, road_network: Dict, scenario_params: Dict) -> Dict:
        """
        Mock simulation for development without MATLAB
        Returns realistic-looking fake data
        """
        print("🔧 Running MOCK simulation (MATLAB not available)")
        
        action = scenario_params.get("action", "UNKNOWN")
        street = scenario_params.get("parameters", {}).get("street_name", "Unknown Street")
        
        # Generate mock results
        baseline_travel_time = 15.0  # minutes
        baseline_congestion = 0.6  # 60% congestion
        
        # Simulate impact based on action
        impact_multiplier = {
            "CLOSE_ROAD": 1.4,  # 40% worse
            "ADD_LANE": 0.8,     # 20% better
            "MODIFY_LANE_COUNT": 1.1,  # 10% worse
            "ADD_TRAFFIC_LIGHT": 1.05,
        }.get(action, 1.0)
        
        new_travel_time = baseline_travel_time * impact_multiplier
        new_congestion = min(1.0, baseline_congestion * impact_multiplier)
        
        improvement_pct = ((baseline_travel_time - new_travel_time) / baseline_travel_time) * 100
        
        return {
            "status": "completed",
            "simulation_type": "mock",
            "scenario": {
                "action": action,
                "street": street,
                "description": f"{action} on {street}"
            },
            "metrics": {
                "baseline_travel_time_min": round(baseline_travel_time, 2),
                "new_travel_time_min": round(new_travel_time, 2),
                "travel_time_change_pct": round(improvement_pct, 1),
                "baseline_congestion": round(baseline_congestion, 2),
                "new_congestion": round(new_congestion, 2),
                "congestion_change_pct": round((baseline_congestion - new_congestion) * 100, 1),
                "affected_vehicles": int(np.random.randint(500, 2000)),
                "avg_delay_min": round(abs(new_travel_time - baseline_travel_time), 2)
            },
            "recommendations": [
                f"Consider alternative routes via parallel streets",
                f"Monitor congestion during peak hours",
                f"Evaluate public transit options"
            ],
            "congestion_heatmap": self._generate_mock_heatmap(),
            "vehicle_trajectories": self._generate_mock_trajectories()
        }
    
    def _mock_optimization(self, current_signals: Dict, congestion_data: Dict) -> Dict:
        """Mock signal optimization"""
        return {
            "status": "optimized",
            "improvement_pct": 15.5,
            "optimized_signals": current_signals  # Would normally be optimized values
        }
    
    def _generate_mock_heatmap(self) -> list:
        """Generate mock congestion heatmap data"""
        # Grid of lat/lng with congestion density values
        heatmap = []
        for i in range(20):
            for j in range(20):
                heatmap.append({
                    "lat": 32.72 + (i * 0.001),
                    "lng": -97.12 + (j * 0.001),
                    "density": float(np.random.rand())
                })
        return heatmap
    
    def _generate_mock_trajectories(self) -> list:
        """Generate mock vehicle trajectory data"""
        trajectories = []
        for vehicle_id in range(10):
            trajectory = {
                "vehicle_id": vehicle_id,
                "path": [
                    {
                        "time": t,
                        "lat": 32.73 + (t * 0.0001 * np.random.randn()),
                        "lng": -97.11 + (t * 0.0001 * np.random.randn()),
                        "speed": max(0, 30 + 10 * np.random.randn())
                    }
                    for t in range(0, 60, 5)  # Every 5 seconds for 1 minute
                ]
            }
            trajectories.append(trajectory)
        return trajectories


# Standalone execution for testing
if __name__ == "__main__":
    service = MATLABSimulationService()
    
    # Test simulation
    test_network = {
        "nodes": [{"id": 1, "lat": 32.73, "lng": -97.11}],
        "edges": [{"from": 1, "to": 2, "length": 1000}]
    }
    
    test_params = {
        "action": "CLOSE_ROAD",
        "parameters": {
            "street_name": "Cooper Street"
        }
    }
    
    results = service.run_traffic_simulation(test_network, test_params)
    print("\nSimulation Results:")
    print(json.dumps(results, indent=2))
