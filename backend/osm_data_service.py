"""
OpenStreetMap Data Service
Fetches real road network data for simulation
"""

import os
from typing import Dict, Tuple, Optional
import json

# Try to import OSMnx - it may not be installed yet
try:
    import osmnx as ox
    import networkx as nx
    OSMNX_AVAILABLE = True
except ImportError:
    print("⚠️  OSMnx not installed. Install with: pip install osmnx")
    OSMNX_AVAILABLE = False


class OSMDataService:
    """Service for fetching and processing OpenStreetMap data"""
    
    def __init__(self, cache_dir: str = "./osm_cache"):
        """
        Initialize OSM data service
        
        Args:
            cache_dir: Directory to cache downloaded OSM data
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        if OSMNX_AVAILABLE:
            ox.settings.use_cache = True
            ox.settings.cache_folder = cache_dir
    
    def fetch_road_network(
        self,
        place_name: str = "Arlington, Texas, USA",
        network_type: str = "drive"
    ) -> Dict:
        """
        Fetch road network for a specified place
        
        Args:
            place_name: Name of place to fetch (city, address, etc.)
            network_type: Type of network ('drive', 'walk', 'bike', 'all')
            
        Returns:
            Dict containing road network data (nodes and edges)
        """
        
        if not OSMNX_AVAILABLE:
            return self._mock_road_network(place_name)
        
        try:
            print(f"Fetching road network for {place_name}...")
            
            # Download street network
            G = ox.graph_from_place(place_name, network_type=network_type)
            
            print(f"✓ Downloaded {len(G.nodes)} nodes and {len(G.edges)} edges")
            
            # Convert to serializable format
            network_data = self._graph_to_dict(G)
            
            # Cache the result
            cache_file = os.path.join(
                self.cache_dir,
                f"{place_name.replace(' ', '_').replace(',', '')}.json"
            )
            with open(cache_file, 'w') as f:
                json.dump(network_data, f)
            
            return network_data
            
        except Exception as e:
            print(f"✗ Error fetching road network: {e}")
            return self._mock_road_network(place_name)
    
    def fetch_road_network_bbox(
        self,
        north: float,
        south: float,
        east: float,
        west: float,
        network_type: str = "drive"
    ) -> Dict:
        """
        Fetch road network for a bounding box
        
        Args:
            north, south, east, west: Bounding box coordinates
            network_type: Type of network
            
        Returns:
            Dict containing road network data
        """
        
        if not OSMNX_AVAILABLE:
            return self._mock_road_network(f"bbox_{north}_{south}_{east}_{west}")
        
        try:
            print(f"Fetching road network for bbox...")
            
            G = ox.graph_from_bbox(
                north=north,
                south=south,
                east=east,
                west=west,
                network_type=network_type
            )
            
            print(f"✓ Downloaded {len(G.nodes)} nodes and {len(G.edges)} edges")
            
            return self._graph_to_dict(G)
            
        except Exception as e:
            print(f"✗ Error fetching road network: {e}")
            return self._mock_road_network("bbox")
    
    def get_ut_arlington_network(self) -> Dict:
        """
        Get road network for UT Arlington area
        Convenience function for hackathon demo
        """
        # UT Arlington bounding box
        return self.fetch_road_network_bbox(
            north=32.7350,
            south=32.7250,
            east=-97.1050,
            west=-97.1200,
            network_type="drive"
        )
    
    def _graph_to_dict(self, G) -> Dict:
        """
        Convert NetworkX graph to serializable dictionary
        
        Args:
            G: NetworkX graph from OSMnx
            
        Returns:
            Dict with nodes and edges
        """
        nodes = []
        for node_id, data in G.nodes(data=True):
            nodes.append({
                "id": str(node_id),
                "lat": data.get('y', 0),
                "lng": data.get('x', 0),
                "street_count": data.get('street_count', 0)
            })
        
        edges = []
        for u, v, key, data in G.edges(keys=True, data=True):
            edges.append({
                "from": str(u),
                "to": str(v),
                "key": key,
                "length": data.get('length', 0),
                "highway": data.get('highway', 'unknown'),
                "name": data.get('name', ''),
                "maxspeed": data.get('maxspeed', '30 mph'),
                "lanes": data.get('lanes', '1'),
                "oneway": data.get('oneway', False)
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges)
        }
    
    def _mock_road_network(self, place_name: str) -> Dict:
        """
        Generate mock road network for development
        Creates a simple grid pattern
        """
        print(f"🔧 Generating MOCK road network for {place_name}")
        
        # Create a 5x5 grid of nodes
        nodes = []
        edges = []
        node_id = 0
        
        # Base coordinates (UT Arlington area)
        base_lat = 32.7299
        base_lng = -97.1161
        spacing = 0.002  # About 220 meters
        
        # Create grid nodes
        node_map = {}
        for i in range(5):
            for j in range(5):
                node = {
                    "id": str(node_id),
                    "lat": base_lat + (i * spacing),
                    "lng": base_lng + (j * spacing),
                    "street_count": 0
                }
                nodes.append(node)
                node_map[(i, j)] = node_id
                node_id += 1
        
        # Create grid edges (horizontal and vertical connections)
        street_names = ["Cooper St", "Center St", "Davis St", "Arkansas Ln", "Mitchell St"]
        
        for i in range(5):
            for j in range(5):
                current_id = node_map[(i, j)]
                
                # Horizontal edge (east)
                if j < 4:
                    next_id = node_map[(i, j + 1)]
                    edges.append({
                        "from": str(current_id),
                        "to": str(next_id),
                        "key": 0,
                        "length": 220,
                        "highway": "residential",
                        "name": street_names[i],
                        "maxspeed": "30 mph",
                        "lanes": "2",
                        "oneway": False
                    })
                
                # Vertical edge (north)
                if i < 4:
                    next_id = node_map[(i + 1, j)]
                    edges.append({
                        "from": str(current_id),
                        "to": str(next_id),
                        "key": 0,
                        "length": 220,
                        "highway": "residential",
                        "name": f"{j}th Ave",
                        "maxspeed": "35 mph",
                        "lanes": "2",
                        "oneway": False
                    })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "mock": True
        }


# Testing
if __name__ == "__main__":
    service = OSMDataService()
    
    # Test fetching UT Arlington network
    print("\n=== Testing UT Arlington Network ===")
    network = service.get_ut_arlington_network()
    
    print(f"\nNetwork summary:")
    print(f"Nodes: {network['node_count']}")
    print(f"Edges: {network['edge_count']}")
    print(f"First node: {network['nodes'][0]}")
    print(f"First edge: {network['edges'][0]}")
