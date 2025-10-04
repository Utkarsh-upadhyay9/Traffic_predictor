"""
SimCity AI - Reporting Agent
Generates summaries and audio reports
"""

from typing import Dict, Any
import os
import sys
import json

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
from gemini_service import GeminiService

# Try to import ElevenLabs
try:
    from elevenlabs import ElevenLabs
    ELEVENLABS_AVAILABLE = True
except ImportError:
    print("⚠️  ElevenLabs not installed. Install with: pip install elevenlabs")
    ELEVENLABS_AVAILABLE = False


@agent("ReportingAgent")
class ReportingAgent(Agent):
    """
    Reporting Agent - Final stage of workflow
    
    Responsibilities:
    1. Receive simulation results from SimulationAgent
    2. Use Gemini to generate natural language summary
    3. Use ElevenLabs to create audio narration
    4. Return complete results package to user
    """
    
    async def run(self, data: Dict[str, Any]):
        """
        Main execution method
        
        Args:
            data: Dict containing:
                - simulation_id: Unique ID
                - simulation_results: Output from MATLAB simulation
                - original_prompt: User's original prompt
        """
        print("=" * 60)
        print("REPORTING AGENT: Generating summary and audio")
        print("=" * 60)
        
        simulation_id = data.get("simulation_id")
        simulation_results = data.get("simulation_results")
        original_prompt = data.get("original_prompt")
        
        print(f"Simulation ID: {simulation_id}")
        
        try:
            # Step 1: Generate text summary with Gemini
            print("\n[Step 1/3] Generating summary with Gemini...")
            gemini_service = GeminiService()
            
            text_summary = gemini_service.summarize_simulation_results(
                simulation_results
            )
            
            print(f"✓ Summary generated:")
            print(f"  {text_summary[:150]}...")
            
            # Step 2: Generate audio narration with ElevenLabs
            print("\n[Step 2/3] Generating audio narration with ElevenLabs...")
            audio_url = await self._generate_audio_narration(text_summary)
            
            if audio_url:
                print(f"✓ Audio generated: {audio_url}")
            else:
                print("⚠️  Audio generation skipped (ElevenLabs not configured)")
            
            # Step 3: Package final results
            print("\n[Step 3/3] Packaging results...")
            
            final_results = {
                "status": "completed",
                "simulation_id": simulation_id,
                "original_prompt": original_prompt,
                "summary": {
                    "text": text_summary,
                    "audio_url": audio_url
                },
                "detailed_results": simulation_results,
                "metrics": simulation_results.get("metrics", {}),
                "visualizations": {
                    "congestion_heatmap": simulation_results.get("congestion_heatmap", []),
                    "vehicle_trajectories": simulation_results.get("vehicle_trajectories", [])
                },
                "recommendations": simulation_results.get("recommendations", [])
            }
            
            print("\n✓ Workflow completed successfully!")
            print(f"  Summary length: {len(text_summary)} chars")
            print(f"  Audio: {'✓' if audio_url else '✗'}")
            print(f"  Recommendations: {len(final_results['recommendations'])}")
            
            # Return final results
            # In production, this would be stored in a database
            # and the user would be notified via WebSocket
            return final_results
            
        except Exception as e:
            print(f"\n✗ REPORTING ERROR: {e}")
            
            return {
                "status": "error",
                "simulation_id": simulation_id,
                "error": str(e),
                "stage": "reporting"
            }
    
    async def _generate_audio_narration(self, text: str) -> str:
        """
        Generate audio narration using ElevenLabs
        
        Args:
            text: Text to convert to speech
            
        Returns:
            str: URL to audio file (or None if failed)
        """
        if not ELEVENLABS_AVAILABLE:
            print("  ElevenLabs not available, skipping audio generation")
            return None
        
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            print("  ELEVENLABS_API_KEY not configured")
            return None
        
        try:
            client = ElevenLabs(api_key=api_key)
            
            voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Default voice
            
            # Generate audio
            audio = client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id="eleven_turbo_v2"
            )
            
            # Save audio file
            audio_dir = os.path.join(os.path.dirname(__file__), '..', 'audio_output')
            os.makedirs(audio_dir, exist_ok=True)
            
            audio_file = os.path.join(audio_dir, f"narration_{data.get('simulation_id', 'test')}.mp3")
            
            with open(audio_file, 'wb') as f:
                for chunk in audio:
                    f.write(chunk)
            
            print(f"  Audio saved: {audio_file}")
            
            # Return URL (in production, upload to S3/CDN and return public URL)
            return f"/audio/{os.path.basename(audio_file)}"
            
        except Exception as e:
            print(f"  Audio generation failed: {e}")
            return None


# For local testing
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    
    load_dotenv()
    
    reporting_agent = ReportingAgent()
    
    # Test data (would come from SimulationAgent)
    test_data = {
        "simulation_id": "test-123",
        "user_id": "user-456",
        "original_prompt": "Close Cooper Street from 7 AM to 10 AM",
        "parsed_params": {
            "action": "CLOSE_ROAD",
            "parameters": {"street_name": "Cooper Street"}
        },
        "simulation_results": {
            "status": "completed",
            "metrics": {
                "baseline_travel_time_min": 15.0,
                "new_travel_time_min": 21.0,
                "travel_time_change_pct": -40.0,
                "baseline_congestion": 0.6,
                "new_congestion": 0.84,
                "affected_vehicles": 1500
            },
            "recommendations": [
                "Consider alternative routes via parallel streets",
                "Monitor congestion during peak hours"
            ],
            "congestion_heatmap": [],
            "vehicle_trajectories": []
        }
    }
    
    async def test():
        result = await reporting_agent.run(test_data)
        print("\n" + "=" * 60)
        print("REPORTING RESULT:")
        print("=" * 60)
        print(json.dumps(result, indent=2))
    
    asyncio.run(test())
