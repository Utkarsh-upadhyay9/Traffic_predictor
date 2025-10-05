"""
Simple startup script for the backend server
"""
import uvicorn
from simple_backend import app

if __name__ == "__main__":
    print("\n🚀 Starting Simple Traffic Predictor Backend")
    print("🌾 Rural area detection: ENABLED")
    print("🤖 Gemini API: CONFIGURED for rural areas")
    print("🧠 Deep Learning: CONFIGURED for urban areas\n")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
