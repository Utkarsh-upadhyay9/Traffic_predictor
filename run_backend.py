"""
Standalone Backend Starter
Run this to start the backend with all features
"""
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Set environment variable
os.environ['GEMINI_API_KEY'] = 'AIzaSyDQBIe6HG58TMe_HYdW6F9AShugLYWYbYk'

# Start uvicorn
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Traffic Predictor v4.1 Backend")
    print("📍 Texas Locations: 33+ cities")
    print("📅 Holiday Detection: ENABLED")
    print("🤖 ML Models: 91.6% avg accuracy")
    print("\n🌐 Server will run on http://localhost:8000\n")
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
