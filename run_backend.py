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
    print("\n🚀 Starting SimCity AI v2.1 Backend")
    print("📍 Pin-to-Place: ENABLED")
    print("📅 Calendar Integration: ENABLED")
    print("🤖 ML Models: LOADED")
    print("\n🌐 Server will run on http://localhost:8001\n")
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
