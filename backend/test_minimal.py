from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "alive", "message": "Minimal server is working"}

@app.get("/test")
def test():
    return {"test": "success"}
