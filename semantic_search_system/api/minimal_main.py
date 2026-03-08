from fastapi import FastAPI
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    import uvicorn
    print("Starting minimal server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
