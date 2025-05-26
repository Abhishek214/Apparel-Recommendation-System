
**Results:**
Based on our evaluation, we found that the best-performing recommendation models are as follows:

1. TF-IDF: This model achieved the highest recommendation performance, suggesting the most relevant and similar apparel products.

2.AVERAGE WORD2VEC

3.BAG OF WORDS

4.BRAND AND COLOR

5.WEIGHTED WORD2VEC

6.IDF

from fastapi import FastAPI
import uvicorn
import platform

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "API is working"}

@app.get("/health")
async def health():
    return {"status": "ok", "server": platform.node()}

if __name__ == "__main__":
    print("Starting API...")
    print("Available routes:")
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            print(f"  {route.methods} {route.path}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

