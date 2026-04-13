import os
# FORCE proxy bypass
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

from fastapi import FastAPI
from pydantic import BaseModel
from rag_engine import RAGPipeline
from contextlib import asynccontextmanager

# Global pipeline instance
pipeline = RAGPipeline()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    print("Initializing system and loading mock claims...")
    
    file_path = "mock_claims.txt"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            mock_data = f.read()
        chunks_indexed = pipeline.ingest_txt(mock_data)
        print(f"Successfully indexed {chunks_indexed} claim rules into the vector store.")
    else:
        print("Warning: mock_claims.txt not found.")
    
    yield
    # --- Shutdown Logic ---
    print("Shutting down...")

# Pass the lifespan to the FastAPI constructor
app = FastAPI(lifespan=lifespan)

class Claimrequest(BaseModel):
    query: str

@app.post("/triage")
async def triage_claim(request: Claimrequest):
    # This calls self.llm.invoke() which now knows to ignore the proxy
    result = pipeline.process_claim(request.query)
    return {
        "status": "success", 
        "triage_result": result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)