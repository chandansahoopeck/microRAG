import os
from fastapi import FastAPI
from pydantic import BaseModel
from rag_engine import RAGPipeline

app = FastAPI()
pipeline = RAGPipeline()

class Claimrequest(BaseModel):
    query: str

@app.on_event("startup")
def startup_event():
    print("Initializing system and loading mock claims...")
    
    # Load the mock data from the text file
    file_path = "mock_claims.txt"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            mock_data = f.read()
            
        # Ingest the data into your custom MiniVectorStore
        chunks_indexed = pipeline.ingest_txt(mock_data)
        print(f"Successfully indexed {chunks_indexed} claim rules into the vector store.")
    else:
        print("Warning: mock_claims.txt not found. Starting with empty vector store.")

@app.post("/triage")
async def triage_claim(request: Claimrequest):
    result = pipeline.process_claim(request.query)
    return {
        "status": "success",
        "triage_result": result}