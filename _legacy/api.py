from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel
from secure_rag.pipeline.runner import run_pipeline

app = FastAPI(title="Secure RAG API", version="0.1.0")

class QueryRequest(BaseModel):
    prompt: str
    username: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/api/v1/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    try:
        result = run_pipeline(prompt=req.prompt, username=req.username)
        return {"answer": result["answer"], "sources": result.get("sources", [])}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/healthz")
def healthcheck():
    return {"status": "ok"}