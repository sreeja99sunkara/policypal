from fastapi import FastAPI
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — runs once when server starts
    # Initialize heavy objects here so they're ready before first request
    print("Starting up PolicyPal...")
    from app.services.embedder import get_embeddings
    from app.services.vectorstore import get_vectorstore
    from app.services.rag_chain import get_rag_chain
    get_embeddings()
    get_vectorstore()
    get_rag_chain()
    print("All services ready.")
    yield
    # Shutdown — runs when server stops
    print("Shutting down PolicyPal...")


app = FastAPI(
    title="PolicyPal",
    description="RAG-powered insurance policy question answering",
    version="0.1.0",
    lifespan=lifespan
)

# Register routers
from app.routers import query                   
app.include_router(query.router, prefix="/api") 

@app.get("/health")
def health():
    return {"status": "ok"}