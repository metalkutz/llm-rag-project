"""
FastAPI main application module.
Provides RESTful endpoints for the RAG microservice.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.models import QueryRequest, QueryResponse, HealthResponse
from src.rag.pipeline import RAGPipeline
from src.config.config import get_settings

# Global RAG pipeline instance
rag_pipeline: Optional[RAGPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    global rag_pipeline
    
    # Startup
    logger.info("Starting RAG microservice...")
    settings = get_settings()
    rag_pipeline = RAGPipeline(settings)
    await rag_pipeline.initialize()
    logger.info("RAG pipeline initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG microservice...")


# Create FastAPI app with lifespan management
app = FastAPI(
    title="RAG Microservice",
    description="A production-ready Retrieval-Augmented Generation API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_rag_pipeline() -> RAGPipeline:
    """Dependency to get the RAG pipeline instance."""
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503, 
            detail="RAG pipeline not initialized"
        )
    return rag_pipeline


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    try:
        # Check if RAG pipeline is available
        pipeline = get_rag_pipeline()
        await pipeline.health_check()
        
        return HealthResponse(
            status="healthy",
            message="RAG microservice is running",
            details={
                "api_version": "1.0.0",
                "rag_pipeline": "ready"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            message=f"Health check failed: {str(e)}",
            details={}
        )


@app.post("/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline)
) -> QueryResponse:
    """
    Query the RAG pipeline with a question.
    
    Args:
        request: The query request containing the question
        pipeline: RAG pipeline dependency
        
    Returns:
        QueryResponse with answer and sources
    """
    try:
        logger.info(f"Received query: {request.query}")
        
        # Get RAG response
        result = await pipeline.query(request.query)
        
        response = QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence_score=result.get("confidence_score", 0.0),
            query_id=result.get("query_id", "")
        )
        
        logger.info(f"Query processed successfully. Answer length: {len(response.answer)}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "message": "RAG Microservice API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/debug/pipeline-status")
async def debug_pipeline_status(pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    """Debug endpoint to check pipeline status."""
    try:
        # Get pipeline health info
        health_info = await pipeline.health_check()
        doc_count = health_info.get("document_count", None)
        index_status = health_info.get("index", None)
        
        return {
            "pipeline_initialized": True,
            "document_count": doc_count,
            "index_status": index_status
        }
    except Exception as e:
        return {
            "pipeline_initialized": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=True
    )
