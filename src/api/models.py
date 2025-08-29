"""
Pydantic models for API request/response schemas.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str = Field(
        ...,
        description="The question or query to ask the RAG system",
        min_length=1,
        max_length=1000
    )
    max_sources: Optional[int] = Field(
        default=3,
        description="Maximum number of source documents to return",
        ge=1,
        le=10
    )
    include_metadata: Optional[bool] = Field(
        default=True,
        description="Whether to include source metadata in the response"
    )


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str = Field(
        ...,
        description="The generated answer from the RAG system"
    )
    sources: List[str] = Field(
        ...,
        description="List of source documents used to generate the answer"
    )
    confidence_score: Optional[float] = Field(
        default=0.0,
        description="Confidence score for the answer (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    query_id: Optional[str] = Field(
        default="",
        description="Unique identifier for the query"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata about the response"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(
        ...,
        description="Health status",
        pattern="^(healthy|unhealthy|degraded)$"
    )
    message: str = Field(
        ...,
        description="Human-readable status message"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional health check details"
    )


class ErrorResponse(BaseModel):
    """Response model for API errors."""
    error: str = Field(
        ...,
        description="Error type or code"
    )
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
