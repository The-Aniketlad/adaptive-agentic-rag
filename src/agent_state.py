from typing import List, Dict, Any, Optional, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# 1. Pydantic Schemas for Structured Outputs & Grading
# ---------------------------------------------------------

class RouteQuery(BaseModel):
    """Route query to the most appropriate execution path."""
    destination: Literal["vectorstore", "direct", "decomposer"] = Field(
        description="Choose 'direct' for general greeting/chit-chat, 'decomposer' for multi-hop/complex bridge questions requiring multiple facts, or 'vectorstore' for standard factual retrieval."
    )

class GradeDocuments(BaseModel):
    """Binary score for document relevance check."""
    binary_score: Literal["yes", "no"] = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class DecomposeQuery(BaseModel):
    """Decompose a complex multi-hop question into 2 simpler search sub-queries."""
    sub_questions: List[str] = Field(
        description="List of 2 simple, independent sub-questions that together solve the original question."
    )

class GradeHallucination(BaseModel):
    """Binary score for answer grounding against context."""
    binary_score: Literal["yes", "no"] = Field(
        description="Answer is grounded in the facts from context, 'yes' or 'no'"
    )

# ---------------------------------------------------------
# 2. LangGraph Shared State Definition
# ---------------------------------------------------------

class AgentState(TypedDict):
    """
    Represents the internal state of our LangGraph Agent.
    """
    question: str                  # Original user question
    current_query: str             # Current search query (may be rewritten)
    sub_questions: List[str]       # Decomposed sub-questions if multi-hop
    documents: List[str]           # Retrieved context passages
    generation: str                # Candidate answer from LLM
    loop_step: int                 # Count of query rewriting iterations (max 2)
    web_search_needed: bool        # Flag whether web fallback should be triggered
    grounded: bool                 # Grounding verification flag
