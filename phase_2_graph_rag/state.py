"""State for the phase 2 GraphRAG workflow."""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field

from .tools.shared import merge_dicts


class GraphRAGState(BaseModel):
    original_query: str
    rewritten_query: Optional[str] = None
    retrieval_mode: str = "auto"

    intent: Optional[str] = None
    intent_rationale: Optional[str] = None

    entities: List[Dict[str, Any]] = Field(default_factory=list)
    cypher_query: Optional[str] = None
    sql_query: Optional[str] = None
    vector_query: Optional[str] = None

    graph_results: Annotated[Dict[str, Any], merge_dicts] = Field(default_factory=dict)
    vector_results: Annotated[Dict[str, Any], merge_dicts] = Field(default_factory=dict)
    timeline_results: Annotated[Dict[str, Any], merge_dicts] = Field(default_factory=dict)
    verification_results: Annotated[Dict[str, Any], merge_dicts] = Field(default_factory=dict)

    final_answer: Optional[str] = None
    evidence: List[Dict[str, Any]] = Field(default_factory=list)

    answer_style: str = "concise"
    max_words: int = 300

    step_count: int = 0
    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    session_id: str = "default"


class GraphRAGStateDict(TypedDict, total=False):
    original_query: str
    query: str
    input_query: str
    rewritten_query: Optional[str]
    retrieval_mode: str

    intent: Optional[str]
    intent_rationale: Optional[str]
    use_vector: bool
    use_graph: bool
    use_timeline: bool
    use_verification: bool

    entities: List[Dict[str, Any]]
    cypher_query: Optional[str]
    sql_query: Optional[str]
    vector_query: Optional[str]

    graph_results: Annotated[Dict[str, Any], merge_dicts]
    vector_results: Annotated[Dict[str, Any], merge_dicts]
    timeline_results: Annotated[Dict[str, Any], merge_dicts]
    verification_results: Annotated[Dict[str, Any], merge_dicts]

    final_answer: Optional[str]
    evidence: List[Dict[str, Any]]

    answer_style: str
    max_words: int

    step_count: int
    chat_history: List[Dict[str, str]]
    session_id: str
