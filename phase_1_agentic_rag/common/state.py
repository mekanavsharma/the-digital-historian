"""LangGraph state definition."""

from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .utils import merge_dicts


class HistorianState(BaseModel):
    original_query: str
    rewritten_query: Optional[str] = None

    plan: Optional[Dict[str, Any]] = None
    lanes: List[Dict[str, Any]] = Field(default_factory=list)

    retrieved_results: Annotated[Dict[str, Any], merge_dicts] = Field(default_factory=dict)
    positions: Annotated[Dict[str, Any], merge_dicts] = Field(default_factory=dict)

    historian_override: Optional[List[str]] = None
    claim_comparison: Optional[Dict] = None
    evaluation: Optional[Dict] = None
    final_answer: Optional[str] = None

    answer_style: str = "concise"
    max_words: Optional[int] = None

    step_count: int = 0
    max_steps: int = 10

    chat_history: List[Dict[str, str]] = Field(default_factory=list)
    session_id: int = 0
