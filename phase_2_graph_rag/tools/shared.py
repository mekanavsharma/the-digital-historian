"""Small shared helpers for phase 2 tools."""

from __future__ import annotations

from typing import Any, Dict, Optional


def merge_dicts(left: Optional[Dict], right: Optional[Dict]) -> Dict:
    left = left or {}
    right = right or {}
    merged = dict(left)
    merged.update(right)
    return merged
