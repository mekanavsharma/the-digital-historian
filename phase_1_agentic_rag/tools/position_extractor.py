# phase_1/tools/position_extractor.py

"""
Extracts structured evidence positions from each retrieval lane's raw results.

LangGraph node – iterates over all lanes, converts the flat
retrieved_results into structured position dicts, and attaches the
full raw evidence (passages + chunk_ids) for the synthesizer.

Pull the top passages and their chunk IDs from a lane's raw context list.

Args:
    query:             The lane's sub-query (kept for future use / logging).
    retrieved_context: List of {"page_content": ..., "metadata": ...} dicts.
    historian:         Historian label for this lane.
    top_passages:      How many passages to extract.

Returns:
    Dict with keys: historian, passages (truncated), chunk_ids.
"""


from typing import Any, Dict, List


class PositionExtractorTool:
    @staticmethod
    def extract_position(
        query: str,
        retrieved_context: List[Dict[str, Any]],
        historian: str = "general",
    ) -> Dict[str, Any]:
        """
        Pull the top passages and their chunk IDs from a lane's raw context list.

        Args:
            query:             The lane's sub-query (kept for future use / logging).
            retrieved_context: List of {"page_content": ..., "metadata": ...} dicts.
            historian:         Historian label for this lane.
            top_passages:      How many passages to extract.

        Returns:
            Dict with keys: historian, passages (truncated), chunk_ids.
        """
        passages = []
        chunk_ids = []

        for doc in retrieved_context[:6]:
            text = doc.get("page_content", "")
            meta = doc.get("metadata", {})
            cid = meta.get("chunk_id", "unknown")

            passages.append(text[:400])
            chunk_ids.append(cid)

        return {
            "historian": historian,
            "passages": passages,
            "chunk_ids": chunk_ids,
        }

    def run(self, state) -> Dict[str, Any]:
        positions = {}

        for lane in state.lanes:
            lane_id = lane["lane_id"]
            res = state.retrieved_results.get(lane_id, {})

            retrieved_docs = res.get("retrieved_docs", [])
            metadata_list = res.get("metadata", [])
            retrieved_list = [
                {"page_content": d, "metadata": m}
                for d, m in zip(retrieved_docs, metadata_list)
            ]
            if not retrieved_list:
                continue

            pos = self.extract_position(
                lane["question"],
                retrieved_list,
                lane["historian"],
            )
            passages = []
            chunk_ids = []
            for doc, meta in zip(retrieved_docs, metadata_list):
                passages.append(doc)
                cid = meta.get("chunk_id")
                if cid:
                    chunk_ids.append(cid)

            pos["passages"] = passages
            pos["chunk_ids"] = chunk_ids
            positions[lane_id] = pos

        return {"positions": positions}
