# shared/prompts/rag_prompts.py
import re
import textwrap
from typing import Sequence, Optional

def build_prompt(
    context_texts: Sequence,
    question: str,
    max_context_chars: int = 10000,
    answer_style: str = "concise",
    max_words: Optional[int] = None,
):
    """
    Build a strict, flexible prompt.

    - context_texts: list[Document] (preferred) or list[str]
    - question: user's question
    - max_context_chars: how much information the model is allowed to "read."
    - answer_style: "short" | "concise" | "detailed"
      - "short": 1-2 short sentences
      - "concise": 2-4 sentences
      - "detailed": allow longer answers, respect max_words if provided
    - max_words: optional integer, if provided enforce max word length for the generated answer

    Returns:
        prompt (str)
        used_chunk_ids (List[str])  # exactly which chunks are in CONTEXT
    """
    processed = []
    chunk_ids = []
    total_chars = 0
    MAX_CHUNKS = 10   # No of chunks passed to prompts

    for c in context_texts:
        if not hasattr(c, "page_content"):
            continue

        meta = c.metadata
        text = c.page_content

        cid = meta.get("chunk_id")
        vol = meta.get("volume")
        page = meta.get("page")

        block = (
            f"[chunk_id={cid}] [volume={vol}] [page={page}]\n"
            f"{text}\n\n"
        )

        if total_chars + len(block) > max_context_chars:
            break

        processed.append(block)
        if cid is not None:
            chunk_ids.append(cid)

        total_chars += len(block)

        if len(processed) == MAX_CHUNKS:
            break

    context = "\n\n".join(processed)

    # Answer length guidance
    if answer_style == "short":
        length_guidance = "Answer in 1-2 short sentences."
    elif answer_style == "concise":
        length_guidance = "Answer in 2-4 sentences."
    elif answer_style == "detailed":
        length_guidance = "Answer in a clear, multi-paragraph explanation as needed."
    else:
        length_guidance = "Answer naturally and concisely."

    if max_words is not None:
        length_guidance += f" Limit the answer to approximately {int(max_words)} words."


    # CRITICAL: Make citation expectations crystal clear
    prompt = textwrap.dedent(f"""\
        You are a knowledgeable history assistant specializing in Indian history.
        Answer the question using the provided CONTEXT.

        CONTEXT:
        {context}

        INSTRUCTIONS:

        CONTENT RULES:
        1. Answer the question using ONLY the provided context.
        2. Do not use outside knowledge.
        3. {length_guidance}
        4. If the context contains no relevant information, respond only with "I dont know.".

        CITATION RULES:
        5. You MUST base your answer only on the chunks in CONTEXT.
        6. At the very end, we will automatically attach all chunk_id values from the CONTEXT.

        STRICT OUTPUT FORMAT:
        7. Write the answer in paragraph form.
        8. Do NOT write any chunk_id=... tokens yourself.

        QUESTION: {question}

        ANSWER:
        """).strip()

    return prompt, chunk_ids
