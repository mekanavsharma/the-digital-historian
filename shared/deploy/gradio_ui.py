# shared/gradio_ui.py
"""
Reusable Gradio Chat UI for The Digital Historian (Gradio 5.x / 6.x)

Fixes:
- Correct history handling (Gradio → agent)
- Persistent memory using gr.State
- Uses agent-returned memory (true follow-ups)
- Historian input support
- Citations rendering
"""


import gradio as gr
import re
from typing import List, Dict, Any, Callable, Optional


def _extract_citations(text: str) -> list[str]:
    """Extract all [chunk_id=XXX] from the answer."""
    return re.findall(r'\[chunk_id=([^\]]+)\]', text)

HISTORIAN_OPTIONS = [
    "Abraham Eraly",
    "AK Priolkar",
    "AL Basham",
    "Andre Wink",
    "Bipan Chandra",
    "Dharmpal",
    "GB Mehendale",
    "GN Sharma",
    "HR Gupta",
    "Irfan Habib",
    "Jadunath Sarkar",
    "KA Nilakanta Sastri",
    "KM Panikkar",
    "Koenraad Elst",
    "KS Lal",
    "Meenakshi Jain",
    "RC Majumdar",
    "RK Gupta",
    "RK Mukherjee",
    "Robert Sewell",
    "Romilla Thappar",
    "RS Sharma",
    "Sandhya Jain",
    "Satish Chandra",
    "S Gopal",
    "Sumit Sarkar",
    "Uday S Kulkarni",
    "VK Dhulipala",
    "Will Durant",
    "William Dalrymple",
]

def launch_historian_ui(
    run_query_func: Callable,
    title: str = "The Digital Historian",
    description: str = "Agentic historical assistant with memory • Ask follow-ups naturally",
):
    """
    run_query_func: your run_query from phase_1.run_query (or any future phase)
    """
    def chat(
        message: str,
        history: List[Dict[str, str]],
        answer_style: str,
        max_words: int,
        historians_selected: Optional[str],
        memory_state: List[Dict[str, str]],
    ):

        # Use persistent memory (NOT UI history)
        chat_memory = memory_state or []

        # # Convert Gradio history (role-based) → your format
        # for msg in history:
        #     if isinstance(msg, dict):
        #         chat_memory.append({
        #             "role": msg.get("role"),
        #             "content": msg.get("content")
        #         })

        # Parse historian input
        historians = historians_selected if historians_selected else None

        # Show thinking message
        yield "Thinking... (retrieving sources, extracting claims, synthesizing answer)"

        # Call the agent
        result = run_query_func(
            question=message,
            chat_memory=chat_memory,
            answer_style=answer_style,
            max_words=max_words,
            historians=historians,
        )

        final_answer = result.get("final_answer", "Sorry, I couldn't generate an answer.")
        rewritten = result.get("rewritten_query", "")

        # Show rewritten query (if changed)
        if rewritten and rewritten.lower() != message.lower():
            final_answer = f"**Rewritten query:** {rewritten}\n\n{final_answer}"

        # Add citations
        citations = _extract_citations(final_answer)
        if citations:
            sources_md = "\n".join([f"- `{cid}`" for cid in citations])
            final_answer += f"\n\n---\n**Sources used:**\n{sources_md}"

        # IMPORTANT: update chat_memory from agent
        memory_state.clear()
        memory_state.extend(result.get("chat_history", chat_memory))

        # Return answer + updated memory
        yield final_answer

    # Controls
    style_dropdown = gr.Dropdown(
        choices=["short", "concise", "detailed"],
        value="concise",
        label="Answer Style",
        info="How detailed should the response be?"
    )

    max_words_slider = gr.Slider(
        minimum=50, maximum=600, value=400, step=50, label="Maximum Words", info="Controls answer length"
    )

    historian_selector = gr.CheckboxGroup(
        choices=HISTORIAN_OPTIONS,
        label="Select Historian(s)",
        info="Choose one or more historians (leave empty for general answer)",
    )

    #  Persistent memory state
    memory_state = gr.State([])

    # Build UI
    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}\n{description}")

        gr.ChatInterface(
            fn=chat,
            additional_inputs=[style_dropdown, max_words_slider, historian_selector, memory_state],
            # additional_inputs_accordion=gr.Accordion("Advanced Options", open=False),
            examples=[
                ["When did Ashoka become Buddhist?"],
                ["According to Romilla Thappar how did he rule?"],
                ["How does this differ from RK Mukherjee's view on the same topic?"],

            ],
            show_progress=True,
        )

    # Launch with theme (Gradio 6.0+ requirement)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,           # Set to True if you want a public link
        theme=gr.themes.Soft(),
        show_error=True,
    )