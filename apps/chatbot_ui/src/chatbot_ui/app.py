import streamlit as st
import requests
import re
from chatbot_ui.core.config import config

# Set page config
st.set_page_config(page_title="Law Assistant", page_icon="⚖️", layout="centered")

st.title("⚖️ Legal Acts Assistant")
st.caption(
    "Ask legal questions and get answers grounded in indexed laws with source citations."
)

st.markdown(
    """
<style>
.source-card {
    border: 1px solid rgba(49, 51, 63, 0.2);
    border-radius: 12px;
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.7rem;
    background: rgba(248, 249, 252, 0.7);
}
.source-meta {
    font-size: 0.88rem;
    color: #4f5b66;
}
</style>
""",
    unsafe_allow_html=True,
)


_SOURCE_REF_RE = re.compile(r"\[Source\s+(\d+)\]")


def _source_lookup(sources: list[dict]) -> dict[int, dict]:
    lookup: dict[int, dict] = {}
    for source in sources:
        citation_id = source.get("citation_id")
        if isinstance(citation_id, int):
            lookup[citation_id] = source
    return lookup


def _linkify_answer(answer: str, sources: list[dict]) -> str:
    lookup = _source_lookup(sources)

    def repl(match: re.Match[str]) -> str:
        citation_id = int(match.group(1))
        source = lookup.get(citation_id)
        if not source:
            return match.group(0)
        source_url = source.get("source_url")
        if not source_url:
            return match.group(0)
        return f"[Source {citation_id}]({source_url})"

    return _SOURCE_REF_RE.sub(repl, answer)


def _render_sources(sources: list[dict], expanded: bool) -> None:
    if not sources:
        return

    with st.expander("Sources", expanded=expanded):
        for source in sources:
            act_title = source.get("act_title") or "Unknown Act"
            act_year = source.get("act_year")
            section = source.get("section_index") or "Unknown"
            score = float(source.get("score", 0.0))
            source_url = source.get("source_url")
            citation_id = source.get("citation_id")
            excerpt = source.get("excerpt") or "No excerpt available."

            year_suffix = f" ({act_year})" if act_year else ""
            source_line = (
                f"<div class='source-card'>"
                f"<strong>[Source {citation_id}] {act_title}{year_suffix}, Section {section}</strong><br>"
                f"<span class='source-meta'>Similarity score: {score:.4f}</span>"
                f"</div>"
            )
            st.markdown(source_line, unsafe_allow_html=True)
            if source_url:
                st.markdown(f"[Open source]({source_url})")
            st.caption(excerpt)


def _render_assistant_turn(answer: str, sources: list[dict], expanded: bool) -> None:
    linked_answer = _linkify_answer(answer, sources)
    st.markdown(linked_answer)
    _render_sources(sources, expanded=expanded)


# Sidebar for configuration
with st.sidebar:
    st.header("Settings")

    top_k = st.slider("Sources to Retrieve", min_value=3, max_value=12, value=6, step=1)

    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.conversation = []
        st.rerun()

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# Display chat messages from history on app rerun
for turn in st.session_state.conversation:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        _render_assistant_turn(
            answer=turn["answer"],
            sources=turn.get("sources") or [],
            expanded=False,
        )

# Accept user input
if prompt := st.chat_input("Ask a legal question..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # Call the FastAPI backend
        try:
            payload = {
                "question": prompt,
                "top_k": top_k,
            }

            with st.spinner("Thinking..."):
                response = requests.post(
                    f"{config.API_URL}/rag/legal/chat", json=payload, timeout=120
                )
                response.raise_for_status()

            response_data = response.json()
            assistant_response = response_data.get(
                "answer", "Error: No answer returned."
            )
            sources = response_data.get("sources", [])
            message_placeholder.empty()
            _render_assistant_turn(
                answer=assistant_response,
                sources=sources,
                expanded=True,
            )

            st.session_state.conversation.append(
                {
                    "question": prompt,
                    "answer": assistant_response,
                    "sources": sources,
                }
            )

        except requests.exceptions.ConnectionError:
            error_msg = (
                f"Connection Error: Could not reach the API at {config.API_URL}."
            )
            st.error(error_msg)
            st.session_state.conversation.append(
                {"question": prompt, "answer": error_msg}
            )
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP Error: {e.response.status_code} - {e.response.text}"
            st.error(error_msg)
            st.session_state.conversation.append(
                {"question": prompt, "answer": error_msg}
            )
        except Exception as e:
            error_msg = f"An unexpected error occurred: {str(e)}"
            st.error(error_msg)
            st.session_state.conversation.append(
                {"question": prompt, "answer": error_msg}
            )
