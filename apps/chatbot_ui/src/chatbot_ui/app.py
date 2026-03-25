import streamlit as st
import requests
from chatbot_ui.core.config import config

# Set page config
st.set_page_config(page_title="Law Assistant", page_icon="⚖️", layout="centered")

st.title("⚖️ Legal Acts Assistant")
st.caption(
    "Ask legal questions and get answers grounded in indexed laws with source citations."
)

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")

    max_tokens = st.number_input(
        "Max Tokens", min_value=10, max_value=8192, value=500, step=50
    )
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
        st.markdown(turn["answer"])
        sources = turn.get("sources") or []
        if sources:
            with st.expander("Sources", expanded=False):
                for source in sources:
                    act_title = source.get("act_title") or "Unknown Act"
                    act_year = source.get("act_year")
                    section = source.get("section_index") or "Unknown"
                    score = float(source.get("score", 0.0))
                    source_url = source.get("source_url")
                    citation_id = source.get("citation_id")
                    excerpt = source.get("excerpt") or "No excerpt available."

                    year_suffix = f" ({act_year})" if act_year else ""
                    st.markdown(
                        f"**[Source {citation_id}] {act_title}{year_suffix}, Section {section}**  "
                        f"Similarity score: `{score:.4f}`"
                    )
                    if source_url:
                        st.markdown(f"[{source_url}]({source_url})")
                    st.caption(excerpt)
                    st.divider()

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
                "max_tokens": max_tokens,
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
            message_placeholder.markdown(assistant_response)

            if sources:
                with st.expander("Sources", expanded=True):
                    for source in sources:
                        act_title = source.get("act_title") or "Unknown Act"
                        act_year = source.get("act_year")
                        section = source.get("section_index") or "Unknown"
                        score = float(source.get("score", 0.0))
                        source_url = source.get("source_url")
                        citation_id = source.get("citation_id")
                        excerpt = source.get("excerpt") or "No excerpt available."

                        year_suffix = f" ({act_year})" if act_year else ""
                        st.markdown(
                            f"**[Source {citation_id}] {act_title}{year_suffix}, Section {section}**  "
                            f"Similarity score: `{score:.4f}`"
                        )
                        if source_url:
                            st.markdown(f"[{source_url}]({source_url})")
                        st.caption(excerpt)
                        st.divider()

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
