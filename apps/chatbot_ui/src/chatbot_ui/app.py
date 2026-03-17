import streamlit as st
import requests
from chatbot_ui.core.config import config

# Set page config
st.set_page_config(page_title="Chatbot UI", page_icon="💬", layout="centered")

st.title("💬 Chatbot Multi-Provider")
st.caption(
    "A Streamlit chatbot powered by FastAPI backend supporting OpenAI, Groq, and Gemini."
)

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")

    provider = st.selectbox("Provider", options=["openai", "groq", "gemini"], index=0)

    # Simple default model logic based on provider
    default_model = "openai/gpt-5.4-nano"
    if provider == "groq":
        default_model = "llama-3.1-8b-instant"
    elif provider == "gemini":
        default_model = "gemini-2.5-flash"

    model_name = st.text_input("Model Name", value=default_model)
    max_tokens = st.number_input(
        "Max Tokens", min_value=10, max_value=8192, value=500, step=50
    )

    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # Call the FastAPI backend
        try:
            payload = {
                "provider": provider,
                "model_name": model_name,
                "messages": st.session_state.messages,
                "max_tokens": max_tokens,
            }

            with st.spinner("Thinking..."):
                response = requests.post(f"{config.API_URL}/chat", json=payload)
                response.raise_for_status()

            response_data = response.json()
            assistant_response = response_data.get(
                "response", "Error: No response from server."
            )
            message_placeholder.markdown(assistant_response)

            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_response}
            )

        except requests.exceptions.ConnectionError:
            error_msg = (
                f"Connection Error: Could not reach the API at {config.API_URL}."
            )
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP Error: {e.response.status_code} - {e.response.text}"
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )
        except Exception as e:
            error_msg = f"An unexpected error occurred: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )
