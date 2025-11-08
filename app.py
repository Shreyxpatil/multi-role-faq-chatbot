import streamlit as st
import requests

BASE_URL = "http://127.0.0.1:5000"  # your Flask backend

ROLES = [
    "student", "faculty", "coordinator",
    "controller", "tpo", "corporate", "admin"
]

st.set_page_config(page_title="Multi-Role Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Multi-Role Chatbot")

# Role selector
role = st.selectbox("Select your role:", ROLES)

# Keep messages for each role separately
if "conversations" not in st.session_state:
    st.session_state.conversations = {r: [] for r in ROLES}

messages = st.session_state.conversations[role]

# Show conversation
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
if prompt := st.chat_input(f"Ask me anything as a {role}..."):
    # Add user message
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend
    url = f"{BASE_URL}/api/{role}/ask"
    try:
        resp = requests.post(url, json={"query": prompt}, timeout=30)
        if resp.status_code == 200:
            answer = resp.json().get("answer", "No response.")
        else:
            answer = f"Error {resp.status_code}: {resp.text}"
    except Exception as e:
        answer = f"❌ Could not reach backend: {e}"

    # Add assistant message
    messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
