import streamlit as st
import json
from rag_engine import answer_query

st.title("User Persona Chatbot")

# Load Persona
with open("persona.json", "r") as f:
    persona = json.load(f)

with st.expander("View Extracted Persona Profile"):
    st.json(persona)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the user (e.g., 'What are their habits?'):"):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get RAG response
    with st.chat_message("assistant"):
        response = answer_query(prompt, persona)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})