import streamlit as st
import chromadb
from groq import Groq
import json

# 1. Initialize Groq Client 
client = Groq(api_key=st.secrets["GROQ_API_KEY"])
# 2. Connect to local ChromaDB
chroma_client = chromadb.PersistentClient(path="./vector_db")

# Safely get collections (prevents errors if they are empty)
try:
    summary_collection = chroma_client.get_collection(name="summaries")
    chunk_collection = chroma_client.get_collection(name="chunks")
except Exception as e:
    print(f"Warning: Could not load collections. Make sure processor.py finished running. Error: {e}")

def answer_query(user_question, persona_data):
    try:
        # 1. Retrieve top 2 topic summaries
        summary_results = summary_collection.query(
            query_texts=[user_question],
            n_results=5
        )
        
        # 2. Retrieve top 3 raw conversation chunks
        chunk_results = chunk_collection.query(
            query_texts=[user_question],
            n_results=10
        )
        
        # Safely extract documents
        context_summaries = "\n".join(summary_results['documents'][0]) if summary_results['documents'] else "No summaries found."
        context_chunks = "\n".join(chunk_results['documents'][0]) if chunk_results['documents'] else "No chunks found."
        
    except Exception as e:
        context_summaries = "Error retrieving summaries."
        context_chunks = "Error retrieving chunks."

    # 3. Build Prompt
   # 3. Build Prompt
    prompt = f"""
    You are an intelligent, conversational AI assistant analyzing a specific user's chat history. 
    Your goal is to answer questions about this user accurately and naturally.

    === KNOWLEDGE BASE ===
    USER PERSONA:
    {json.dumps(persona_data, indent=2)}

    RELEVANT CONTEXT (Summaries & Chat Logs):
    {context_summaries}
    {context_chunks}

    === INSTRUCTIONS ===
    1. CONVERSATIONAL TONE: If the user input is a greeting (like "hi", "hello") or casual chat, respond politely as an AI assistant ready to help analyze the user's data. Do not search for facts if the user is just saying hello.
    2. FACTUAL RAG: Answer questions based EXCLUSIVELY on the Knowledge Base above. Do not guess, assume, or hallucinate facts outside this data.
    3. SMART FAILBACK: If the exact answer is not in the data, do not use robotic error messages. Instead, respond naturally. For example: "The chat logs don't mention their real name" or "The user doesn't mention owning a car, but they were talking to someone who owns a 1964 Impala."
    4. CLARITY: Keep your answers concise, direct, and helpful.

    User Input: {user_question}
    """
    
    # 4. Call Groq API (Using Llama 3)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", 
        messages=[
            {"role": "system", "content": "You are a helpful data assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3 
    )
    
    return response.choices[0].message.content