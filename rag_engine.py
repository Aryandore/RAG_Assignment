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
    prompt = f"""
    You are a strict, factual AI assistant built to answer questions about a user based ONLY on their provided conversation history and persona.
    
    Here is their extracted Persona:
    {json.dumps(persona_data, indent=2)}
    
    Here are the most relevant past topic summaries:
    {context_summaries}
    
    Here are the most relevant raw conversation chunks:
    {context_chunks}
    
    CRITICAL RULES:
    1. Answer the user's question clearly and concisely based ONLY on the provided text above.
    2. If the answer is NOT explicitly stated in the context summaries or chunks, you MUST reply exactly with: "I cannot find the answer to this in the conversation history." 
    3. DO NOT guess, make up locations, or hallucinate information.
    
    Question: {user_question}
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