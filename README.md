# Conversational RAG & Persona Extractor

This project processes chronological chat logs, detects topic boundaries, extracts user personas, and allows users to query the data via a Streamlit Chatbot.

## 🧠 Approach & Logic

### 1. Topic Change Detection
Instead of sending every message to an LLM to check if the topic changed (which is slow and expensive), this system uses **lightweight local embeddings** (`SentenceTransformer('all-MiniLM-L6-v2')`). 
* A rolling buffer of recent messages is maintained.
* As a new message arrives, we compute the cosine similarity between the new message and the buffer. 
* If the similarity drops below a specific threshold (e.g., `< 0.25`), it signifies a **topic drift**. The buffer is flushed, summarized by the LLM, and saved as a Checkpoint in the vector database.

### 2. Retrieval Works (RAG)
We use **ChromaDB** as the local vector store. When a user asks a question:
1. The query is embedded and searched against ChromaDB to find the Top 3 most relevant Topic Summaries.
2. The system pulls the attached metadata, which includes the exact raw message chunks.
3. Both the summary and the raw chunks are passed to the LLM to synthesize the final answer.

### 3. Persona Building
The persona is extracted continuously at every **100-message checkpoint**. The LLM is prompted to extract facts strictly into a JSON schema (Habits, Personal Facts, Traits, Style). A Python function safely merges this new JSON into a global `persona.json` state, ensuring it updates dynamically over time without losing previous facts.

## 🚀 How to Run Locally

1. Install dependencies: `pip install -r requirements.txt`
2. Create a `.env` file and add your OpenAI API Key: `OPENAI_API_KEY=sk-...`
3. Run the backend processor (Takes a few minutes): `python process_data.py`
4. Run the Chatbot: `streamlit run app.py`