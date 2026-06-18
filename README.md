Here is a shorter, more natural-sounding version. It sounds less like a textbook and more like a real engineer who is proud of their work but honest about its limits.

AI Engineer Intern Assignment: RAG & Persona Chatbot
Here is my submission for the AI/ML Engineer Intern role! I built an end-to-end RAG system that reads chronological chat histories, detects natural topic changes, extracts user personas, and answers questions using a local vector database and the Groq API.

🔗 Live Links
Live Chatbot: [Insert your Streamlit URL here]

Video Demo: [Insert your Loom video URL here]

🛠️ How It Works
1. Smart Topic Splitting
Instead of just blindly chunking the text, I processed the messages chronologically. I used local sentence embeddings (all-MiniLM-L6-v2) to track the conversation. If the cosine similarity drops below a certain threshold, the system knows the topic changed, creates a checkpoint, and summarizes it.

2. Accurate Retrieval (RAG)
To prevent hallucinations, the bot casts a wide net. When you ask a question, it retrieves both high-level topic summaries AND raw message chunks. I also used strict system prompts so the bot will admit "I don't know" instead of making up facts.

3. Data-Driven Persona
The instructions asked to base the persona on actual signals, not just ChatGPT guesses. I used Python logic to calculate the user's average word count and emoji usage to figure out their communication style mathematically, and stored it all in a clean JSON format.

⚖️ System Evaluation (The Good & The Bad)
Building this taught me a lot about handling messy conversational data. Here is an honest look at the architecture:

🌟 What Went Well
Fast & Local: The heavy lifting (topic detection and embeddings) runs locally using basic tensor math, so it's incredibly fast and cheap. I also avoided heavy wrappers like LangChain to ensure I had total control over the chronological parsing.

⚠️ What I'd Improve (Known Limitations)
The Chunk Boundary Problem: Right now, raw messages are saved in rigid blocks of 5. Sometimes, a question and its answer get split across two different chunks. In a V2, I would use overlapping sliding windows (e.g., chunking 10 messages with a 5-message overlap) to keep the context intact.

Semantic Dilution: The dataset has over 190,000 messages. If you ask a generic question like "Where are they moving?", the database finds dozens of people talking about moving. The system currently requires specific keywords in the prompt to find the exact needle in the haystack.

💻 How to Run Locally
Clone the repo and install dependencies:

Bash
pip install -r requirements.txt
Run the Chatbot:

Bash
streamlit run app.py