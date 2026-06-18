import pandas as pd
from sentence_transformers import SentenceTransformer, util
import chromadb

# 1. Initialize Local Models & DB
embedder = SentenceTransformer('all-MiniLM-L6-v2') # Fast, local model
chroma_client = chromadb.PersistentClient(path="./vector_db")
summary_collection = chroma_client.get_or_create_collection(name="summaries")
chunk_collection = chroma_client.get_or_create_collection(name="chunks")

def parse_conversations(csv_path):
    """Flattens the CSV into a single chronological list of messages."""
    df = pd.read_csv(csv_path, header=None)
    all_messages = []
    for _, row in df.iterrows():
        # Split the multi-line string into individual messages
        messages = str(row[0]).strip().split('\n')
        all_messages.extend(messages)
    return all_messages

def generate_summary(messages):
    """Call your LLM API here to summarize a list of messages."""
    text = "\n".join(messages)
    # Pseudo-code: return llm_api.generate(f"Summarize this: {text}")
    return "Summary of messages..."

def process_data(messages):
    current_topic_msgs = []
    current_topic_embedding = None
    
    # Checkpoint trackers
    msg_counter = 0
    topic_counter = 1
    
    for msg in messages:
        msg_counter += 1
        current_topic_msgs.append(msg)
        
        # Save raw chunk every 5 messages for granular RAG retrieval
        if msg_counter % 5 == 0:
            chunk_text = "\n".join(current_topic_msgs[-5:])
            chunk_collection.add(
                documents=[chunk_text],
                metadatas=[{"start": msg_counter-4, "end": msg_counter, "type": "raw_chunk"}],
                ids=[f"chunk_{msg_counter}"]
            )

        # 100-Message Checkpoint
        if msg_counter % 100 == 0:
            summary = generate_summary(messages[msg_counter-100 : msg_counter])
            summary_collection.add(
                documents=[summary],
                metadatas=[{"type": "100_msg_checkpoint", "end_msg": msg_counter}],
                ids=[f"100_chkpt_{msg_counter}"]
            )

        # Topic Detection Logic
        msg_emb = embedder.encode(msg)
        if current_topic_embedding is None:
            current_topic_embedding = msg_emb
        else:
            # Check similarity
            similarity = util.cos_sim(current_topic_embedding, msg_emb).item()
            
            if similarity < 0.35 and len(current_topic_msgs) > 5: # Threshold for topic change
                # 1. Topic Changed! Create Checkpoint
                summary = generate_summary(current_topic_msgs)
                
                # 2. Store in Vector DB
                summary_collection.add(
                    documents=[summary],
                    metadatas=[{"type": "topic_checkpoint", "topic_id": topic_counter}],
                    ids=[f"topic_{topic_counter}"]
                )
                
                # 3. Reset for next topic
                print(f"Topic {topic_counter} -> messages {msg_counter-len(current_topic_msgs)} to {msg_counter} -> Saved")
                topic_counter += 1
                current_topic_msgs = [msg]
                current_topic_embedding = msg_emb
            else:
                # Update running topic embedding (moving average)
                current_topic_embedding = (current_topic_embedding + msg_emb) / 2

if __name__ == "__main__":
    msgs = parse_conversations("data/conversations.csv")
    process_data(msgs)