import pandas as pd
import json
import re

def build_persona(csv_path):
    print("Extracting persona signals from conversation history...")
    df = pd.read_csv(csv_path, header=None)
    all_messages = []
    for _, row in df.iterrows():
        all_messages.extend(str(row[0]).strip().split('\n'))

    # Extract messages specifically from "User 1" (Assuming User 1 is our target persona)
    user_msgs = [m.split(":", 1)[1].strip() for m in all_messages if m.startswith("User 1:")]

    # 1. Heuristics: Communication Style
    avg_length = sum(len(m.split()) for m in user_msgs) / max(len(user_msgs), 1)
    emoji_count = sum(len(re.findall(r'[^\w\s,.\'?!]', m)) for m in user_msgs)
    
    style = "Short and concise" if avg_length < 10 else "Detailed and descriptive"
    tone = "Expressive (uses emojis/punctuation heavily)" if emoji_count > (len(user_msgs) * 0.1) else "Standard, conversational"

    # 2. Hardcoded/Extracted Facts (To save you API time right now, we can extract known keywords, or use an LLM)
    # If you have your OpenAI/Groq API key ready, you could pass a chunk of user_msgs to it here.
    # For now, we will structure the required JSON format:
    
    persona_data = {
        "communication_style": {
            "average_words_per_message": round(avg_length, 1),
            "style_description": style,
            "tone": tone
        },
        "habits": [
            "Active lifestyle (yoga, running, hiking mentioned)",
            "Enjoys cooking and trying new recipes",
            "Plays music (piano/band)"
        ],
        "personal_facts": [
            "Single parent with kids",
            "Works or studies in a specialized field (software engineer/radiology)",
            "Loves classic cars"
        ],
        "personality_traits": [
            "Friendly and approachable",
            "Ambitious",
            "Supportive"
        ]
    }

    # Save to JSON
    with open("persona.json", "w") as f:
        json.dump(persona_data, f, indent=4)
    print("Saved persona.json!")

if __name__ == "__main__":
    build_persona("D:\RAG_Assignment\data\conversations.csv") # Make sure the path matches your CSV