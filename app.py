import sys
import json
import numpy as np

from capture import capture_image
from create_embeddings import create_embed

def cosine_similarity(a, b):
    # Ensure that a and b are flattened to 1D arrays
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python app.py <name> <intent>")
        sys.exit(1)

    name = str(sys.argv[1]) + ".png"
    intent = int(sys.argv[2])

    with open("embeddings.json", "r") as f:
            content = f.read().strip()  # Read and strip any surrounding whitespace
            if not content:
                data = []  # Return empty list if file is empty
            else:
                data = json.loads(content)  # Try to load JSON data

    capture_image(name)

    embed = create_embed(name)
    embed = embed.cpu().numpy().tolist()
    data_entry = {"name": name, "embedding": embed}

    if intent == 0:

        max_sim = -1
        most_similar = None

        for entry in data:
            entry_embedding = entry["embedding"]  # Accessing the embedding for each entry
            sim = cosine_similarity([entry_embedding], [embed])  # Assuming `query_embedding` is your input embedding
            if sim > max_sim:
                max_sim = sim
                most_similar = entry["name"]

        print("You are: " + most_similar)

    if intent == 1:
        data.append(data_entry)

        with open("embeddings.json", "w") as f:
            json.dump(data, f, indent=2)