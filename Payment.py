import sys
import json

from ImageCapture import capture_image
from EmbeddingGeneration import generate_embedding, cosine_similarity

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Payment.py <cost>")
        sys.exit(1)

    cost = int(sys.argv[1])

    image = capture_image()

    embedding = generate_embedding(image)

    with open("database.json", 'r') as f:
        db = json.load(f)

    best_similarity = -1
    idx = -1
    name = ""

    for i, entry in enumerate(db):
        db_embedding = entry.get("embedding")
        
        similarity = cosine_similarity(embedding, db_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            idx = i
            name = entry.get("name")
    new_balance = db[idx].get("balance") - cost
    if new_balance < 0:
        print("Transaction failed. Insufficient funds.")
        sys.exit(1)
    else:
        print("Transaction passed for " + name + ". New balance: " + str(new_balance))

    db[idx]["balance"] = new_balance

    with open("database.json", 'w') as f:
        json.dump(db, f, indent=4)

    
