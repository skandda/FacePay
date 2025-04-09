import sys
import json
import os

from ImageCapture import capture_image
from EmbeddingGeneration import generate_embedding

def update_json(js):
    if not os.path.exists("database.json") or os.path.getsize("database.json") == 0:
        db = []
    else:
        with open("database.json", 'r') as f:
            db = json.load(f)

    db = [user for user in db if user.get("name") != js.get("name")]

    db.append(js)

    with open("database.json", 'w') as f:
        json.dump(db, f, indent=4)

    print("You have successfully been added to the database.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python UserUpload.py <name> <starting balance>")
        sys.exit(1)

    name = str(sys.argv[1])
    balance = int(sys.argv[2])

    image = capture_image()

    embedding = generate_embedding(image)
    embedding = embedding.tolist()

    js = {"name": name, "balance": balance, "embedding": embedding}

    update_json(js)



