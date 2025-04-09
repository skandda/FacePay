from PIL import Image
import clip
import torch
import cv2
import json
import numpy as np
from numpy.linalg import norm

def cosine_similarity(a, b):
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-8)

def generate_embedding(arr):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    
    if arr is not None:
        img_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        image_input = preprocess(img_pil).unsqueeze(0).to(device)  # Shape: [1, 3, 224, 224]

    with torch.no_grad(): embedding = model.encode_image(image_input)

    return embedding