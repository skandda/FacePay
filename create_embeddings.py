import clip
import torch
from PIL import Image

def create_embed(img_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    
    image = Image.open(img_path)
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad(): embedding = model.encode_image(image)

    return embedding
        