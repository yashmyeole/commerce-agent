# backend/scripts/seed_image_embeddings.py
import os
import json
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
CAT_IN = os.path.join(DATA_DIR, "catalog_with_text_embeddings.json")
CAT_OUT = os.path.join(DATA_DIR, "catalog_with_image_embeddings.json")

MODEL_NAME = "openai/clip-vit-base-patch32"  # good default

def load_catalog(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_catalog(catalog, path):
    # convert numpy to lists
    for p in catalog:
        if "image_embedding" in p and isinstance(p["image_embedding"], np.ndarray):
            p["image_embedding"] = p["image_embedding"].tolist()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

def compute_image_embedding(image_path, model, processor, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    emb = outputs.cpu().numpy().astype(float).reshape(-1)
    # normalize
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Loading CLIP model:", MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    catalog = load_catalog(CAT_IN)

    # Ensure image folder exists
    for p in catalog:
        img_url = p.get("image_url")
        if not img_url:
            continue
        # accept either absolute path or data/images/...
        if img_url.startswith("data/") or img_url.startswith("./data/"):
            # leave as-is
            img_path = os.path.join(REPO_ROOT, img_url) if not os.path.isabs(img_url) else img_url
        else:
            # assume name only, construct path
            img_path = os.path.join(DATA_DIR, "images", os.path.basename(img_url))
        if not os.path.exists(img_path):
            print(f"WARNING: image not found for product {p.get('id')}: {img_path}")
            continue
        print("Embedding image for:", p.get("id"), img_path)
        emb = compute_image_embedding(img_path, model, processor, device)
        p["image_embedding"] = emb.tolist()

    print("Saving augmented catalog to:", CAT_OUT)
    save_catalog(catalog, CAT_OUT)
    print("Done.")

if __name__ == "__main__":
    main()
