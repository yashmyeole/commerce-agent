# backend/scripts/seed_text_embeddings.py
import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
CATALOG_IN = os.path.join(DATA_DIR, "catalog.json")
CATALOG_OUT = os.path.join(DATA_DIR, "catalog_with_text_embeddings.json")

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, free

def load_catalog(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_catalog(catalog, path):
    # Convert numpy arrays to lists for JSON
    for p in catalog:
        if "text_embedding" in p and isinstance(p["text_embedding"], np.ndarray):
            p["text_embedding"] = p["text_embedding"].tolist()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

def main():
    print("Loading model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    print("Loading catalog:", CATALOG_IN)
    catalog = load_catalog(CATALOG_IN)

    texts = []
    for p in catalog:
        txt = (p.get("title", "") or "") + ". " + (p.get("description", "") or "")
        texts.append(txt)

    print("Computing embeddings for", len(texts), "products...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    for p, emb in zip(catalog, embeddings):
        p["text_embedding"] = emb.tolist()  # store as list

    print("Saving catalog with embeddings to:", CATALOG_OUT)
    save_catalog(catalog, CATALOG_OUT)
    print("Done.")

if __name__ == "__main__":
    main()
