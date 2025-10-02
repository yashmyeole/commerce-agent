# backend/backend_app/main.py
import os
import json
import numpy as np
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel

# sentence-transformers model (already installed in previous step)
from sentence_transformers import SentenceTransformer

# ---------- Config ----------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
CATALOG_PATH = os.path.join(DATA_DIR, "catalog_with_text_embeddings.json")
MODEL_NAME = "all-MiniLM-L6-v2"  # same model used to seed
DEFAULT_TOP_K = 5

# ---------- App ----------
app = FastAPI(title="Commerce Agent Demo Backend")

# ---------- Health / Echo (kept from prior scaffold) ----------
class HealthOut(BaseModel):
    status: str
    message: str

@app.get("/", response_model=HealthOut)
async def root():
    return HealthOut(status="ok", message="Commerce Agent Backend is running")

class EchoIn(BaseModel):
    text: str

class EchoOut(BaseModel):
    text: str

@app.post("/api/echo", response_model=EchoOut)
async def echo(payload: EchoIn):
    return EchoOut(text=payload.text)

# ---------- Load model & catalog at startup ----------
print("Loading embedding model:", MODEL_NAME)
embed_model = SentenceTransformer(MODEL_NAME)

if not os.path.exists(CATALOG_PATH):
    print(f"WARNING: Catalog not found at {CATALOG_PATH}. Make sure you ran the seed script.")
    catalog = []
    catalog_embeddings = np.zeros((0, embed_model.get_sentence_embedding_dimension()))
else:
    print("Loading catalog with embeddings from:", CATALOG_PATH)
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    # Build numpy matrix of embeddings for fast computation
    catalog_embeddings = []
    for p in catalog:
        emb = p.get("text_embedding")
        if emb is None:
            # If catalog lacks embeddings, compute here (fallback)
            txt = (p.get("title", "") or "") + ". " + (p.get("description", "") or "")
            emb = embed_model.encode(txt, convert_to_numpy=True).tolist()
            p["text_embedding"] = emb
        catalog_embeddings.append(np.array(emb, dtype=np.float32))
    if catalog_embeddings:
        catalog_embeddings = np.vstack(catalog_embeddings)
    else:
        catalog_embeddings = np.zeros((0, embed_model.get_sentence_embedding_dimension()))

print("Catalog items loaded:", len(catalog))
print("Embeddings matrix shape:", catalog_embeddings.shape)

# ---------- Helper: cosine similarity ----------
def cosine_sim(a: np.ndarray, b: np.ndarray):
    """
    Compute cosine similarities between a single vector a (shape D,)
    and matrix b (shape N x D).
    Returns array shape (N,)
    """
    # normalize to avoid divide-by-zero
    a_norm = np.linalg.norm(a)
    b_norms = np.linalg.norm(b, axis=1)
    if a_norm == 0:
        return np.zeros(b.shape[0])
    # dot product
    dots = b.dot(a)
    # safe division
    denom = b_norms * a_norm
    # avoid division by zero
    denom_safe = np.where(denom == 0, 1e-8, denom)
    sims = dots / denom_safe
    return sims

# ---------- Request/Response models for text search ----------
class TextSearchIn(BaseModel):
    query: str
    top_k: Optional[int] = DEFAULT_TOP_K

class ProductOut(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    category: Optional[str] = None
    price: Optional[float] = None
    image_url: Optional[str] = None
    score: float

class TextSearchOut(BaseModel):
    query: str
    results: List[ProductOut]

# ---------- Endpoint: POST /api/search/text ----------
@app.post("/api/search/text", response_model=TextSearchOut)
async def search_text(payload: TextSearchIn):
    q = payload.query
    top_k = payload.top_k or DEFAULT_TOP_K

    if not q or len(q.strip()) == 0:
        return TextSearchOut(query=q, results=[])

    # encode query
    q_emb = embed_model.encode(q, convert_to_numpy=True)
    # compute similarities
    if catalog_embeddings.shape[0] == 0:
        return TextSearchOut(query=q, results=[])

    sims = cosine_sim(q_emb, catalog_embeddings)  # shape (N,)
    # get top_k indices
    top_k = min(top_k, len(sims))
    top_idx = np.argsort(-sims)[:top_k]  # descending
    results = []
    for idx in top_idx:
        prod = catalog[idx]
        results.append(ProductOut(
            id=prod.get("id"),
            title=prod.get("title"),
            description=prod.get("description"),
            category=prod.get("category"),
            price=prod.get("price"),
            image_url=prod.get("image_url"),
            score=float(sims[idx])
        ))

    return TextSearchOut(query=q, results=results)
