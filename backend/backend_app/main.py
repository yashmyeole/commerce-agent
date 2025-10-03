# backend/backend_app/main.py
import os
import json
import numpy as np
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image


origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- after app = FastAPI(...) and CORS middleware code, mount static images ----
# serve product images from /static/images/
IMAGES_DIR = os.path.join(DATA_DIR, "images")
if os.path.isdir(IMAGES_DIR):
    app.mount("/static/images", StaticFiles(directory=IMAGES_DIR), name="static-images")
else:
    print("Warning: images dir not found:", IMAGES_DIR)

# ---- load CLIP model & image embeddings at startup (below text embedding loading) ----
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading CLIP model for image embeddings:", CLIP_MODEL_NAME, "on", device)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

# Try to load catalog with image embeddings if available
CAT_IMG_PATH = os.path.join(DATA_DIR, "catalog_with_image_embeddings.json")
if os.path.exists(CAT_IMG_PATH):
    print("Loading catalog with image embeddings from:", CAT_IMG_PATH)
    with open(CAT_IMG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)
else:
    print("No catalog_with_image_embeddings.json found; attempting to load with text embeddings.")
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

# build image_embeddings matrix and keep track of indices that have embeddings
image_embeddings_list = []
image_indices = []  # indices in catalog that have image embeddings
for idx, p in enumerate(catalog):
    emb = p.get("image_embedding")
    if emb:
        image_embeddings_list.append(np.array(emb, dtype=np.float32))
        image_indices.append(idx)

if image_embeddings_list:
    image_embeddings = np.vstack(image_embeddings_list)
else:
    image_embeddings = np.zeros((0, clip_model.visual_projection.out_features if hasattr(clip_model, 'visual_projection') else 512))
print("Loaded image embeddings matrix shape:", image_embeddings.shape, "for", len(image_indices), "products")

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

# ---- helper to compute CLIP image embedding (same normalization as seed) ----
def compute_clip_image_embedding_from_bytes(file_bytes):
    image = None
    from io import BytesIO
    try:
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        print("Error opening image:", e)
        raise HTTPException(status_code=400, detail="Invalid image uploaded.")

    
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        img_feats = clip_model.get_image_features(**inputs)
    emb = img_feats.cpu().numpy().astype(float).reshape(-1)
    # normalize
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb

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

# ---- add endpoint: POST /api/search/image ----
@app.post("/api/search/image", response_model=TextSearchOut)
async def search_image(file: UploadFile = File(...), top_k: Optional[int] = DEFAULT_TOP_K):
    # read bytes
    contents = await file.read()
    q_emb = compute_clip_image_embedding_from_bytes(contents)

    if image_embeddings.shape[0] == 0:
        # no image embeddings available
        return TextSearchOut(query=f"image:{file.filename}", results=[])

    # compute cosine similarities to image_embeddings matrix
    sims = cosine_sim(q_emb, image_embeddings)
    # get top indices within image_embeddings list then map to catalog index
    k = min(top_k or DEFAULT_TOP_K, len(sims))
    top_idx_local = np.argsort(-sims)[:k]
    results = []
    for local_idx in top_idx_local:
        catalog_idx = image_indices[local_idx]
        prod = catalog[catalog_idx]
        # rewrite image_url to static path if possible
        image_url = prod.get("image_url", "")
        # if image_url points to local data/images, convert to static route
        if image_url and (image_url.startswith("data/images") or image_url.startswith("./data/images") or os.path.basename(image_url) == image_url):
            filename = os.path.basename(image_url)
            prod_image_url = f"/static/images/{filename}"
        else:
            prod_image_url = image_url
        results.append(ProductOut(
            id=prod.get("id"),
            title=prod.get("title"),
            description=prod.get("description"),
            category=prod.get("category"),
            price=prod.get("price"),
            image_url=prod_image_url,
            score=float(sims[local_idx])
        ))

    return TextSearchOut(query=f"image:{file.filename}", results=results)