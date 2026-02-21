from contextlib import asynccontextmanager
import io
from typing import List
from PIL import Image
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from geomatcher import Geomatcher

import torch, uvicorn

class GeomatchResult(BaseModel):
    id: str
    initial_rank: int
    vector_distance: float
    refined_rank: int
    matches: int
    latitude: float
    longitude: float
    elevation: float
    date: str

class GeomatchResponse(BaseModel):
    count: int
    results: List[GeomatchResult]

@asynccontextmanager
async def lifespan(app: FastAPI):
    FEATURES_PATH = '/mnt/storage-box-1/prague-streetview-50k-features-alikedn16-1024points-int8-2'
    HF_VECTOR_DB = 'josefbednar/prague-streetview-50k-vectors'

    vector_dim = 2048
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Initializing Geomatcher...")
    geomatcher = Geomatcher(FEATURES_PATH, HF_VECTOR_DB, vector_dim, device)

    app.state.geomatcher = geomatcher
    print("Ready for production.")
    yield
    

app = FastAPI(title="Geomatch", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "geomatch ok"}

@app.post('/query')
async def query(request: Request, image_file: UploadFile = File(...), top_k: int = 50):
    print(f'Received a query request.')
    if not image_file:
        raise HTTPException(status_code=400, detail='"image_file" must not be empty.')
    
    image_bytes = await image_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    try:
        _initial, refined = request.app.state.geomatcher.get_ranked(image, top_k, verbose=True, print_results=False)
        return GeomatchResponse(
            count=len(refined),
            results=refined
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Geomatching failed: {e}")

if __name__ == "__main__":
    print("Starting Geomatch API server...")
    uvicorn.run(app, host="0.0.0.0", port=1717)
