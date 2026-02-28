from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fastapi import FastAPI
from pydantic import BaseModel
from app.model import load_model
import os
import joblib
from huggingface_hub import hf_hub_download

"""
GET /healthz

GET /readyz

POST /infer
"""


MODEL_REPO = "radhaagr/ipl_cricket_lora"
MODEL_FILE = "ipl_cricket_lora"

def load_model():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable not set")

    model_path = hf_hub_download( repo_id=MODEL_REPO,
                                  filename=MODEL_FILE,
                                  token=hf_token
                                )
    model = joblib.load(model_path)
    return model

    
app = FastAPI(title="IPL Inference Service")

# Load model at startup
model = load_model()

# -------------------------
# Schemas
# -------------------------
class IPLRequest(BaseModel):
    batting_team: str
    bowling_team: str
    venue: str
    runs: int
    wickets: int
    overs: float

class IPLResponse(BaseModel):
    prediction: float

# -------------------------
# Health checks
# -------------------------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/readyz")
def readyz():
    if model is None:
        return {"ready": False}
    return {"ready": True}

# -------------------------
# Inference endpoint
# -------------------------
@app.post("/infer", response_model=IPLResponse)
def infer(request: IPLRequest):
    features = [[ request.runs,
                  request.wickets,
                  request.overs
                    # categorical features should be encoded during training
               ]]

    prediction = model.predict(features)[0]

    return {"prediction": float(prediction)}


app = FastAPI(title='Cricket Inference Service', version='0.1.0')

class InferRequest(BaseModel):
    prompt: str

@app.get('/healthz')
def health():
    return {'status': 'ok'}

@app.get('/readyz')
def ready():
    return {'ready': True}

@app.post('/infer')
def infer(req: InferRequest):
    if not req.prompt:
        raise HTTPException(status_code=400, detail='Prompt required')
    # TODO: load model from private HF repo using HF_TOKEN; call with req.prompt + computed stats context
    return {'answer': f"Stub answer to: {req.prompt}"}
