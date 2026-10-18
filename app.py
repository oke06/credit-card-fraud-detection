# app.py — serves fraud predictions from the trained model over HTTP

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# --- load artifacts ONCE at startup (loading is slow; don't redo per request) ---
model = joblib.load("models/fraud_model.joblib")
scaler = joblib.load("models/scaler.joblib")

# the operating threshold chosen from the precision/recall trade-off (see notebook)
THRESHOLD = 0.3

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Fraud detection API. Go to /docs to try it."}

# --- define the shape of an incoming transaction ---
# one field per feature the model expects, in the same order as training
class Transaction(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float
    V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float
    Amount: float

@app.post("/predict")
def predict(txn: Transaction):
    # turn the incoming JSON into a one-row DataFrame with the right columns
    data = pd.DataFrame([txn.dict()])

    # the four-step dance from the sanity check:
    scaled = scaler.transform(data)                 # 1. scale with the SAME scaler
    prob = model.predict_proba(scaled)[:, 1][0]     # 2. get fraud probability
    is_fraud = prob >= THRESHOLD                     # 3. apply the threshold

    return {"fraud": bool(is_fraud), "probability": float(prob)}  # 4. send answer back