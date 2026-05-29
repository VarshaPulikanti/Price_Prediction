from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.config import MODEL_PATH
from src.predict import predict_price
from src.train import train_model

app = FastAPI(title="California Housing Prediction API", version="1.0.0")


class HouseFeatures(BaseModel):
    MedInc: float = Field(..., gt=0)
    HouseAge: float = Field(..., ge=1)
    AveRooms: float = Field(..., gt=0)
    AveBedrms: float = Field(..., gt=0)
    Population: float = Field(..., gt=0)
    AveOccup: float = Field(..., gt=0)
    Latitude: float
    Longitude: float


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


@app.post("/train")
def train():
    metrics = train_model()
    return {"message": "training completed", "metrics": metrics}


@app.post("/predict")
def predict(payload: HouseFeatures):
    prediction = predict_price(payload.model_dump())
    return {
        "predicted_house_value_100k_units": prediction,
        "predicted_house_value_usd": round(prediction * 100000, 2),
    }
