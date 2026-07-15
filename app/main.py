from fastapi import FastAPI, UploadFile, File
from app.model_loader import load_models
from app.predictor import predict_genre, predict_instrument
import shutil
import os


app = FastAPI(
    title="EchoMind API",
    description="AI Audio Analysis API",
    version="1.0"
)


# Load models once when server starts
models = load_models()

(
    genre_model,
    genre_feature_extractor,
    genre_encoder,
    instrument_model,
    instrument_feature_extractor,
    instrument_encoder
) = models


@app.get("/")
def home():
    return {
        "message": "EchoMind API is running!"
    }


@app.post("/predict")
async def predict(audio_file: UploadFile = File(...)):

    audio_path = f"temp_{audio_file.filename}"

    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    genre, genre_conf = predict_genre(
        audio_path,
        genre_model,
        genre_feature_extractor,
        genre_encoder
    )

    instrument, instrument_conf = predict_instrument(
        audio_path,
        instrument_model,
        instrument_feature_extractor,
        instrument_encoder
    )

    os.remove(audio_path)

    return {
        "filename": audio_file.filename,
        "genre": genre,
        "genre_confidence": round(genre_conf * 100, 2),
        "instrument": instrument,
        "instrument_confidence": round(instrument_conf * 100, 2)
    }