from app.model_loader import load_models
models=load_models()
(
    genre_model,
    genre_feature_extractor,
    genre_encoder,
    instrument_model,
    instrument_feature_extractor,
    instrument_encoder
) = models
from app.predictor import predict_genre, predict_instrument
audio_path = "010__[sax][nod][cla]1692__3.wav"

genre_name, genre_conf = predict_genre(
    audio_path,
    genre_model,
    genre_feature_extractor,
    genre_encoder
)
instrument_name, instrument_conf = predict_instrument(
    audio_path,
    instrument_model,
    instrument_feature_extractor,
    instrument_encoder
)
print("Genre:", genre_name)
print("Genre Confidence:", f"{genre_conf*100:.2f}%")

print("Instrument:", instrument_name)
print("Instrument Confidence:", f"{instrument_conf*100:.2f}%")