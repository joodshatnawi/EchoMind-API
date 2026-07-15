import pickle
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor


def load_models():

    genre_model = AutoModelForAudioClassification.from_pretrained(
        "models/genre_model"
    )

    genre_feature_extractor = AutoFeatureExtractor.from_pretrained(
        "models/genre_model"
    )

    with open("models/label_encoder_genre.pkl", "rb") as f:
        genre_encoder = pickle.load(f)


    instrument_model = AutoModelForAudioClassification.from_pretrained(
        "models/irmas_model"
    )

    instrument_feature_extractor = AutoFeatureExtractor.from_pretrained(
        "models/irmas_model"
    )

    with open("models/label_encoder_IRMAS.pkl", "rb") as f:
        instrument_encoder = pickle.load(f)


    return (
        genre_model,
        genre_feature_extractor,
        genre_encoder,
        instrument_model,
        instrument_feature_extractor,
        instrument_encoder
    )
