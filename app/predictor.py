import librosa
import torch
def predict_genre(audio_path,genre_model,genre_feature_extractor,genre_encoder):
    audio,sr= librosa.load(audio_path,sr=16000,duration=5)
    inputs = genre_feature_extractor(audio,sampling_rate=sr,return_tensors="pt")
    with torch.no_grad():
        outputs = genre_model(**inputs)
        probs = torch.softmax(outputs.logits,dim=1)
        pred_idx=torch.argmax(probs,dim=1).item()
        genre_name=genre_encoder.inverse_transform([pred_idx])[0]
        confidence = probs[0][pred_idx].item()
        return genre_name,confidence
    
def predict_instrument(audio_path, instrument_model, instrument_feature_extractor, instrument_encoder):    
    audio,sr= librosa.load(audio_path,sr=16000,duration=3)
    inputs = instrument_feature_extractor(audio,sampling_rate=sr,return_tensors="pt")
    with torch.no_grad():
        outputs = instrument_model(**inputs)
        probs = torch.softmax(outputs.logits,dim=1)
        pred_idx=torch.argmax(probs,dim=1).item()
        instrument_name=instrument_encoder.inverse_transform([pred_idx])[0]
        confidence = probs[0][pred_idx].item()
        return instrument_name,confidence
    