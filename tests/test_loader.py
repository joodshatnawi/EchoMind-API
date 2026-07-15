from app.model_loader import load_models


models = load_models()

print("Models loaded successfully!")

print(type(models[0]))
print(type(models[3]))