import os
import joblib
from datetime import datetime

def save_model(model, metadata, folder="models"):
    os.makedirs(folder, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{folder}/model_{timestamp}.pkl"
    meta_path = f"{folder}/model_{timestamp}_meta.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(metadata, meta_path)
    
    return model_path, meta_path