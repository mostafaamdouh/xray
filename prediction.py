import joblib
import numpy as np
from PIL import Image
import os

# Load the Scikit-Learn Random Forest model
# This matches the "penguin-streamlit" logic exactly
model = joblib.load("xray_classifier.pkl")

def get_prediction(img_path):
    # 1. Load image
    img = Image.open(img_path).convert('RGB')
    
    # 2. Resize to 150x150 (exactly like the Kaggle training)
    img = img.resize((150, 150))
    
    # 3. Convert to numpy and normalize
    # Training was img / 255.0
    img_array = np.array(img) / 255.0
    
    # 4. Flatten the image
    # Random Forest expects (1, features)
    # Features = 150 * 150 * 3 = 67500
    flat_img = img_array.flatten().reshape(1, -1)
    
    # 5. Predict
    # returns [prob_class0, prob_class1, prob_class2]
    probs = model.predict_proba(flat_img)[0]
    
    return probs
