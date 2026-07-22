import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Path to your dataset folders
dataset_path = r"C:\Users\anjal\Desktop"
categories = ['Genuine', 'Counterfeit']
denominations = ['10', '20', '50', '100', '200', '500']

data = []
labels = []

print("🔄 Starting dataset scan and feature extraction...")

# 1. Load Genuine Sub-folders
genuine_path = os.path.join(dataset_path, 'Genuine')
if os.path.exists(genuine_path):
    for denom in denominations:
        denom_path = os.path.join(genuine_path, denom)
        if os.path.exists(denom_path):
            for img_name in os.listdir(denom_path):
                img_path = os.path.join(denom_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (64, 64))
                    feature_vector = img.flatten()
                    data.append(feature_vector)
                    labels.append(f"Genuine_{denom}")

# 2. Load Counterfeit Folder
counterfeit_path = os.path.join(dataset_path, 'Counterfeit')
if os.path.exists(counterfeit_path):
    for img_name in os.listdir(counterfeit_path):
        img_path = os.path.join(counterfeit_path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            feature_vector = img.flatten()
            data.append(feature_vector)
            labels.append("Counterfeit")

# 3. Train the AI Model
if len(data) > 0:
    X = np.array(data)
    y = np.array(labels)
    
    print(f"🧠 Training AI brain on {len(X)} images...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    with open('currency_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Success! 'currency_classifier.pkl' has been created and saved.")
else:
    print("❌ Error: No images were found. Double check your folder names and paths!")