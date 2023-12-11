# app/utils/diabetes_utils.py

import pickle

def load_parkinsons_model():
    return pickle.load(open('app/models/parkinsons_model.sav', 'rb'))

def predict_parkinsons(features):
    model = load_parkinsons_model()
    prediction = model.predict(features)
    return prediction
