# app/utils/diabetes_utils.py

import pickle

def load_heart_model():
    return pickle.load(open('app/models/heart_disease_model.sav', 'rb'))

def predict_heart(features):
    model = load_heart_model()
    prediction = model.predict(features)
    return prediction
