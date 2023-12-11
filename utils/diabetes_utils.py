# app/utils/diabetes_utils.py

import pickle

def load_diabetes_model():
    return pickle.load(open('app/models/diabetes_model.sav', 'rb'))

def predict_diabetes(features):
    model = load_diabetes_model()
    prediction = model.predict(features)
    return prediction
