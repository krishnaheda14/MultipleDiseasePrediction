# -*- coding: utf-8 -*-
"""
Created on Tue Feb 7 20:12:12 2023
@author: Admin
"""

import pickle

import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='templates')

# Loading the saved models
diabetes_model = pickle.load(open('C:/Users/Krishna/PycharmProjects/EDAIProject/models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('C:/Users/Krishna/PycharmProjects/EDAIProject/models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('C:/Users/Krishna/PycharmProjects/EDAIProject/models/parkinsons_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes_prediction():
    if request.method == 'POST':
        Pregnancies = request.form['Pregnancies']
        Glucose = request.form['Glucose']
        BloodPressure = request.form['BloodPressure']
        SkinThickness = request.form['SkinThickness']
        Insulin = request.form['Insulin']
        BMI = request.form['BMI']
        DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
        Age = request.form['Age']

        input_data = np.array(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        diab_prediction = diabetes_model.predict(input_data)

        if diab_prediction[0] == 1:
            result = 'Diabetes Detected'
            detection_warning = '*This prediction may be inaccurate, so it is better to consult a doctor'
        else:
            result = 'Diabetes not detected'
            detection_warning = ''

        return render_template('result.html', disease_detected=diab_prediction[0], disease_name='Diabetes', detection_warning=detection_warning)

    return render_template('diabetes.html')

@app.route('/heart', methods=['GET', 'POST'])
def heart_disease_prediction():
    if request.method == 'POST':
        age = request.form['Age']
        sex = request.form['Sex']
        cp = request.form['ChestPainType']
        trestbps = request.form['RestingBloodPressure']
        chol = request.form['Cholesterol']
        fbs = request.form['FastingBloodSugar']
        restecg = request.form['RestECG']
        thalach = request.form['MaxHeartRate']
        exang = request.form['ExerciseAngina']
        oldpeak = request.form['STDepression']
        slope = request.form['Slope']
        ca = request.form['NumVessels']
        thal = request.form['Thalassemia']

        heart_prediction = heart_disease_model.predict(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        if heart_prediction[0] == 1:
            result = 'Heart Disease Detected'
            detection_warning = ''
        else:
            result = 'Heart Disease not detected'
            detection_warning = ''

        return render_template('result.html', disease_detected=heart_prediction[0], disease_name='Heart Disease', detection_warning=detection_warning)

    return render_template('heart.html')

@app.route('/parkinsons')
def parkinsons_prediction():
    if request.method == 'POST':
        fo = request.form['fo']
        fhi = request.form['fhi']
        flo = request.form['flo']
        Jitter_percent = request.form['Jitter_percent']
        Jitter_Abs = request.form['Jitter_Abs']
        RAP = request.form['RAP']
        PPQ = request.form['PPQ']
        DDP = request.form['DDP']
        Shimmer = request.form['Shimmer']
        Shimmer_dB = request.form['Shimmer_dB']
        APQ3 = request.form['APQ3']
        APQ5 = request.form['APQ5']
        APQ = request.form['APQ']
        DDA = request.form['DDA']
        NHR = request.form['NHR']
        HNR = request.form['HNR']
        RPDE = request.form['RPDE']
        DFA = request.form['DFA']
        spread1 = request.form['spread1']
        spread2 = request.form['spread2']
        D2 = request.form['D2']
        PPE = request.form['PPE']

        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                                                           Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE,
                                                           DFA, spread1, spread2, D2, PPE]])

        if parkinsons_prediction[0] == 1:
            result = 'Parkinsons Disease Detected'
            detection_warning = ''
        else:
            result = 'Parkinsons Disease not detected'
            detection_warning = ''

        return render_template('result.html', disease_detected=parkinsons_prediction[0], disease_name='Parkinsons Disease', detection_warning=detection_warning)

    return render_template('parkinsons.html')

if __name__ == '__main__':
    app.run(debug=True)
