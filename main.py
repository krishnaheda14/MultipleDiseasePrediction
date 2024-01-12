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
cancer_model = pickle.load(open('C:/Users/Krishna/PycharmProjects/EDAIProject/models/cancer_model.sav', 'rb'))

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
            detection_warning = 'This prediction maybe inaccurate , Please consult a doctor immediately'
        else:
            result = 'Heart Disease not detected'
            detection_warning = ''

        return render_template('result.html', disease_detected=heart_prediction[0], disease_name='Heart Disease', detection_warning=detection_warning)

    return render_template('heart.html')

@app.route('/cancer', methods=['GET', 'POST'])
def cancer_prediction():
    if request.method == 'POST':
        # Extract input values from the form
        radius_mean = float(request.form['radius_mean'])
        texture_mean = float(request.form['texture_mean'])
        perimeter_mean = float(request.form['perimeter_mean'])
        area_mean = float(request.form['area_mean'])
        smoothness_mean = float(request.form['smoothness_mean'])
        compactness_mean = float(request.form['compactness_mean'])
        concavity_mean = float(request.form['concavity_mean'])
        concave_points_mean = float(request.form['concave_points_mean'])
        symmetry_mean = float(request.form['symmetry_mean'])
        fractal_dimension_mean = float(request.form['fractal_dimension_mean'])
        radius_se = float(request.form['radius_se'])
        texture_se = float(request.form['texture_se'])
        perimeter_se = float(request.form['perimeter_se'])
        area_se = float(request.form['area_se'])
        smoothness_se = float(request.form['smoothness_se'])
        compactness_se = float(request.form['compactness_se'])
        concavity_se = float(request.form['concavity_se'])
        concave_points_se = float(request.form['concave_points_se'])
        symmetry_se = float(request.form['symmetry_se'])
        fractal_dimension_se = float(request.form['fractal_dimension_se'])
        radius_worst = float(request.form['radius_worst'])
        texture_worst = float(request.form['texture_worst'])
        perimeter_worst = float(request.form['perimeter_worst'])
        area_worst = float(request.form['area_worst'])
        smoothness_worst = float(request.form['smoothness_worst'])
        compactness_worst = float(request.form['compactness_worst'])
        concavity_worst = float(request.form['concavity_worst'])
        concave_points_worst = float(request.form['concave_points_worst'])
        symmetry_worst = float(request.form['symmetry_worst'])
        fractal_dimension_worst = float(request.form['fractal_dimension_worst'])

        # Create an input array
        input_data = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                                compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
                                fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se,
                                smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se,
                                fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst,
                                smoothness_worst, compactness_worst, concavity_worst, concave_points_worst,
                                symmetry_worst, fractal_dimension_worst]])

        # Make predictions using the cancer model
        cancer_prediction = cancer_model.predict(input_data)

        # Determine the result message
        if cancer_prediction[0] == 1:
            result = 'Cancer Detected'
            detection_warning = ''
        else:
            result = 'Cancer not detected'
            detection_warning = 'Please consult a doctor immediately'

        # Render the result template with the prediction information
        return render_template('result.html', disease_detected=cancer_prediction[0], disease_name='Cancer', detection_warning=detection_warning)

    # Render the cancer prediction form template for GET requests
    return render_template('cancer.html')

if __name__ == '__main__':
    app.run(debug=True)
