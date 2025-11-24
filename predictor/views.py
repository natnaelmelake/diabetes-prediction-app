from django.shortcuts import render

# Create your views here.


import numpy as np
import pandas as pd
from django.shortcuts import render
from tensorflow.keras.models import load_model

# Load the model once when the server starts
model = load_model('predictor/models/diabetes_model.keras')

def index(request):
    prediction = None
    if request.method == 'POST':
        # Get form data
        age = float(request.POST.get('age'))
        gender = int(request.POST.get('gender'))  # 0 or 1
        pulse_rate = float(request.POST.get('pulse_rate'))
        systolic_bp = float(request.POST.get('systolic_bp'))
        diastolic_bp = float(request.POST.get('diastolic_bp'))
        glucose = float(request.POST.get('glucose'))
        bmi = float(request.POST.get('bmi'))
        family_diabetes = int(request.POST.get('family_diabetes'))
        hypertensive = int(request.POST.get('hypertensive'))
        family_hypertension = int(request.POST.get('family_hypertension'))
        cardiovascular_disease = int(request.POST.get('cardiovascular_disease'))
        stroke = int(request.POST.get('stroke'))

        # Create input array
        input_data = np.array([[age, gender, pulse_rate, systolic_bp, diastolic_bp,
                                glucose, bmi, family_diabetes, hypertensive,
                                family_hypertension, cardiovascular_disease, stroke]])

        # Predict
        pred_prob = model.predict(input_data)[0][0]
        prediction = 'Diabetic' if pred_prob >= 0.5 else 'Not Diabetic'
        
        print("MODEL OUTPUT:", pred_prob)


    return render(request, 'predictor/index.html', {'prediction': prediction})
