import numpy as np
import pandas as pd
import joblib
import streamlit as st

model = joblib.load('SVM_model.joblib')
st.title('CAD Risk Level Assessment')


age = st.number_input('Input age (20-80 years old)', min_value=20, max_value=80, format="%d")
sex = st.number_input('Input sex (0-1)', min_value=0, max_value=1, format="%d")
st.caption("0 for Female")
st.caption("1 for Male")
chestPainType = st.number_input('Input chest pain type (0-3)', min_value=0, max_value=3, format="%d")
st.caption("0 for Typical Angina")
st.caption("1 for Atypical Angina")
st.caption("2 for Non-anginal Pain")
st.caption("3 for Asymptomatic")
restingBP = st.number_input('Input restingBP (80-250 mmHg)', min_value=80, max_value=250, format="%d")
cholesterol = st.number_input('Input cholesterol level (100-600 mg/dl)', min_value=100, max_value=600, format="%d")
fastingBloodSugar = st.number_input('Input fasting blood sugar (0-1)', min_value=0, max_value=1, format="%d")
st.caption("0 for fasting blood sugar < 120 mg/dl")
st.caption("1 for fasting blood sugar > 120 mg/dl")
restECG = st.number_input('Input resting ECG (0-2)', min_value=0, max_value=2, format="%d")
st.caption("0 for Normal")
st.caption("1 for having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)")
st.caption("2 for showing probable or definite left ventricular hypertrophy by Estes' criteria")
maxHeartRate = st.number_input('Input maximum heart rate (40-210 bpm)', min_value=40, max_value=210, format="%d")
exerciseInducedAngina = st.number_input('Exercise Induced Angina? (0-1)', min_value=0, max_value=1, format="%d")
st.caption("0 for NO exercise induced angina")
st.caption("1 for YES exercise induced angina")
previousPeak = st.number_input('Input previous peak achieved (0-6.5)', min_value=0.0, max_value=6.5, format="%3.1f")
slope = st.number_input('Input slope of ST segment on the ECG (0-2)', min_value=0, max_value=2, format="%d")
st.caption("0 for downwards slope")
st.caption("1 for flat")
st.caption("2 for upwards slope")
majorVesselsNumber = st.number_input('Input number of major vessels (0-4)', min_value=0, max_value=4, format="%d")
thaliumStressTest = st.number_input('Input Thalium stress test result (0-3)', min_value=0, max_value=3, format="%d")
st.caption("0 for test indicating normal blood flow")
st.caption("1 for test indicating abnormal blood flow during exercise")
st.caption("2 for test indicating low blood flow during both rest and exercise")
st.caption("3 for test indicating no thallium visible in parts of the heart")

cols = ['age', 'sex', 'chestPainType', 'restingBP', 'cholesterol', 'fastingBloodSugar', 'restECG', 'maxHeartRate', 'exerciseInducedAngina', 'previousPeak', 'slope', 'majorVesselsNumber', 'thaliumStressTest']

def predict():
    row = np.array([age, sex, chestPainType, restingBP, cholesterol, fastingBloodSugar, restECG, maxHeartRate,exerciseInducedAngina, previousPeak, slope, majorVesselsNumber, thaliumStressTest])
    x = pd.DataFrame([row], columns=cols)
    prediction = model.predict(x)[0]

    if prediction == 0:
        st.success('Low risk')
    else:
        st.error('High risk')


st.button('Predict', on_click=predict)


