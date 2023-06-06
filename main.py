import joblib
import streamlit as st

model = joblib.load('SVM_model.joblib')
st.title('Accessing risk level of CAD')


age = st.number_input('Input age(years old)')
sex = st.number_input('Input sex')
st.caption("0 for Female")
st.caption("1 for Male")
chestPainType = st.number_input('Input chest pain type')
st.caption("1 for Typical Angina")
st.caption("2 for Atypical Angina")
st.caption("3 for Non-anginal Pain")
st.caption("4 for Asymptomatic")
restingBP = st.number_input('Input restingBP(mmHg)')
cholesterol = st.number_input('Input cholesterol level(mg/dl)')
fastingBloodSugar = st.number_input('Input fasting blood sugar')
st.caption("0 for fasting blood sugar < 120 mg/dl")
st.caption("1 for fasting blood sugar > 120 mg/dl")
restECG = st.number_input('Input resting ECG')
st.caption("0 for Normal")
st.caption("1 for having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)")
st.caption("2 for showing probable or definite left ventricular hypertrophy by Estes' criteria")
maxHeartRate = st.number_input('Input maximum heart rate')
exerciseInducedAngina = st.number_input('Exercise Induced Angina?')
st.caption("0 for NO exercise induced angina")
st.caption("1 for YES exercise induced angina")
previousPeak = st.number_input('Input previous peak achieved')
slope = st.number_input('Input slope of ST segment on the ECG')
majorVesselsNumber = st.number_input('Input number of major vessels')
thaliumStressTest = st.number_input('Input Thalium stress test result')
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


