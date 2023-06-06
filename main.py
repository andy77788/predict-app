import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split
# import seaborn as sns
# sns.set_theme(style="whitegrid")
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import RobustScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
import streamlit as st
# from utils import columns


# # define column names
# cols = ["age", "sex", "chestPainType", "restingBP", "cholesterol", "fastingBloodSugar",
#         "restECG", "maxHeartRate","exerciseInducedAngina", "previousPeak", "slope",
#         "majorVesselsNumber", "thaliumStressTest", "target"]
#
# # read csv file
# df = pd.read_csv("heart.csv", names=cols)
# df.drop(index=df.index[0], axis=0, inplace=True)
#
# # assign data types
# numerics = ['age', 'restingBP', 'cholesterol', 'maxHeartRate', 'previousPeak']
# for col in df.columns:
#   if col in numerics:
#     df[col] = df[col].astype(float)
#   else:
#     df[col] = df[col].astype(int)
#
# # point-biserial correlation of all continuous variables vs target
# from scipy.stats import pointbiserialr
# x = np.array(df["target"])
# pd.DataFrame([pointbiserialr(x, df[y]) for y in numerics], index=numerics)
#
# # calculate composition of slope where patients were diagnosed with CAD
# df[df.target==1].slope.value_counts(normalize=True)
#
# df.isnull().sum()
# df.duplicated().sum()
# df.drop_duplicates(inplace=True)
# df.duplicated().sum()
#
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
#
# target = df["target"]
# df = df.drop("target", axis=1)
#
# split the dataset into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.3, random_state=42)
#
# cats = ['chestPainType', 'restECG', 'slope', 'majorVesselsNumber', 'thaliumStressTest']
# numerics = ['age', 'restingBP', 'cholesterol', 'maxHeartRate', 'previousPeak']

# Preprocess Data
# preprocessor = ColumnTransformer([
#     ('one-hot-encoder', OneHotEncoder(), cats),
#     ('standard_scaler', RobustScaler(), numerics)])

# SVM Classifier

# C = 0.1
# pipe = make_pipeline(preprocessor, SVC(C=C))
# pipe.fit(x_train, y_train)
# y_pred = pipe.predict(x_test)
# svc_score = accuracy_score(y_test, y_pred)
# print("SVM Classifier Accuracy: " f"{svc_score:.4f}")
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# joblib.dump(pipe, 'SVM_model.joblib') # first run to save model to physical files

# pipe = joblib.load('SVM_model.joblib')
# predictions = pipe.predict(x_test)
# print(predictions)

model = joblib.load('SVM_model.joblib')
st.title('Accessing risk level of CAD')

# st.text_input('Age')

# cols = ["age", "sex", "chestPainType", "restingBP", "cholesterol", "fastingBloodSugar",
#         "restECG", "maxHeartRate","exerciseInducedAngina", "previousPeak", "slope",
#         "majorVesselsNumber", "thaliumStressTest", "target"]

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

