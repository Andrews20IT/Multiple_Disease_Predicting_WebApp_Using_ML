import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

loaded_model1 = pickle.load(open('trained_diabetes_model.sav', 'rb'))
loaded_model2 = pickle.load(open('heart_disease_predict_model.sav', 'rb'))

with st.sidebar:
    selected = option_menu("Multiple Diseases Prediction",
                           ["Diabetes Prediction", "Heart Disease Prediction"],
                           icons=["activity", "heart"],
                           default_index=0)


def main():
    if selected == "Diabetes Prediction":
        st.title("Diabetes Predicting Web Application")

        Pregnancies = st.text_input("Number of Pregnancies (Enter a number between 0 and 20)")
        Glucose = st.text_input("Glucose Level (mg/dL, typical range: 0-300)")
        Blood_pressure = st.text_input("Blood Pressure Value (mm Hg, typical range: 0-200)")
        skin_thickness = st.text_input("Skin Thickness Value (mm, typical range: 0-100, optional)")
        Insulin = st.text_input("Insulin Level (μU/mL, typical range: 0-900, optional)")
        BMI = st.text_input("Body Mass Index (BMI, kg/m², typical range: 0-70)")
        Diabetes_pedigree_function = st.text_input("Diabetes Pedigree Function (Value between 0.0 and 2.5)")
        Age = st.text_input("Your Age (Enter your age between 1 and 120 years)")

        diagnosis1 = ''

        if st.button("Diabetes test result"):
            try:
                diagnosis1 = diabetes_prediction([Pregnancies, Glucose, Blood_pressure, skin_thickness, Insulin, BMI,
                                              Diabetes_pedigree_function, Age])
            except ValueError:
                st.error("Please enter valid numerical values for all fields.")

        st.success(diagnosis1)

    if selected == "Heart Disease Prediction":
        st.title("Heart Disease Predicting Web Application")

        age = st.text_input("Enter Age (Years, typically between 1 and 120)")
        gender = st.text_input("Enter Gender (0 for Female, 1 for Male)")
        chest_pain_type = st.text_input("Chest Pain Type (Enter a number: "
                                        "0 - Typical angina, "
                                        "1 - Atypical angina, "
                                        "2 - Non-anginal pain, "
                                        "3 - Asymptomatic)")
        resting_Blood_Pressure = st.text_input("Resting Blood Pressure (mm Hg, typical range: 80-200)")
        serum_Cholesterol = st.text_input("Serum Cholesterol (mg/dL, typical range: 100-600)")
        fasting_Blood_Sugar = st.text_input("Fasting Blood Sugar (Enter 1 if > 120 mg/dL, otherwise 0)")
        restecg = st.text_input("Resting Electrocardiographic Results (Enter a number: "
                                "0 - Normal, "
                                "1 - Having ST-T wave abnormality, "
                                "2 - Showing probable or definite left ventricular hypertrophy)")
        thalach = st.text_input("Maximum Heart Rate Achieved (bpm, typical range: 60-220)")
        exang = st.text_input("Exercise Induced Angina (Enter 1 for Yes, 0 for No)")
        Oldpeak = st.text_input("Oldpeak (ST depression induced by exercise relative to rest, typically 0-6)")
        Slope = st.text_input("Slope of the Peak Exercise ST Segment (Enter a number: "
                              "0 - Upsloping, "
                              "1 - Flat, "
                              "2 - Downsloping)")
        ca = st.text_input("Number of Major Vessels Colored by Fluoroscopy (Enter a number between 0-3)")
        thal = st.text_input("Thalassemia (Enter a number: "
                             "1 - Normal, "
                             "2 - Fixed defect, "
                             "3 - Reversible defect)")

        diagnosis2 = ''
        if st.button("heart disease test result"):
            try:
                diagnosis2 = heart_disease_prediction(
                    [age, gender, chest_pain_type, resting_Blood_Pressure, serum_Cholesterol, fasting_Blood_Sugar, restecg,
                     thalach, exang, Oldpeak, Slope, ca, thal])
            except ValueError:
                st.error("Please enter valid numerical values for all fields.")

        st.success(diagnosis2)


def diabetes_prediction(input_data):
    input_as_np_array = np.asarray(input_data)
    reshaped_np_array = input_as_np_array.reshape(1, -1)
    std_data = scaler.transform(reshaped_np_array)
    prediction = loaded_model1.predict(std_data)

    if prediction[0] == 1:
        return "Patient is Diabetic"
    else:
        return "Patient is Not Diabetic"


def heart_disease_prediction(input_data):
    input_data = [float(i) for i in input_data]

    input_as_nparray = np.asarray(input_data)

    input_reshaped = input_as_nparray.reshape(1, -1)

    res = loaded_model2.predict(input_reshaped)

    if res == 1:
        return "The user has heart disease"
    else:
        return "The user is healthy"


if __name__ == '__main__':
    main()
