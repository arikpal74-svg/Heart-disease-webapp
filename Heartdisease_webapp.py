import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load the trained model
model = pickle.load(open('mytrained_model.sav', 'rb'))

st.title("❤️ Heart Disease Prediction Web App")
st.write("Use the sidebar to enter patient details and check the likelihood of heart disease.")

# Sidebar inputs
st.sidebar.header("Patient Input Features")

age = st.sidebar.slider("Age", 1, 120, 30)
sex = st.sidebar.radio("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.sidebar.slider("Serum Cholestoral (mg/dl)", 100, 600, 200)
fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "False" if x == 0 else "True")
restecg = st.sidebar.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 250, 150)
exang = st.sidebar.radio("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 10.0, 1.0, step=0.1)
slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
thal = st.sidebar.selectbox("Thal (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", [1, 2, 3])

# Initialize session state for logging
if "log" not in st.session_state:
    st.session_state.log = pd.DataFrame(columns=[
        "Age","Sex","Chest Pain Type","Resting BP","Cholesterol","Fasting Blood Sugar",
        "Resting ECG","Max Heart Rate","Exercise Angina","Oldpeak","Slope","Major Vessels","Thal",
        "Prediction","Confidence (%)"
    ])

# Prediction button
if st.sidebar.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        result_text = f"⚠️ The model predicts **Heart Disease** with probability {probability[0][1]*100:.2f}%"
        st.error(result_text)
    else:
        result_text = f"✅ The model predicts **No Heart Disease** with probability {probability[0][0]*100:.2f}%"
        st.success(result_text)

    # Create new entry
    new_entry = {
        "Age": age, "Sex": sex, "Chest Pain Type": cp, "Resting BP": trestbps,
        "Cholesterol": chol, "Fasting Blood Sugar": fbs, "Resting ECG": restecg,
        "Max Heart Rate": thalach, "Exercise Angina": exang, "Oldpeak": oldpeak,
        "Slope": slope, "Major Vessels": ca, "Thal": thal,
        "Prediction": "Heart Disease" if prediction[0] == 1 else "No Heart Disease",
        "Confidence (%)": round(probability[0][prediction[0]] * 100, 2)
    }

    # Add to session log
    st.session_state.log = pd.concat([st.session_state.log, pd.DataFrame([new_entry])], ignore_index=True)

    # Convert single prediction to DataFrame
    result_df = pd.DataFrame([new_entry])

    # Download buttons
    st.download_button(
        label="📥 Download This Prediction as CSV",
        data=result_df.to_csv(index=False).encode('utf-8'),
        file_name="single_prediction.csv",
        mime="text/csv"
    )

    st.download_button(
        label="📥 Download Full Session Log as CSV",
        data=st.session_state.log.to_csv(index=False).encode('utf-8'),
        file_name="session_predictions.csv",
        mime="text/csv"
    )

# Display session log
st.subheader("📊 Session Log of Predictions")
st.dataframe(st.session_state.log)



