import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
model = joblib.load('LightGBMUAS.pkl')
X_test = pd.read_csv('X_testUAS.csv')
feature_names = ['Gender', 'Age', 'UAS', 'Double_J_indwelled ', 'Lower_pole_stone','RIPA','The_number_of_stones','RIL', 'Diameter_of_the_stone', 'HU','BMI']
st.title("Predict the stone-free rate after RIRS surgery")
Gender = st.selectbox("Gender:", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
Age = st.selectbox("Age:", options=[1, 2,3,4], format_func=lambda x: "0-35" if x == 1 else ("35-47" if x == 2 else ("47-59" if x == 3 else ">59")))
UAS = st.selectbox("UAS:", options=[1, 2], format_func=lambda x: "12/10 UAS" if x == 1 else "14/12 UAS")
Double_J_indwelled = st.selectbox("Double_J_indwelled:", options=[1, 2], format_func=lambda x: "Yes" if x == 2 else "No")
Lower_pole_stone = st.selectbox("Lower_pole_stone:", options=[1, 2], format_func=lambda x: "No" if x == 1 else "Yes")
RIPA = st.selectbox("RIPA:", options=[1, 2], format_func=lambda x: "RIPA>45°" if x == 1 else "RIPA≤45°")
The_number_of_stones = st.selectbox("The_number_of_stones:", options=[1, 2], format_func=lambda x: "Single" if x == 1 else "Multiple")
RIL = st.selectbox("RIL:", options=[1, 2], format_func=lambda x: "RIL≤25mm" if x == 1 else "RIL>25mm")
Diameter_of_the_stone = st.selectbox("Diameter_of_the_stone:", options=[1, 2], format_func=lambda x: "≤20mm" if x == 1 else ">20mm")
HU = st.selectbox("HU:", options=[1, 2,3,4], format_func=lambda x: "≤840" if x == 1 else ("840-980" if x == 2 else ("980-1120" if x == 3 else ">1120")))
BMI = st.selectbox("BMI:", options=[1, 2,3], format_func=lambda x: "≤18.5" if x == 1 else ("18.5-24" if x == 2 else ">24"))
feature_values = [Gender, Age, UAS, Double_J_indwelled, Lower_pole_stone, RIPA, The_number_of_stones, RIL, Diameter_of_the_stone, HU, BMI]
features = np.array([feature_values])
if st.button("Predict"):
    # Predict class and probabilities
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class} (1: Residual stones after RIRS, 0: No residual stones after RIRS)")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    # Generate advice based on prediction results
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"According to our model, Residual stones after RIRS. "
            f"The probability of residual stones after RIRS is  {probability:.1f}%. "
             )
    else:
        advice = (
            f"According to our model, No residual stones after RIRSe. "
            f"The probability of no residual stones after RIRS is {probability:.1f}%. "
            )
    st.write(advice)
    # SHAP Explanation
    st.subheader("SHAP Force Plot Explanation")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    # Display the SHAP force plot for the predicted class
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)        
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
    # LIME Explanation
    st.subheader("LIME Explanation")
    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=X_test.columns.tolist(),
        class_names=['Ineffective', 'Effective'],
        mode='classification'
    )
    # Explain the instance
    lime_exp = lime_explainer.explain_instance(
        data_row=features.flatten(),
        predict_fn=model.predict_proba
    )
    # Display the LIME explanation without the feature value table
    lime_html = lime_exp.as_html(show_table=False)
    st.components.v1.html(lime_html, height=800, scrolling=True)