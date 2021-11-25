import streamlit as st
import model
import pandas as pd


def user_input():
    """User input variables"""
    st.sidebar.title("Please select the value accordingly.")

    base_excess = st.sidebar.slider('Base Excess', -35.0, 45.0, -35.0, 0.1)
    anion_gap = st.sidebar.slider('Anion Gap', -5.0, 50.0, -5.0, 0.1)
    chloride = st.sidebar.slider('Chloride Level', 60.0, 160.0, 60.0, 0.1)
    creatinine = st.sidebar.slider('Creatinine Level', 0.0, 25.0, 0.0, 0.1)
    potassium = st.sidebar.slider('Potassium Level', 1.0, 20.0, 1.0, 0.1)
    sodium = st.sidebar.slider('Sodium Level', 100.0, 175.0, 100.0, 0.1)
    urea_nitrogen = st.sidebar.slider('Urea nitrogen', 0.0, 220.0, 0.0, 0.1)
    rdw = st.sidebar.slider('Red Cell Distribution Width', 0.0, 35.0, 0.0, 0.1)
    white_cell = st.sidebar.slider('White blood cells', 0.0, 450.0, 0.0, 0.1)
    icu_los = st.sidebar.slider('Length of Stay', 0.0, 265.0, 0.0, 0.1)

    user_info = {
        "Base Excess": base_excess,
        "Anion Gap": anion_gap,
        "Chloride Level": chloride,
        "Creatinine Level": creatinine,
        "Potassium Level": potassium,
        "Sodium Level": sodium,
        "Urea Nitrogen": urea_nitrogen,
        "Red Cell Distribution Width": rdw,
        "White Blood Cells": white_cell,
        "ICU Los": icu_los
    }
    return pd.DataFrame(user_info, index=[0])


st.subheader("Variable Used for Death Risk Prediction:")
st.text("""
    - Base Excess: the amount of excess or insufficient level of bicarbonate in the system.
    - Anion Gap: a measurement of the difference-or gap-between the negatively charged and positively charged electrolytes.
    - Chloride: measures the amount of chloride in your blood.
    - Creatinine: a chemical compound left over from energy-producing processes in your muscles.
    - Potassium: measures the amount of potassium in your blood.
    - Sodium: measures the amount of sodium in your blood.
    - Urea Nitrogen: reveals important information about how well your kidneys are working.
    - Red Cell Distribution Width (RDW): a measurement of the range in the volume and size of your red blood cells.
    - White Blood Cells: measures the number of white blood cells in your body.
    - icu_los: ICU length of stay.
    """)
user_input = user_input()
st.subheader("User Input: Blood Test Variables")
st.write(user_input)
st.subheader("Prediction Outcome: ")
prediction = model.death_risk_predict(user_input.iloc[0, 0], user_input.iloc[0, 1], user_input.iloc[0, 2],
                                             user_input.iloc[0, 3], user_input.iloc[0, 4], user_input.iloc[0, 5],
                                             user_input.iloc[0, 6], user_input.iloc[0, 7], user_input.iloc[0, 8],
                                             user_input.iloc[0, 9])
st.write("The risk of death: ", prediction)
st.subheader("Choose the model to view the ROC Curve:")
figure_selection = ['Logistic', 'Linear SVM', 'Poly SVM', 'RBF SVM']
st.set_option('deprecation.showPyplotGlobalUse', False)
input = st.selectbox("Select", figure_selection)
figure = model.ROC_plot(input)
st.pyplot(figure)
