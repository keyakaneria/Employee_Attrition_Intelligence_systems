import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import streamlit as st
# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Employee Attrition Intelligence System",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Model and Data
# -----------------------------
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

df = pd.read_csv("Employee-Attrition.csv")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Employee Attrition System")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Prediction"]
)

# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    st.title("📊 Employee Attrition Dashboard")

    total = len(df)
    attrition = len(df[df["Attrition"]=="Yes"])
    rate = round(attrition/total*100,2)

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Employees", total)
    c2.metric("Attrition", attrition)
    c3.metric("Attrition Rate", f"{rate}%")
    c4.metric("Average Income", f"${int(df['MonthlyIncome'].mean())}")

    st.divider()

    fig1 = px.pie(
        df,
        names="Attrition",
        title="Employee Attrition Distribution"
    )

    st.plotly_chart(fig1,use_container_width=True)

    fig2 = px.histogram(
        df,
        x="Department",
        color="Attrition",
        title="Department Wise Attrition"
    )

    st.plotly_chart(fig2,use_container_width=True)

# =====================================================
# PREDICTION
# =====================================================

else:

    st.title("🎯 Employee Attrition Prediction")

    age = st.slider("Age",18,60,30)

    monthly_income = st.number_input(
        "Monthly Income",
        1000,
        25000,
        5000
    )

    years = st.slider(
        "Years at Company",
        0,
        40,
        5
    )

    overtime = st.selectbox(
        "OverTime",
        ["No","Yes"]
    )

    job_level = st.slider(
        "Job Level",
        1,
        5,
        2
    )

    # -----------------------------
    # Create Input Data
    # -----------------------------

    input_df = pd.DataFrame(
        np.zeros((1,len(feature_names))),
        columns=feature_names
    )

    # Numeric Features
    if "Age" in feature_names:
        input_df["Age"] = age

    if "MonthlyIncome" in feature_names:
        input_df["MonthlyIncome"] = monthly_income

    if "YearsAtCompany" in feature_names:
        input_df["YearsAtCompany"] = years

    if "JobLevel" in feature_names:
        input_df["JobLevel"] = job_level

    # One-Hot Encoded Overtime
    if "OverTime_Yes" in feature_names:
        input_df["OverTime_Yes"] = 1 if overtime=="Yes" else 0

    # Scale
    input_scaled = scaler.transform(input_df)

    # -----------------------------
    # Prediction
    # -----------------------------

    if st.button("Predict Attrition"):

        prediction = model.predict(input_scaled)[0]

        probability = model.predict_proba(input_scaled)[0]

        confidence = max(probability)*100

        st.subheader("Prediction")

        if prediction==1:

            st.error("⚠ High Risk of Attrition")

        else:

            st.success("✅ Low Risk of Attrition")

        st.write(f"### Confidence : {confidence:.2f}%")

        st.progress(float(confidence)/100)

        st.subheader("Retention Suggestions")

        if overtime=="Yes":
            st.warning("Reduce overtime workload.")

        if monthly_income<4000:
            st.warning("Employee salary is relatively low.")

        if years<2:
            st.info("New employees benefit from mentoring and onboarding support.")

        if prediction==0:
            st.success("Employee appears likely to stay with the company.")

st.markdown("---")
st.markdown(
    "<center><b>Employee Attrition Intelligence System</b><br>Built using Machine Learning & Streamlit</center>",
    unsafe_allow_html=True
)