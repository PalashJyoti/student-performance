import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

model = joblib.load(Path("../models/student_exam_score_model.pkl"))

df = pd.read_csv(Path("../data/processed/dataset.csv"))
family_income_options = list(df["Family_Income"].unique())
parental_education_options = list(df["Parental_Education_Level"].unique())
parental_involvement_options = list(df["Parental_Involvement"].unique())
teacher_quality_options = list(df["Teacher_Quality"].unique())
school_type_options = list(df["School_Type"].unique())
access_to_resources_options = list(df["Access_to_Resources"].unique())
distance_from_home_options = list(df["Distance_from_Home"].unique())
internet_access_options = list(df["Internet_Access"].unique())
extracurricular_activities_options = list(df["Extracurricular_Activities"].unique())
motivation_level_options = list(df["Motivation_Level"].unique())
peer_influence_options_options = list(df["Peer_Influence"].unique())
learning_disabilities_options = list(df["Learning_Disabilities"].unique())
gender_options = list(df["Gender"].unique())

st.title("Student Performance App")

st.subheader("Academic")
hours_studied = st.number_input("Hours Studied: ")
attendance = st.number_input("Attendance: ")
previous_score = st.number_input("Previous Score: ")
tutoring_sessions = st.number_input("Tutoring Sessions: ")

st.subheader("Family")
family_income = st.selectbox("Family Income: ", family_income_options)
parental_education = st.selectbox("Parental Education", parental_education_options)
parental_involvement = st.selectbox(
    "Parental Involvement", parental_involvement_options
)

st.subheader("School")
teacher_quality = st.selectbox("Teacher Quality: ", teacher_quality_options)
school_type = st.selectbox("School Type: ", school_type_options)
access_to_resources = st.selectbox("Access to Resources: ", access_to_resources_options)
distance_from_home = st.selectbox("Distance from Home: ", distance_from_home_options)

st.subheader("Lifestyle")
sleep_hours = st.number_input("Sleep Hours: ")
physical_activity = st.number_input("Physical Activity: ")
internet_access = st.selectbox("Internet Access: ", internet_access_options)
extracurricular_activities = st.selectbox(
    "Extracurricular Activities: ", extracurricular_activities_options
)
motivation_level = st.selectbox("Motivation Level: ", motivation_level_options)
peer_influence = st.selectbox("Peer Influence: ", peer_influence_options_options)
learning_disabilities = st.selectbox(
    "Learning Disabilities: ", learning_disabilities_options
)
gender = st.selectbox("Gender: ", gender_options)


if st.button("Predict"):
    input_df = {
        "Hours_Studied": [hours_studied],
        "Attendance": [attendance],
        "Parental_Involvement": [parental_involvement],
        "Access_to_Resources": [access_to_resources],
        "Extracurricular_Activities": [extracurricular_activities],
        "Sleep_Hours": [sleep_hours],
        "Previous_Scores": [previous_score],
        "Motivation_Level": [motivation_level],
        "Internet_Access": [internet_access],
        "Tutoring_Sessions": [tutoring_sessions],
        "Family_Income": [family_income],
        "Teacher_Quality": [teacher_quality],
        "School_Type": [school_type],
        "Peer_Influence": [peer_influence],
        "Physical_Activity": [physical_activity],
        "Learning_Disabilities": [learning_disabilities],
        "Parental_Education_Level": [parental_education],
        "Distance_from_Home": [distance_from_home],
        "Gender": [gender],
    }
    input_df = pd.DataFrame(input_df)
    pred_score = model.predict(input_df)[0]
    st.success(f"Predicted Exam Score: {pred_score:.2f}")
