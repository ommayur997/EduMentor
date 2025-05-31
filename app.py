import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the dataset
df = pd.read_csv("datasets/edu_mentor_dataset_final(5000).csv")

# Load trained models
regression_model = joblib.load('regression_model.pkl')
classification_model = joblib.load('classification_model.pkl')

# Define categorical mappings
mapping_learning_style = {
    'Auditory': 0,
    'Kinesthetic': 1,
    'Visual': 2
}

mapping_content = {
    'Audio': 0,
    'Interactive': 1,
    'Text': 2,
    'Video': 3
}

mapping_comments = {
    'Falling behind in multiple areas': 0,
    'Lacks focus': 1,
    'Needs improvement in basics': 2,
    'Good participation': 3,
    'Average progress': 4,
    'Can do better with more effort': 5,
    'Consistent performance': 6,
    'Excellent progress': 7,
    'Shows initiative': 8
}

# CORRECT FEATURE ORDER AS EXPECTED BY THE MODEL
MODEL_FEATURE_ORDER = [
    'std',
    'math_grade',
    'english_grade',
    'science_grade',
    'history_grade',
    'overall_grade',
    'assignment_completion',
    'engagement_score',
    'math_lec_present',
    'science_lec_present',
    'history_lec_present',
    'english_lec_present',
    'attendance_ratio',
    'login_frequency_per_week',
    'average_session_duration_minutes',
    'learning_style',
    'content_type_preference',
    'completed_lessons',
    'practice_tests_taken',
    'lms_test_scores',
    'teacher_comments_summary'
]

# Title
st.title("Student Academic Performance Dashboard")

# Input: Student ID and Name
student_id = st.text_input("Enter Student ID")
student_name = st.text_input("Enter Student Name")

# Validate student
if student_id and student_name:
    student_record = df[(df['student_id'].astype(str) == str(student_id)) &
                        (df['student_name'].str.lower() == student_name.lower())]

    if not student_record.empty:
        student = student_record.squeeze()
        st.success("Student found. Please choose a performance category.")

        # Prepare data for model prediction
        data = student_record.copy()

        # Apply categorical mappings
        data['learning_style'] = data['learning_style'].map(mapping_learning_style)
        data['content_type_preference'] = data['content_type_preference'].map(mapping_content)
        data['teacher_comments_summary'] = data['teacher_comments_summary'].map(mapping_comments)

        # Ensure all required columns exist
        for col in MODEL_FEATURE_ORDER:
            if col not in data.columns:
                st.error(f"Missing column in dataset: {col}")
                st.stop()

        # Select and reorder columns to match model's expected order
        data = data[MODEL_FEATURE_ORDER]

        # Make predictions
        try:
            risk_score_pred = regression_model.predict(data)[0]
            is_at_risk_pred = classification_model.predict(data)[0]
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.stop()

        # Display model predictions
        st.subheader("Risk Prediction Results")
        col1, col2 = st.columns(2)

        # Show predicted risk score with color indicator
        risk_color = "red" if risk_score_pred > 70 else "orange" if risk_score_pred > 40 else "green"
        col1.metric("Predicted Risk Score",
                    f"{risk_score_pred:.2f}",
                    help="Risk score interpretation: 0-40 (Low), 41-70 (Medium), 71-100 (High)")

        # Show predicted risk status with appropriate icon
        if is_at_risk_pred == 1:
            col2.metric("Predicted Risk Status", "At Risk 🚨",
                        help="Student is predicted to be at risk. Immediate attention recommended.",
                        delta_color="inverse")
            st.warning(
                "This student is predicted to be at risk. Please review their performance details below and consider intervention strategies.")
        else:
            col2.metric("Predicted Risk Status", "Not At Risk ✅",
                        help="Student is predicted to be performing adequately.",
                        delta_color="off")
            st.success("This student is predicted to be performing adequately. Continue monitoring progress.")

        # Dropdown to select performance category
        option = st.selectbox("Select performance category:",
                              ("Academic Performance (Grades)",
                               "Attendance & Participation",
                               "Learning Behavior & Engagement",
                               "Learning Preferences",
                               "Qualitative Feedback",
                               "All"))

        if option in ["Academic Performance (Grades)", "All"]:
            st.subheader("Academic Performance (Grades)")
            cols = ['math_grade', 'english_grade', 'science_grade', 'history_grade', 'overall_grade']
            st.write(student[cols])

            for subject in cols[:-1]:
                if student[subject] < 40:
                    st.warning(
                        f"{subject.replace('_', ' ').title()}: Score below 40. Please consult respective teacher.")
            if student['overall_grade'] < 60:
                st.warning("Overall grade is below 60. Consult all subject teachers and work harder.")
            elif 60 <= student['overall_grade'] <= 85:
                st.info("Performance is average. Keep pushing to improve further.")
            elif student['overall_grade'] > 85:
                st.success("Congratulations! Great academic performance.")

        if option in ["Attendance & Participation", "All"]:
            st.subheader("Attendance & Participation")
            cols = ['math_lec_present', 'science_lec_present', 'history_lec_present', 'english_lec_present',
                    'attendance_ratio', 'login_frequency_per_week']
            st.write(student[cols])

            for lec in cols[:-2]:
                if student[lec] < 10:
                    st.warning(f"{lec.replace('_', ' ').title()}: Less than 10 classes attended. Minimum 15 required.")
            if student['attendance_ratio'] < 0.5:
                st.warning("Attendance ratio below 0.5. Please consult with the class teacher.")
            if student['login_frequency_per_week'] < 4:
                st.warning("Login frequency less than 4 per week. Please consult with the class teacher.")

        if option in ["Learning Behavior & Engagement", "All"]:
            st.subheader("Learning Behavior & Engagement")
            cols = ['average_session_duration_minutes', 'engagement_score', 'completed_lessons',
                    'practice_tests_taken', 'lms_test_scores', 'assignment_completion']
            st.write(student[cols])

            if student['average_session_duration_minutes'] < 60:
                st.warning("Average session duration is low (< 60 minutes). Increase your study time.")
            if student['engagement_score'] < 0.8:
                st.warning("Engagement score is below 0.8. Stay focused and consistent.")
            if student['completed_lessons'] < 10:
                st.warning("Fewer than 10 lessons completed. Please complete more lessons.")
            if student['practice_tests_taken'] < 5:
                st.warning("Practice tests taken < 5. Practice more.")
            if student['lms_test_scores'] < 40:
                st.warning("LMS test scores are low (< 40). Focus on understanding topics.")
            if student['assignment_completion'] < 60:
                st.warning("Assignment completion rate is low (< 60%). Submit assignments on time.")

        if option in ["Learning Preferences", "All"]:
            st.subheader("Learning Preferences")
            cols = ['learning_style', 'content_type_preference']
            st.write(student[cols])

        if option in ["Qualitative Feedback", "All"]:
            st.subheader("Qualitative Feedback")
            st.write({
                "Teacher Comments": student['teacher_comments_summary']
            })

    else:
        st.error("Student not found. Please check your input or contact the staff.")
