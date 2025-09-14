import streamlit as st
import pandas as pd, numpy as np, os
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

st.title("🎓 Student Performance Prediction")
st.write("Predict whether a student will Pass or Fail based on study habits & attendance.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Internet'], df['Passed'] = LabelEncoder().fit_transform(df['Internet']), LabelEncoder().fit_transform(df['Passed'])
    features = ['StudyHours', 'Attendance', 'PastScore', 'SleepHours']
    scaler, df_scaled = StandardScaler(), df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    X, y = df_scaled[features], df_scaled['Passed']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if st.checkbox("Show Dataset"):
        st.dataframe(df.head())

    if st.checkbox("Show Classification Report"):
        st.text(classification_report(y_test, y_pred))

    if st.checkbox("Show Confusion Matrix"):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"], ax=ax)
        st.pyplot(fig)

    st.subheader("🔮 Try Your Own Input")
    study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, value=5.0)
    attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=85.0)
    past_score = st.number_input("Past Score (%)", min_value=0.0, max_value=100.0, value=70.0)
    sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)

    if st.button("Predict"):
        user_input = pd.DataFrame([{'StudyHours': study_hours,'Attendance': attendance,'PastScore': past_score,'SleepHours': sleep_hours}])
        user_scaled = scaler.transform(user_input)
        prediction = model.predict(user_scaled)[0]
        result = "✅ Pass" if prediction == 1 else "❌ Fail"
        st.success(f"Prediction: {result}")
else:
    st.warning("Please upload a CSV file to continue.")

