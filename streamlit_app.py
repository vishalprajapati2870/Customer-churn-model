import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = load_data()

def preprocess_data(data):
    data = data.copy()
    data.dropna(inplace=True)
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data.dropna(subset=['TotalCharges'], inplace=True)
    data.drop(['customerID'], axis=1, inplace=True)

    for col in data.select_dtypes(include='object').columns:
        if col != 'Churn':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    return data

processed_df = preprocess_data(df)

X = processed_df.drop('Churn', axis=1)
y = processed_df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(model_name):
    if model_name == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000)
    elif model_name == 'Decision Tree':
        model = DecisionTreeClassifier()
    elif model_name == 'Random Forest':
        model = RandomForestClassifier()
    else:
        raise ValueError("Invalid model")

    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy


st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Predict", "History", "Data"])

if section == "Data":
    st.title("📂 Dataset Preview")
    st.dataframe(df.head(20))

elif section == "History":
    st.title("📜 Prediction History")
    if "history" in st.session_state and st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df)
    else:
        st.info("No predictions made yet.")

elif section == "Predict":
    st.title("🤖 Predict Customer Churn")

    st.markdown("### Choose Model")
    selected_model = st.radio("Select ML Model", ["Logistic Regression", "Decision Tree", "Random Forest"], horizontal=True)


    st.markdown("### Enter Customer Details")

    def input_fields():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            gender = st.radio("Gender", ["Male", "Female"])
        with col2:
            SeniorCitizen = st.radio("Senior Citizen", [0, 1])
        with col3:
            Partner = st.radio("Partner", ["Yes", "No"])
        with col4:
            Dependents = st.radio("Dependents", ["Yes", "No"])

        col5, col6, col7, col8 = st.columns(4)
        with col5:
            PhoneService = st.radio("Phone Service", ["Yes", "No"])
        with col6:
            MultipleLines = st.radio("Multiple Lines", ["Yes", "No", "No phone service"])
        with col7:
            InternetService = st.radio("Internet Service", ["DSL", "Fiber optic", "No"])
        with col8:
            OnlineSecurity = st.radio("Online Security", ["Yes", "No", "No internet service"])

        col9, col10, col11, col12 = st.columns(4)
        with col9:
            OnlineBackup = st.radio("Online Backup", ["Yes", "No", "No internet service"])
        with col10:
            DeviceProtection = st.radio("Device Protection", ["Yes", "No", "No internet service"])
        with col11:
            TechSupport = st.radio("Tech Support", ["Yes", "No", "No internet service"])
        with col12:
            StreamingTV = st.radio("Streaming TV", ["Yes", "No", "No internet service"])

        col13, col14, col15, col16 = st.columns(4)
        with col13:
            StreamingMovies = st.radio("Streaming Movies", ["Yes", "No", "No internet service"])
        with col14:
            Contract = st.radio("Contract", ["Month-to-month", "One year", "Two year"])
        with col15:
            PaperlessBilling = st.radio("Paperless Billing", ["Yes", "No"])
        with col16:
            PaymentMethod = st.radio("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])

        st.markdown("### Additional Details")
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        MonthlyCharges = st.slider("Monthly Charges", 1, 150, 70)
        TotalCharges = st.slider("Total Charges", 1, 10000, 1000)

        input_dict = {
            'gender': gender,
            'SeniorCitizen': SeniorCitizen,
            'Partner': Partner,
            'Dependents': Dependents,
            'PhoneService': PhoneService,
            'MultipleLines': MultipleLines,
            'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup,
            'DeviceProtection': DeviceProtection,
            'TechSupport': TechSupport,
            'StreamingTV': StreamingTV,
            'StreamingMovies': StreamingMovies,
            'Contract': Contract,
            'PaperlessBilling': PaperlessBilling,
            'PaymentMethod': PaymentMethod,
            'tenure': tenure,
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges
        }

        return input_dict

    user_input = input_fields()

    if st.button("Submit"):
        input_df = pd.DataFrame([user_input])


        base_df = df.drop(columns=['Churn'], errors='ignore')
        base_df = base_df.drop(columns=['customerID'], errors='ignore')

        full_data = pd.concat([base_df, input_df], ignore_index=True)


        for col in full_data.columns:
            if full_data[col].dtype == 'object' or isinstance(full_data[col][0], str):
                full_data[col] = full_data[col].astype(str)
                le = LabelEncoder()
                full_data[col] = le.fit_transform(full_data[col])

        input_processed = full_data.tail(1)

        model, acc = train_model(selected_model)
        prediction = model.predict(input_processed)[0]
        prob = model.predict_proba(input_processed)[0][1]

        label = "🔴 Customer is likely to churn" if prediction == 1 else "🟢 Customer is likely to stay"
        st.markdown(f"### {label}")
        st.success(f"🔍 Probability of Churn: **{prob * 100:.2f}%**")
        st.info(f"Model Accuracy: **{acc * 100:.2f}%**")

        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({**user_input, "Model": selected_model, "Churn Probability": round(prob, 2)})
