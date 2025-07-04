# Customer Churn Prediction App 📉🧠

---

## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-ml.streamlit.app/)

This is a Machine Learning-powered web app built with **Streamlit** that predicts whether a customer will **churn** based on various input features.

It leverages:
* 🧪 Trained ML models such as **Logistic Regression**, **Decision Tree**, and **Random Forest**.
* 📊 Data preprocessing and encoding techniques.
* 🧾 An intuitive input form for real-time predictions.
* 🗂 Session-based history tracking for past predictions.

---

## 📌 Features

* **Manual Input**: Enter customer information manually to get predictions.
* **Churn Prediction & Probability**: Get a churn prediction along with the model's confidence score.
* **Model Selection**: Choose between different ML models for predictions.
* **Prediction History**: View a history of your predictions within the current session.
* **Sample Dataset**: Utilizes the `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset for training and demonstration.

---

## 💻 Technologies Used

* **Python** 🐍
* **Streamlit** 🚀
* **Pandas & NumPy** 📊
* **Scikit-learn** 🤖

---

## ⚙️ How to Run Locally

Follow these steps to set up and run the application on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/vishalprajapati2870/Customer-churn-model.git](https://github.com/vishalprajapati2870/Customer-churn-model.git)
    cd Customer-churn-model
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
    * **On Windows:**
        ```bash
        venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run Customer_chrun.py
    ```

---

Made with ❤️ by Vishal Prajapati
