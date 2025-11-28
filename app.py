import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Data Science App", layout="wide")

st.title("🟠 Beginner Data Science Project")
st.write("Upload dataset → Explore → Visualize → Train Model → Predict")

# -----------------------------
# UPLOAD DATASET
# -----------------------------
uploaded = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.session_state["data"] = df

    st.subheader("📌 Dataset Preview")
    st.write(df.head())

    st.subheader("📌 Shape of Dataset")
    st.write(df.shape)

    st.subheader("📌 Summary Statistics")
    st.write(df.describe())
else:
    st.warning("Please upload a CSV file to start!")

# -----------------------------
# VISUALIZATION
# -----------------------------
st.header("📈 Visualizations")

if "data" in st.session_state:
    df = st.session_state["data"]

    col = st.selectbox("Select column for Histogram", df.columns)

    fig, ax = plt.subplots()
    sns.histplot(df[col], ax=ax)
    st.pyplot(fig)

    st.subheader("📊 Correlation Heatmap")
    fig2, ax2 = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)
else:
    st.info("Upload dataset to see visualizations.")

# -----------------------------
# ML MODEL TRAINING
# -----------------------------
st.header("🤖 Train Machine Learning Model")

if "data" in st.session_state:
    df = st.session_state["data"]

    target = st.selectbox("Select target column", df.columns)

    if st.button("Train Model"):
        try:
            X = df.drop(columns=[target])
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            model = LinearRegression()
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)

            st.success("Model Trained Successfully!")
            st.write(f"📉 Mean Squared Error: `{mse}`")

        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Upload dataset and select target column to train model.")
