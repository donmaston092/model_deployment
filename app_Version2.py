import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

# --- Page Config ---
st.set_page_config(layout="wide")
st.title("🌸 Iris Explainable AI App")

# --- Paths ---
MODEL_PATH = os.path.join("xai_iris_model.joblib")
SCALER_PATH = os.path.join("iris_scaler.joblib")  # Remove if you don't use a scaler

# --- Load Model and Scaler ---
@st.cache_data
def load_resources():
    model = joblib.load(MODEL_PATH)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    else:
        scaler = None
    return model, scaler

model, scaler = load_resources()

# --- Feature and Class Names ---
feature_names = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
class_names = ["setosa", "versicolor", "virginica"]

# --- Load Iris Data for EDA ---
@st.cache_data
def load_iris_df():
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = [*feature_names, "species"]
    df["species"] = df["species"].map(dict(enumerate(class_names)))
    return df

iris_df = load_iris_df()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🔮 Prediction", "🧑‍🔬 LIME Explanation"])

# --- TAB 1: EDA ---
with tab1:
    st.header("Exploratory Data Analysis")
    st.subheader("Dataset Preview")
    st.dataframe(iris_df.head())

    st.subheader("Species Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="species", data=iris_df, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots()
    sns.heatmap(iris_df[feature_names].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax2)
    st.pyplot(fig2)

    st.subheader("Pairplot")
    fig3 = sns.pairplot(iris_df, hue="species")
    st.pyplot(fig3)

# --- Sidebar for User Input ---
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
sepal_width  = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
petal_width  = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2, 0.1)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# --- TAB 2: Prediction ---
with tab2:
    st.header("Predict Iris Species")
    if st.button("Predict"):
        if scaler is not None:
            input_scaled = scaler.transform(input_data)
        else:
            input_scaled = input_data
        pred = model.predict(input_scaled)[0]
        pred_proba = model.predict_proba(input_scaled)[0]
        st.success(f"**Predicted Species:** {class_names[pred].capitalize()}")
        st.write("### Prediction Probabilities")
        st.bar_chart(pd.Series(pred_proba, index=class_names))

# --- TAB 3: LIME Explanation ---
with tab3:
    st.header("LIME Explanation for the Model's Prediction")

    @st.cache_data
    def load_background():
        # For LIME, use a sample of the training data (scaled as needed)
        X = iris_df[feature_names].values
        if scaler is not None:
            X = scaler.transform(X)
        return X

    background_data = load_background()

    lime_explainer = LimeTabularExplainer(
        training_data=background_data,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification"
    )

    if st.button("Explain with LIME"):
        if scaler is not None:
            input_for_lime = scaler.transform(input_data)
        else:
            input_for_lime = input_data
        exp = lime_explainer.explain_instance(
            input_for_lime[0],
            model.predict_proba,
            num_features=4,
            top_labels=1
        )
        st.write("#### LIME Explanation (HTML)")
        st.components.v1.html(exp.as_html(), height=800, scrolling=True)
# Footer with author info
st.markdown("---")
st.markdown("""
#### 👤 About the Author

Developed by **Dr. Anthony Onoja**
📧 Email: [donmaston09@gmail.com, YouTube:@tonyonoja7880](YouTube:@tonyonoja7880)
""")
