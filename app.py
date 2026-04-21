import pickle
import pandas as pd
import numpy as np
import streamlit as st

# Cache model and scaler
@st.cache_resource
def load_artifacts(scaler_path, model_path):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return scaler, model


def predict_species(sep_len, sep_width, pet_len, pet_wid, scaler, model):
    try:
        # prepare input data
        x_new = pd.DataFrame({
            'SepalLengthCm': [sep_len],
            'SepalWidthCm': [sep_width],
            'PetalLengthCm': [pet_len],
            'PetalWidthCm': [pet_wid]
        })

        # Transform input data
        xnew_pre = scaler.transform(x_new)

        # make predictions
        pred = model.predict(xnew_pre)
        probs = model.predict_proba(xnew_pre)
        max_prob = np.max(probs)

        return pred, max_prob

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None


# Streamlit UI
st.title("Iris Species Predictor")

sep_len = st.number_input("SepalLengthCm", min_value=0.0, step=0.1, value=5.1)
sep_width = st.number_input("SepalWidthCm", min_value=0.0, step=0.1, value=3.5)
pet_len = st.number_input("PetalLengthCm", min_value=0.0, step=0.1, value=1.4)
pet_wid = st.number_input("PetalWidthCm", min_value=0.0, step=0.1, value=0.2)  # fixed default

if st.button("Predict"):
    scaler_path = 'notebook/scaler.pkl'
    model_path = 'notebook/model.pkl'

    scaler, model = load_artifacts(scaler_path, model_path)

    pred, max_prob = predict_species(
        sep_len, sep_width, pet_len, pet_wid, scaler, model
    )

    if pred is not None:
        st.subheader(f'Predicted Species: {pred[0]}')
        st.subheader(f'Prediction Probability: {max_prob:.4f}')
        st.progress(float(max_prob))
    else:
        st.error("Prediction failed. Check model files and inputs.")