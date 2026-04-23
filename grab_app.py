import json
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import os

from feature_pipeline import (
    load_feature_artifacts,
    read_final_columns,
    transform_for_inference,
    validate_input_columns,
)

st.title('Grab Food Delivery Time & Cost Prediction App')

try:
    with open('model_metrics.json') as _f:
        _m = json.load(_f)
    _time_name = _m.get('delivery_time', {}).get('selected_model', 'lightgbm').replace('_', ' ').title()
    _cost_name = _m.get('delivery_cost', {}).get('selected_model', 'lightgbm').replace('_', ' ').title()
    st.write(f'Models: **{_time_name}** (delivery time) · **{_cost_name}** (delivery cost)')
except Exception:
    st.write('Model: Phase 2 ensemble winner')


@st.cache_data
def download_and_load_model(url, model_path):
    if not os.path.exists(model_path):
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        with open(model_path, 'wb') as f:
            f.write(response.content)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


@st.cache_data
def load_test_dataset(file_path):
    return pd.read_csv(file_path)


test_dataset = load_test_dataset('test_dataset.csv')
st.download_button(
    label="Download Test Dataset",
    data=test_dataset.to_csv(index=False),
    file_name='test_dataset.csv',
    mime='text/csv',
)

# Use secrets if configured, fall back to OneDrive URLs
try:
    time_model_url = st.secrets["time_model_url"]
    cost_model_url = st.secrets["cost_model_url"]
except (KeyError, FileNotFoundError):
    time_model_url = 'https://huggingface.co/jengyang/Grab-Food-Delivery-Time-Prediction/resolve/main/xg_best_time.pkl'
    cost_model_url = 'https://huggingface.co/jengyang/Grab-Food-Delivery-Cost-Prediction/resolve/main/xg_best_cost.pkl'

time_model_path = 'xg_best_time.pkl'
cost_model_path = 'xg_best_cost.pkl'

with st.spinner("Loading models, please wait..."):
    try:
        time_model = download_and_load_model(time_model_url, time_model_path)
        cost_model = download_and_load_model(cost_model_url, cost_model_path)
    except Exception as e:
        st.error(
            f"Failed to load prediction models. Check your internet connection or contact the maintainer.\n\n"
            f"Error: {e}"
        )
        st.stop()

feature_artifacts = load_feature_artifacts('feature_artifacts.json')
final_columns = read_final_columns('final_columns.txt')
fill_values = feature_artifacts["fill_values"]


def preprocess_data(df):
    validate_input_columns(df, require_targets=False)
    return transform_for_inference(df, feature_artifacts, final_columns, fill_values)


def predict_delivery_time(model, test_df):
    return model.predict(test_df)


def predict_delivery_cost(model, test_df):
    return np.clip(model.predict(test_df), 0.0, None)


def build_output(test_df, predictions, pred_col_name):
    output_df = test_df.copy()
    front_cols = [c for c in output_df.columns if c not in ('delivery_time', 'delivery_cost')]
    back_cols = [c for c in ('delivery_time', 'delivery_cost') if c in output_df.columns]
    output_df = output_df[front_cols + back_cols]
    output_df[pred_col_name] = predictions.round(2)
    return output_df


uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    st.write("Uploaded Data:")
    st.dataframe(test_df)

    try:
        df_preprocessed = preprocess_data(test_df)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    if st.button("Predict Delivery Time"):
        with st.spinner("Running prediction..."):
            try:
                predictions = predict_delivery_time(time_model, df_preprocessed.copy())
            except Exception as e:
                st.error(
                    "Time model prediction failed. Make sure the deployed model files were retrained "
                    "together with the Phase 2 feature artifacts.\n\n"
                    f"Error: {e}"
                )
                st.stop()
        output_df = build_output(test_df, predictions, 'predicted_delivery_time')
        st.write("Time Predictions Data: (Result in last column)")
        st.dataframe(output_df)
        st.download_button(
            label="Download Predictions as CSV",
            data=output_df.to_csv(index=False).encode(),
            file_name='predicted_delivery_times.csv',
            mime='text/csv',
        )

    if st.button("Predict Delivery Cost"):
        with st.spinner("Running prediction..."):
            try:
                predictions = predict_delivery_cost(cost_model, df_preprocessed.copy())
            except Exception as e:
                st.error(
                    "Cost model prediction failed. Make sure the deployed model files were retrained "
                    "together with the Phase 2 feature artifacts.\n\n"
                    f"Error: {e}"
                )
                st.stop()
        output_df = build_output(test_df, predictions, 'predicted_delivery_cost')
        st.write("Cost Predictions Data: (Result in last column)")
        st.dataframe(output_df)
        st.download_button(
            label="Download Predictions as CSV",
            data=output_df.to_csv(index=False).encode(),
            file_name='predicted_delivery_costs.csv',
            mime='text/csv',
        )
