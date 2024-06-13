import streamlit as st
import pandas as pd
import pickle
import requests
import os

# Title of the Streamlit app
st.title('Grab Food Delivery Time & Cost Prediction App')
st.write('Model used: Random Forest Regressor')

# Function to download and load the trained model
@st.cache_data
def download_and_load_model(url, model_path):
    if not os.path.exists(model_path):
        response = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Function to load the test dataset
@st.cache_data
def load_test_dataset(file_path):
    return pd.read_csv(file_path)

# Load the test dataset
test_dataset_path = 'test_dataset.csv'  # Update with your test dataset path
test_dataset = load_test_dataset(test_dataset_path)

# Create a download button for the test dataset
test_dataset_csv = test_dataset.to_csv(index=False)
st.download_button(
    label="Download Test Dataset",
    data=test_dataset_csv,
    file_name='test_dataset.csv',
    mime='text/csv',
)

# OneDrive shareable links for the models
# time_model_url = 'https://365umedumy-my.sharepoint.com/:u:/g/personal/22115126_siswa365_um_edu_my/EZza65faJadIlNaqdDICbaYBBldgw_FikLhqn09Mwp1CqA?e=2POuqZ'  # Replace with your actual OneDrive shareable link
# cost_model_url = 'https://365umedumy-my.sharepoint.com/:u:/g/personal/22115126_siswa365_um_edu_my/EVKy-M-FrPZLsc7Quctc-soBEkgS_UH32O7Qr7rN_RdIqg?e=A3UMMT'  # Replace with your actual OneDrive shareable link

time_model_url = 'https://365umedumy-my.sharepoint.com/:u:/g/personal/22115126_siswa365_um_edu_my/EZza65faJadIlNaqdDICbaYBBldgw_FikLhqn09Mwp1CqA?download=1'
cost_model_url = 'https://365umedumy-my.sharepoint.com/:u:/g/personal/22115126_siswa365_um_edu_my/EVKy-M-FrPZLsc7Quctc-soBEkgS_UH32O7Qr7rN_RdIqg?download=1'

# Paths to save the downloaded models
time_model_path = 'rf_best_time.pkl'
cost_model_path = 'rf_best_cost.pkl'

# Download and load the trained models
time_model = download_and_load_model(time_model_url, time_model_path)
cost_model = download_and_load_model(cost_model_url, cost_model_path)

# Load unique cuisines
with open('unique_cuisines.txt', 'r') as f:
    unique_cuisines = f.read().splitlines()

# Load other columns for get_dummies
with open('other_columns.txt', 'r') as f:
    other_columns = f.read().splitlines()

# Load final columns used in the model
with open('final_columns.txt', 'r') as f:
    final_columns = f.read().splitlines()

def preprocess_data(df):
    # Split the cuisine column into lists of cuisines
    df['cuisine'] = df['cuisine'].apply(lambda x: x.strip("[]").replace("'", "").split(", "))

    # Create an empty DataFrame to store one-hot encoded values for cuisines
    one_hot_encoded_df = pd.DataFrame(0, index=df.index, columns=unique_cuisines)
    one_hot_encoded_df = one_hot_encoded_df.reindex(sorted(one_hot_encoded_df.columns), axis=1)

    # Update the one-hot encoded DataFrame based on the original DataFrame
    for index, row in df.iterrows():
        for cuisine in row['cuisine']:
            if cuisine in one_hot_encoded_df.columns:
                one_hot_encoded_df.at[index, cuisine] = 1

    # Concatenate original data with one-hot encoded values
    df_encoded = pd.concat([df, one_hot_encoded_df], axis=1)
    df_encoded.drop(columns=['cuisine'], inplace=True)

    # Apply get_dummies to specified columns
    df_encoded = pd.get_dummies(df_encoded, columns=other_columns, drop_first=False, dtype=int)

    # Replace 'Yes'/'No' with 1/0 in 'promo' column
    df_encoded['promo'] = df_encoded['promo'].replace({'Yes': 1, 'No': 0})

    # Ensure all final columns are present in the DataFrame
    concatenated_columns = [df_encoded]  # Start with the existing DataFrame
    for col in final_columns:
        if col not in df_encoded.columns:
            # Add a new DataFrame with the missing column
            new_column = pd.DataFrame({col: [0] * len(df_encoded)}, index=df_encoded.index)
            concatenated_columns.append(new_column)

    # Concatenate all DataFrames along axis 1
    df_encoded = pd.concat(concatenated_columns, axis=1)

    # Ensure the DataFrame has the same column order as the final columns
    df_encoded = df_encoded[final_columns]

    return df_encoded

# Function to make predictions
def predict_delivery_time(model, test_df):
    # Remove the 'delivery_time' feature from the test data
    test_df.drop(columns=['delivery_time'], inplace=True)

    # Predict delivery_time
    predictions = model.predict(test_df)
    return predictions

# Function to make predictions
def predict_delivery_cost(model, test_df):
    # Remove the 'delivery_time' feature from the test data
    test_df.drop(columns=['delivery_cost'], inplace=True)

    # Predict delivery_time
    predictions = model.predict(test_df)
    return predictions

# Function to download predictions
def download_time_predictions(predictions, test_df):
    output_df = test_df.copy()
     # Move the "delivery_time" and "delivery_cost" columns to the last two positions
    cols = output_df.columns.tolist()
    cols.remove('delivery_time')
    cols.remove('delivery_cost')
    cols.append('delivery_time')
    cols.append('delivery_cost')
    output_df = output_df[cols]
    output_df['predicted_delivery_time'] = predictions.round(2)
    output_csv = output_df.to_csv(index=False)
    output_csv = output_df.to_csv(index=False)
    return output_df, output_csv

def download_cost_predictions(predictions, test_df):
    output_df = test_df.copy()
    # Move the "delivery_time" and "delivery_cost" columns to the last two positions
    cols = output_df.columns.tolist()
    cols.remove('delivery_time')
    cols.remove('delivery_cost')
    cols.append('delivery_time')
    cols.append('delivery_cost')
    output_df = output_df[cols]
    output_df['predicted_delivery_cost'] = predictions.round(2)
    output_csv = output_df.to_csv(index=False)
    output_csv = output_df.to_csv(index=False)
    return output_df, output_csv

# File uploader to let user upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    st.write("Uploaded Data:")
    st.dataframe(test_df)

    df_preprocessed = preprocess_data(test_df)

    # Button to trigger prediction
    if st.button("Predict Delivery Time"):
        # Predict delivery_time
        predictions = predict_delivery_time(time_model, df_preprocessed)

        # Download predictions and display the output DataFrame
        output_df, output_csv = download_time_predictions(predictions, test_df)
        st.write("Time Predictions Data: (Result in last column)")
        st.dataframe(output_df)

        st.download_button(
            label="Download Predictions as CSV",
            data=output_csv.encode(),
            file_name='predicted_delivery_times.csv',
            mime='text/csv',
        )

     # Button to trigger prediction
    if st.button("Predict Delivery Cost"):
        # Predict delivery_cost
        predictions = predict_delivery_cost(cost_model, df_preprocessed)
        
        # Download predictions and display the output DataFrame
        output_df, output_csv = download_cost_predictions(predictions, test_df)
        st.write("Cost Predictions Data: (Result in last column)")
        st.dataframe(output_df)
        
        st.download_button(
            label="Download Predictions as CSV",
            data=output_csv.encode(),
            file_name='predicted_delivery_costs.csv',
            mime='text/csv',
        )
