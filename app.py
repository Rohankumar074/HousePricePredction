import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained pipeline and model
pipeline = joblib.load('house_price_pipeline.pkl')
stackreg = joblib.load('house_price_model.pkl')

st.title("House Price Prediction App")

st.write("Enter the features of the house to predict its price.")

# You would add input fields for each feature in your dataset here
# For simplicity, let's just show a few examples

# Example input fields (replace with your actual features)
# Example: LotArea, OverallQual, YearBuilt, etc.
lot_area = st.number_input("Lot Area", min_value=0)
overall_qual = st.slider("Overall Quality", 1, 10, 5)
year_built = st.number_input("Year Built", min_value=1800, max_value=2024, value=2000)

# Create a dictionary with the input data
# You'll need to create a dictionary with all the features your model expects
# Ensure the column names match those used during training
input_data = {
    'LotArea': lot_area,
    'OverallQual': overall_qual,
    'YearBuilt': year_built,
    # Add all other features here with appropriate input fields
}

# Convert the input data to a DataFrame
# Make sure the columns are in the same order as your training data
input_df = pd.DataFrame([input_data])

if st.button("Predict Price"):
    # Preprocess the input data using the loaded pipeline
    # Need to add all columns that the pipeline expects
    # For now, only the few columns are added
    # Create a dummy DataFrame with all columns to fit the preprocessor
    dummy_data = pd.DataFrame(columns=pipeline.feature_names_in_)
    input_df = pd.concat([dummy_data, input_df], ignore_index=True)

    # Fill any missing columns with default values (e.g., 0 or mean/most_frequent from training data)
    # This is a simplified approach. In a real app, you'd handle missing values more robustly.
    for col in pipeline.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0 # Or use the imputation strategy from your pipeline

    # Reorder columns to match the training data
    input_df = input_df[pipeline.feature_names_in_]

    preprocessed_input = pipeline.transform(input_df)

    # Predict the price using the loaded model
    predicted_price_log = stackreg.predict(preprocessed_input)

    # Inverse transform the prediction (if you applied log transformation to the target variable)
    predicted_price = np.expm1(predicted_price_log)

    st.success(f"Predicted House Price: ${predicted_price[0]:,.2f}")
