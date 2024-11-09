import streamlit as st
import pandas as pd
import pickle
import time

# Dummy input_df
input_data = {
    'temparature': 30,
    'humidity': 80,
    'moisture': 15,
    'nitrogen': 10,
    'potassium': 20,
    'phosphorous': 5,
    'soil_type': 'Sandy',  # Can be 'Sandy', 'Loamy', 'Black', 'Red', 'Clayey'
    'crop_type': 'Maize',  # Can be one of the available crop types
}

# Creating the Streamlit app
def app():
    st.header("Welcome to Fertilizer Prediction App 🌾")
    st.subheader("Predict the fertilizer recommendation based on environmental conditions and crop type")

    # Define available options
    soil_types = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
    crop_types = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley',
                  'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']

    # Input form for prediction
    with st.form("prediction_form", clear_on_submit=True):
        st.subheader("Enter the details for prediction")

        # Numerical input fields for features
        temparature = st.number_input("Temperature (°C)", min_value=-50, max_value=50, value=input_data['temparature'])
        humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=input_data['humidity'])
        moisture = st.number_input("Moisture (%)", min_value=0, max_value=100, value=input_data['moisture'])
        # Categorical input fields for soil and crop types
        soil_type = st.selectbox("Soil Type", soil_types, index=soil_types.index(input_data['soil_type']))
        crop_type = st.selectbox("Crop Type", crop_types, index=crop_types.index(input_data['crop_type']))
        nitrogen = st.number_input("Nitrogen (ppm)", min_value=0, max_value=100, value=input_data['nitrogen'])
        potassium = st.number_input("Potassium (ppm)", min_value=0, max_value=100, value=input_data['potassium'])
        phosphorous = st.number_input("Phosphorous (ppm)", min_value=0, max_value=100, value=input_data['phosphorous'])

        submit_button = st.form_submit_button("Predict Fertilizer")

        # Process the input data for prediction
        if submit_button:
            # Check if inputs are correct
            st.success("Inputs received successfully ✅")
            st.write("Prediction in progress...")

            # Prepare the input DataFrame
            input_df = pd.DataFrame({
                'temparature': [temparature],
                'humidity': [humidity],
                'moisture': [moisture],
                'nitrogen': [nitrogen],
                'potassium': [potassium],
                'phosphorous': [phosphorous],
                'soil_type': [soil_type],
                'crop_type': [crop_type]
            })

            # Load the model (you can replace this with your actual model)
            with open('C:\\code\\ml\\ds-projects\\machinelearning\\machinelearningmodels\\final_rf_model.pkl', 'rb') as f:
                model = pickle.load(f)

            # Make the prediction (assuming the model has the `predict` method)
            predicted_fertilizer = model.predict(input_df)

            # Display the predicted fertilizer
            st.write(f"The recommended fertilizer for your input is: {predicted_fertilizer[0]}")
            
            # Display additional insights or results (if any)
            st.success("Prediction complete! ✅")

if __name__ == "__main__":
    app()
