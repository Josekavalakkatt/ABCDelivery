import streamlit as st
import pandas as pd
import joblib

# Load the trained Logistic Regression model
logi = joblib.load('logi.sav')

st.title('Delivery Delay Prediction App')
st.write('Enter the details below to predict if a delivery will be delayed.')

# Input features (matching the X.columns from your notebook)
delivery_distance = st.slider('Delivery Distance', 0.0, 100.0, 20.0)
traffic_congestion = st.slider('Traffic Congestion (1-5)', 1, 5, 3)
weather_condition = st.slider('Weather Condition (1-5)', 1, 5, 2)
delivery_slot = st.slider('Delivery Slot (1-3)', 1, 3, 2)
driver_experience = st.slider('Driver Experience (years)', 0, 20, 5)
num_stops = st.slider('Number of Stops', 1, 10, 2)
vehicle_age = st.slider('Vehicle Age (years)', 0, 10, 3)
road_condition_score = st.slider('Road Condition Score (1-5)', 1, 5, 3)
package_weight = st.slider('Package Weight (kg)', 0.0, 50.0, 10.0)
fuel_efficiency = st.slider('Fuel Efficiency (km/L)', 5.0, 25.0, 15.0)
warehouse_processing_time = st.slider('Warehouse Processing Time (minutes)', 0, 100, 30)

# Create a DataFrame from the inputs
input_data = pd.DataFrame([{
    'Delivery_Distance': delivery_distance,
    'Traffic_Congestion': traffic_congestion,
    'Weather_Condition': weather_condition,
    'Delivery_Slot': delivery_slot,
    'Driver_Experience': driver_experience,
    'Num_Stops': num_stops,
    'Vehicle_Age': vehicle_age,
    'Road_Condition_Score': road_condition_score,
    'Package_Weight': package_weight,
    'Fuel_Efficiency': fuel_efficiency,
    'Warehouse_Processing_Time': warehouse_processing_time
}])

if st.button('Predict Delivery Delay'):
    prediction = logi.predict(input_data)
    prediction_proba = logi.predict_proba(input_data)[0]

    if prediction[0] == 1:
        st.error(f"Prediction: Delivery WILL be delayed (Probability: {prediction_proba[1]:.2f})")
    else:
        st.success(f"Prediction: Delivery WILL NOT be delayed (Probability: {prediction_proba[0]:.2f})")

    st.subheader('Input Features Used for Prediction:')
    st.write(input_data)
