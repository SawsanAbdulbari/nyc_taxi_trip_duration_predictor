import os
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from joblib import load
from feature_engineering import prepare_data
st.set_page_config(page_title="NYC Taxi Predictor", layout="wide")
# Load the ridge regression model
model_path = 'D:/ml_projects/project-nyc-taxi-trip-duration/models/ridge_regression_model.pkl'
st.markdown("""
<style>
.main .block-container {
    max-width: 65%;
    
    padding-top: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
</style>
""", unsafe_allow_html=True)
model_loaded = False
if os.path.exists(model_path):
    ridge_model = load(model_path)
    model_loaded = True
    # st.sidebar.write(f"Loaded model type: {type(ridge_model)}")
else:
    st.sidebar.error("Model file not found. Please ensure the model file is in the correct path.")

# Sidebar information
image = "img.png"
st.sidebar.image(image, caption='NYC Taxi(AI)', use_column_width=True)
st.sidebar.write("Machine Learning Diploma Project 📊📈")
st.sidebar.markdown(
    "Made with :orange_heart: by [Sawsan Abdulbari](https://www.linkedin.com/in/sawsanabdulbari/)"
)
# Page header
st.title(':taxi: :orange[NYC Taxi Trip Duration Predictor]')
st.write("This application predicts the duration of a taxi trip in NYC based on input features.")

# Function to predict trip duration
def predict_duration(input_data):
    try:
        processed_data = prepare_data(input_data, is_training=False)
        prediction = ridge_model.predict(processed_data)
        return np.expm1(prediction[0]), None  # Assuming log transformation was used
    except Exception as e:
        return None, str(e)

# Function to validate inputs
def validate_inputs(input_data):
    errors = []
    
    if not (-74.2591 <= input_data['pickup_longitude'][0] <= -73.7004):
        errors.append('Pickup Longitude must be between -74.2591 and -73.7004.')
    if not (40.4774 <= input_data['pickup_latitude'][0] <= 40.9176):
        errors.append('Pickup Latitude must be between 40.4774 and 40.9176.')
    if not (-74.2591 <= input_data['dropoff_longitude'][0] <= -73.7004):
        errors.append('Dropoff Longitude must be between -74.2591 and -73.7004.')
    if not (40.4774 <= input_data['dropoff_latitude'][0] <= 40.9176):
        errors.append('Dropoff Latitude must be between 40.4774 and 40.9176.')
    
    return errors

# Default values for inputs
default_values = {
    'pickup_latitude': 40.7128,
    'pickup_longitude': -74.0060,
    'dropoff_latitude': 40.7128,
    'dropoff_longitude': -74.0060,
    'pickup_datetime': pd.to_datetime("now"),
    'vendor_id': 1,
    'passenger_count': 1,
    'store_and_fwd_flag': 'N'
}

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Function to reset inputs to default values
def reset_inputs():
    for key, value in default_values.items():
        st.session_state[key] = value

# Collect user inputs
def collect_inputs():
    st.session_state['input_data']['pickup_latitude'] = pickup_latitude
    st.session_state['input_data']['pickup_longitude'] = pickup_longitude
    st.session_state['input_data']['dropoff_latitude'] = dropoff_latitude
    st.session_state['input_data']['dropoff_longitude'] = dropoff_longitude
    st.session_state['input_data']['pickup_datetime'] = pickup_datetime_combined
    st.session_state['input_data']['vendor_id'] = vendor_id
    st.session_state['input_data']['passenger_count'] = passenger_count
    st.session_state['input_data']['store_and_fwd_flag'] = store_and_fwd_flag

# Placeholder for input data
if 'input_data' not in st.session_state:
    st.session_state['input_data'] = {}

# Map setup
def create_map(lat, lon):
    folium_map = folium.Map(location=[lat, lon], zoom_start=12)
    folium.Marker(location=[lat, lon], draggable=True).add_to(folium_map)
    return folium_map

pickup_map = create_map(st.session_state['pickup_latitude'], st.session_state['pickup_longitude'])
dropoff_map = create_map(st.session_state['dropoff_latitude'], st.session_state['dropoff_longitude'])

# Form section
with st.form("prediction_form"):
    st.write("Select Pickup Location")
    pickup_map_data = st_folium(pickup_map, width=700, height=450, key="pickup_map")

    st.write("Select Dropoff Location")
    dropoff_map_data = st_folium(dropoff_map, width=700, height=450, key="dropoff_map")

    col1, col2 = st.columns(2)
    with col1:
        pickup_latitude = st.slider(
            'Pickup Latitude', 
            min_value=40.4774, max_value=40.9176, 
            value=pickup_map_data['last_clicked']['lat'] if pickup_map_data and pickup_map_data.get('last_clicked') else st.session_state.pickup_latitude, 
            format="%.6f",
            help="Latitude coordinate of the pickup location. Must be between 40.4774 and 40.9176."

        )
        pickup_longitude = st.slider(
            'Pickup Longitude', 
            min_value=-74.2591, max_value=-73.7004, 
            value=pickup_map_data['last_clicked']['lng'] if pickup_map_data and pickup_map_data.get('last_clicked') else st.session_state.pickup_longitude, 
            format="%.6f",
            help="Longitude coordinate of the pickup location. Must be between -74.2591 and -73.7004."

        )
    with col2:
        dropoff_latitude = st.slider(
            'Dropoff Latitude', 
            min_value=40.4774, max_value=40.9176, 
            value=dropoff_map_data['last_clicked']['lat'] if dropoff_map_data and dropoff_map_data.get('last_clicked') else st.session_state.dropoff_latitude, 
            format="%.6f",
            help="Latitude coordinate of the dropoff location. Must be between 40.4774 and 40.9176."

        )
        dropoff_longitude = st.slider(
            'Dropoff Longitude', 
            min_value=-74.2591, max_value=-73.7004, 
            value=dropoff_map_data['last_clicked']['lng'] if dropoff_map_data and dropoff_map_data.get('last_clicked') else st.session_state.dropoff_longitude, 
            format="%.6f",
            help="Longitude coordinate of the dropoff location. Must be between -74.2591 and -73.7004."

        )

    pickup_date = st.date_input("Pickup Date", value=pd.to_datetime(st.session_state.pickup_datetime).date(),
                                        help="Date when the trip starts.")
    pickup_time = st.time_input("Pickup Time", value=pd.to_datetime(st.session_state.pickup_datetime).time(), help="Time when the trip starts.")
    vendor_id = st.selectbox('Vendor ID', [1, 2], index=st.session_state.vendor_id - 1, help="ID of the vendor providing the trip. 1 for vendor 1, 2 for vendor 2."
)
    passenger_count = st.number_input('Passenger Count', min_value=1, max_value=10, value=st.session_state.passenger_count, help="Number of passengers in the trip. Must be between 1 and 10."
)
    store_and_fwd_flag = st.selectbox('Store and Forward Flag', ['Y', 'N'], index=['Y', 'N'].index(st.session_state.store_and_fwd_flag),        help="Indicates if the trip data was stored and forwarded. 'Y' for yes, 'N' for no.")

    # Combine date and time into a single datetime column
    pickup_datetime_combined = pd.to_datetime(f"{pickup_date} {pickup_time}")

    # Button to submit the form
    submit_button = st.form_submit_button("Predict :rocket:")

# Handle form submission
if submit_button:
    # Collecting user inputs with the collect_inputs function
    collect_inputs()

    input_df = pd.DataFrame.from_dict({key: [value] for key, value in st.session_state['input_data'].items()})

    errors = validate_inputs(input_df)  # Checking for input validation errors
    if errors:
        for error in errors:
            st.error(error)
    else:
        # If no errors are found, perform the prediction
        with st.spinner('Predicting...'):
            predicted_duration, error_message = predict_duration(input_df)
        if error_message:
            st.error(f"Prediction error: {error_message}")
        else:
            # Format the duration to hours, minutes, and seconds
            hours, remainder = divmod(predicted_duration, 3600)
            minutes, seconds = divmod(remainder, 60)
            formatted_duration = f"{int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds"
            
            st.success(f'The predicted trip duration is {formatted_duration}.')

# Reset button to reset inputs to default values
if st.button("Reset"):
    reset_inputs()
    st.experimental_rerun()  # st.experimental_rerun() used to refresh the page with default values

# Providing instructions and guidance for the user
st.markdown("### Instructions")
st.write("Use the maps to select the pickup and dropoff locations, and use the sliders to fine-tune the latitude and longitude. Fill in the other fields based on the characteristics of the taxi trip in NYC. Click 'Predict' to see the model's prediction.")

# Add a footer

footer = """
<style>
.footer a:link, .footer a:visited{
    color: red;
    background-color: transparent;
    text-decoration: underline;
}

.footer a:hover, .footer a:active {
    color: blue;
    background-color: transparent;
    text-decoration: underline;
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
}
</style>
<div class="footer">
<p>Developed with <span style='color:red;'>❤</span> by <a href="https://www.linkedin.com/in/sawsanabdulbari/" target="_blank">Sawsan Abdulbari</a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
