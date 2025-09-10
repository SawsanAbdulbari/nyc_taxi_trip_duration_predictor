import os
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

# Main page content
st.title(":taxi: :orange[NYC Taxi Trip Duration Prediction App]")
st.write("""
## NYC Taxi Trip Duration Prediction App
This application uses a pre-trained Ridge Regression model to predict NYC taxi trip duration based on user-provided trip details via Streamlit.
This project used for educational purposes.
""")

# Sidebar information
image = "img.png"
st.sidebar.image(image, caption='NYC Taxi(AI)', use_container_width=True)
st.sidebar.write("Machine Learning Diploma Project 📊📈")
st.sidebar.markdown(
    "Made with :orange_heart: by [Sawsan Abdulbari](https://www.linkedin.com/in/sawsanabdulbari/)"
)

# Function to simulate live data
def generate_fake_data():
    np.random.seed(0)
    latitudes = np.random.uniform(low=40.63, high=40.85, size=1000)
    longitudes = np.random.uniform(low=-74.03, high=-73.75, size=1000)
    return pd.DataFrame({'latitude': latitudes, 'longitude': longitudes})

def main():
    st.header('Live Dashboard: NYC Taxi Trip Duration')
    
    # Generate fake data
    data = generate_fake_data()
    
    # Display map with fake data using folium
    st.subheader('Map showing current taxi locations')
    
    # Initialize map centered around NYC
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=12)
    
    # Add points to the map
    for _, row in data.iterrows():
        folium.CircleMarker(
            location=(row['latitude'], row['longitude']),
            radius=2,
            color='blue',
            fill=True,
            fill_color='blue'
        ).add_to(m)
    
    # Display map in Streamlit
    st_folium(m, width=700, height=500)
    
# Insert the image at the top of the page

# Helper function to load image with error handling
def load_image(image_path):
    if os.path.exists(image_path):
        return image_path
    else:
        st.error(f"Error: '{image_path}' not found.")
        return None

# Insert the image at the top of the page
image_path = "taxi.png"
image = load_image(image_path)
if image:
    st.image(image, use_container_width=True)
# Project Background
st.markdown("""
## Project Background
In New York City, the ability to predict taxi trip durations accurately is crucial for various stakeholders including passengers, drivers, and fleet managers. This project aims to leverage machine learning to forecast taxi trip durations based on historical data and user inputs.

### Project Goals
- :dart: **Develop a predictive model** to forecast NYC taxi trip durations.
- :bar_chart: **Analyze historical NYC taxi data** to identify patterns and trends.
- :clipboard: **Provide actionable insights and forecasts** to stakeholders for better decision-making.
- :mega: **Enhance public awareness and engagement** by disseminating accurate and useful information about taxi trip durations.
- :handshake: **Foster collaboration and partnerships** between local stakeholders to improve urban mobility.

### How to Use This App
1. :point_right: Navigate to the [**Model**](Model) page.
2. :writing_hand: Enter the required input features.
3. :chart_with_upwards_trend: Get the predicted trip duration.

**Note:** Ensure that all input fields are filled in accurately for the best prediction results.

### Useful Links
- [CS Get-Skilled Academy](https://www.linkedin.com/company/cs-get-skilled/) :mortar_board:
- [New York City Taxi Trip Duration Kaggel](https://www.kaggle.com/c/nyc-taxi-trip-duration) :taxi:
- [Data Repository by the Center for Urban Science and Progress (CUSP) at NYU](https://cusp.nyu.edu/) :globe_with_meridians:

### Contact Us
For more information, please reach out at [info@nyctaxipredictapp.com](sawsan.abdulbari@gmail.com) :email:
""")

# Live Dashboard
def generate_fake_data():
    np.random.seed(0)
    latitudes = np.random.uniform(low=40.63, high=40.85, size=1000)
    longitudes = np.random.uniform(low=-74.03, high=-73.75, size=1000)
    return pd.DataFrame({'latitude': latitudes, 'longitude': longitudes})

def main():
    st.header('Live Dashboard: NYC Taxi Trip Duration')
    
    # Generate fake data
    data = generate_fake_data()
    
    # Display map with fake data using folium
    st.subheader('Map showing current taxi locations')
    
    # Initialize map centered around NYC
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=12)
    
    # Add points to the map
    for _, row in data.iterrows():
        folium.CircleMarker(
            location=(row['latitude'], row['longitude']),
            radius=2,
            color='blue',
            fill=True,
            fill_color='blue'
        ).add_to(m)
    
    # Display map in Streamlit
    st_folium(m, width=700, height=500)


if __name__ == "__main__":
    main()

# Footer
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
    padding: 10px 0;
}
</style>
<div class="footer">
<p>Developed with <span style='color:orange;'>❤</span> by <a href="https://www.linkedin.com/in/sawsanabdulbari/" target="_blank">Sawsan Abdulbari</a></p>
</div>
""", unsafe_allow_html=True)
