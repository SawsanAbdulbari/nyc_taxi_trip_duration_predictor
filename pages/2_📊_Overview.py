import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import plotly.figure_factory as ff

import warnings
from feature_engineering import prepare_data
from utils import bearing_array, create_airport_features, distance, extract_datetime_features, filter_geographic_bounds, manhattan_distance, remove_outliers

warnings.filterwarnings('ignore')
st.set_page_config(page_title="NYC Taxi Data Dashboard", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
.main .block-container {
    max-width: 85%;
    padding-top: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
</style>
""", unsafe_allow_html=True)

st.title(":bar_chart: :orange[Overview Of NYC Taxi Data]")

# Sidebar information
st.sidebar.title('NYC Taxi Dashboard')
st.sidebar.write("This dashboard uses the NYC Taxi dataset for analysis.")

# Helper function to load images with error handling
def load_image(image_path):
    if os.path.exists(image_path):
        return image_path
    else:
        st.error(f"Error: '{image_path}' not found.")
        return None

image_path = load_image("img.png")
if image_path:
    st.sidebar.image(image_path, caption='NYC Taxi(AI)', use_column_width=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("train.csv")
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'train.csv' is in the correct path.")
        return None

df = load_data()
if df is not None:
    df = prepare_data(df, is_training=False)

    total_trips = len(df)
    total_passengers = df['passenger_count'].sum()
    total_trip_duration = df['trip_duration'].sum() / 3600
    average_speed = df['distance_haversine'].sum() / total_trip_duration

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Total Trips", total_trips)
    a2.metric("Total Passengers", total_passengers)
    a3.metric("Total Trip Duration (hours)", f"{total_trip_duration:,.2f}")
    a4.metric("Average Speed (mph)", f"{average_speed:.2f}")

    if 'payment_type' in df.columns:
        payment_type = st.sidebar.multiselect("Pick Payment Type", df["payment_type"].unique())
        if payment_type:
            df = df[df["payment_type"].isin(payment_type)]

    st.sidebar.header("Date Range Filters:")
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2016-01-01'))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime('2016-12-31'))

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    if start_date > end_date:
        st.error("Start date should be before the end date.")
    else:
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        df = df[(df['pickup_datetime'] >= start_date) & (df['pickup_datetime'] <= end_date)]

    st.subheader("Visualizations")

    col1, col2 = st.columns((2))
    with col1:
        st.subheader("Trip Distance by Pickup Location")
        trip_distance = df.groupby(by=['pickup_latitude', 'pickup_longitude'], as_index=False)['distance_haversine'].sum().round(2)
        fig = px.scatter_mapbox(
            trip_distance, 
            lat="pickup_latitude", 
            lon="pickup_longitude", 
            size="distance_haversine", 
            color="distance_haversine",
            color_continuous_scale=px.colors.cyclical.IceFire,
            size_max=15, 
            zoom=10,
            mapbox_style="carto-positron",
            title="Trip Distance by Pickup Location"
        )
        fig.update_layout(
            autosize=False,
            width=1000,
            height=800,
            margin={"r":0, "t":50, "l":0, "b":0},
            coloraxis_colorbar=dict(
                title="Trip Distance",
                tickvals=[0, df['distance_haversine'].max()/2, df['distance_haversine'].max()],
                ticktext=["Low", "Medium", "High"]
            )
        )
        st.plotly_chart(fig)

    st.subheader("Time Series Analysis of Trip Duration")
    st.markdown("Analyze trip duration over a specified date range.")
    time_series_data = df.groupby(df['pickup_datetime'].dt.strftime("%Y-%b"))['trip_duration'].sum().reset_index()
    fig = px.line(time_series_data, x='pickup_datetime', y="trip_duration", markers=True, labels={"trip_duration": "Total Trip Duration (seconds)"})
    fig.update_xaxes(type='category')
    fig.update_layout(height=500, width=1000, template="plotly_white")

    trendline = go.Scatter(x=time_series_data['pickup_datetime'], y=time_series_data['trip_duration'],
                           mode='lines', line=dict(color='red'), name='Trendline')
    fig.add_trace(trendline)

    fig.add_annotation(x="2016-Mar", y=650000, 
                       text="Significant Drop",
                       showarrow=True, arrowhead=1,
                       arrowsize=1.5, arrowwidth=2)

    st.plotly_chart(fig, use_container_width=True)

    plot_vendor = df.groupby('vendor_id')['trip_duration'].mean().reset_index()

    st.subheader("Average Trip Duration per Vendor")
    fig = px.bar(
        plot_vendor,
        x='vendor_id',
        y='trip_duration',
        text='trip_duration',
        title='Average Trip Duration per Vendor',
        labels={'vendor_id': 'Vendor ID', 'trip_duration': 'Average Trip Duration (Seconds)'},
        color='vendor_id',
        color_continuous_scale='viridis'
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        width=800,
        height=300,
        margin={"r":0, "t":50, "l":0, "b":0},
        yaxis=dict(range=[plot_vendor['trip_duration'].min() * 0.9, plot_vendor['trip_duration'].max() * 1.1])
    )
    st.plotly_chart(fig)

    def view_and_download_data(data, download_filename, download_label):
        st.write(data.style.background_gradient(cmap="Blues"))
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(download_label, 
                           data=csv, 
                           file_name=download_filename, 
                           mime="text/csv", 
                           help=f'Click here to download the {download_label} as a CSV file')

    cl1, cl2 = st.columns(2)

    with cl1:
        with st.expander("Pickup Location Data"):
            view_and_download_data(trip_distance, "Pickup_Location_Data.csv", "Download Pickup Location Data")

    if 'payment_type' in df.columns:
        with cl2:
            with st.expander("Payment Type Data"):
                payment_type_data = df.groupby(by='payment_type', as_index=False)['trip_duration'].sum().round(2)
                view_and_download_data(payment_type_data, "Payment_Type_Data.csv", "Download Payment Type Data")

    with st.expander("View Filtered Data"):
        st.write(df.head(500).style.background_gradient(cmap="Oranges"))

        csv_filtered = df.to_csv(index=False).encode('utf-8')
        st.download_button('Download Filtered Data', 
                           data=csv_filtered, 
                           file_name="Filtered_NYC_Taxi_Data.csv",
                           mime="text/csv")

    st.sidebar.write("")
    st.sidebar.markdown(
        "Made with :orange_heart: by [Sawsan Abdulbari](https://www.linkedin.com/in/sawsanabdulbari/)"
    )

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["month_name"] = df["pickup_datetime"].dt.month_name()

    st.subheader(":point_right: Monthly Trip Summary")
    with st.expander("Summary Table"):
        df_sample = df[0:5][["pickup_latitude", "pickup_longitude", "distance_haversine", "trip_duration", "passenger_count"]]
        fig = ff.create_table(df_sample, colorscale="Cividis")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("Month-wise Trip Table")
        sub_category_Year = pd.pivot_table(data=df, 
                                           values="trip_duration", 
                                           index=["pickup_latitude"],
                                           columns="month_name")
        st.write(sub_category_Year.style.background_gradient(cmap="Blues"))
