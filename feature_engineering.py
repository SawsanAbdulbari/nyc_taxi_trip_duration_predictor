import numpy as np
import pandas as pd
from utils import bearing_array, create_airport_features, distance, extract_datetime_features, filter_geographic_bounds, manhattan_distance, remove_outliers

def extract_datetime_features(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df['day'] = df['pickup_datetime'].dt.day
    df['month'] = df['pickup_datetime'].dt.month
    df['year'] = df['pickup_datetime'].dt.year
    return df

def prepare_data(train, is_training=True):
    print(f"Initial shape: {train.shape}")
    train = extract_datetime_features(train)
    print(f"After datetime extraction: {train.shape}")
    
    # Default geographic bounds
    longitude_min, longitude_max = -74.2591, -73.7004
    latitude_min, latitude_max = 40.4774, 40.9176
    train = filter_geographic_bounds(train, longitude_min, longitude_max, latitude_min, latitude_max)
    print(f"After geographic bounds filter: {train.shape}")
    print(train)

    train['distance_haversine'] = distance(train['pickup_latitude'],
                                           train['pickup_longitude'],
                                           train['dropoff_latitude'],
                                           train['dropoff_longitude'])
    train['distance_manhattan'] = manhattan_distance(train['pickup_latitude'],
                                                     train['pickup_longitude'],
                                                     train['dropoff_latitude'],
                                                     train['dropoff_longitude'])
    train['direction'] = bearing_array(train['pickup_latitude'],
                                       train['pickup_longitude'],
                                       train['dropoff_latitude'],
                                       train['dropoff_longitude'])
    
    train = create_airport_features(train)
    print(f"After creating airport features: {train.shape}")
    print(train)

    if is_training:
        train['trip_speed'] = train['distance_haversine'] / (train['trip_duration'] / 3600)
        train['log_trip_duration'] = np.log1p(train['trip_duration'])
        train = train.drop(columns=['id', 'pickup_datetime', 'trip_duration'], errors='ignore')
        train = remove_outliers(train, 'log_trip_duration', n_std=2)
        print(f"After removing outliers: {train.shape}")
    else:
        train['vendor_id'] = 1
        train['passenger_count'] = 1
        train['store_and_fwd_flag'] = 'N'
        train['trip_speed'] = train['distance_haversine'] / (1 + 10 / 3600)  # Assuming average speed
        print(f"Final shape for prediction: {train.shape}")

    return train
