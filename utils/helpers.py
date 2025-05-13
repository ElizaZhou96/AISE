"""Helper functions for the species occurrence pipeline."""

import numpy as np
from shapely.geometry import Point
import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on Earth.
    
    Args:
        lat1, lon1: Coordinates of the first point (in decimal degrees)
        lat2, lon2: Coordinates of the second point (in decimal degrees)
        
    Returns:
        Distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    
    return c * r

def create_bounding_box(points_df, buffer_degree=2.0):
    """
    Create a bounding box around a set of points with a buffer.
    
    Args:
        points_df: DataFrame containing latitude and longitude columns
        buffer_degree: Buffer size in degrees to add to the bounding box
        
    Returns:
        Dictionary with min/max latitude and longitude
    """
    min_lat = points_df['decimalLatitude'].min() - buffer_degree
    max_lat = points_df['decimalLatitude'].max() + buffer_degree
    min_lon = points_df['decimalLongitude'].min() - buffer_degree
    max_lon = points_df['decimalLongitude'].max() + buffer_degree
    
    # Ensure coordinates are within valid ranges
    min_lat = max(-90, min_lat)
    max_lat = min(90, max_lat)
    min_lon = max(-180, min_lon)
    max_lon = min(180, max_lon)
    
    return {
        'min_lat': min_lat,
        'max_lat': max_lat,
        'min_lon': min_lon,
        'max_lon': max_lon
    }

def points_to_gdf(df, lat_col='decimalLatitude', lon_col='decimalLongitude'):
    """
    Convert a DataFrame with latitude and longitude columns to a GeoDataFrame.
    
    Args:
        df: DataFrame with latitude and longitude columns
        lat_col: Name of the latitude column
        lon_col: Name of the longitude column
        
    Returns:
        GeoDataFrame with Point geometry
    """
    import geopandas as gpd
    
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

def is_on_land(lon, lat, land_mask_gdf):
    """
    Check if a point is on land using a land mask.
    
    Args:
        lon, lat: Coordinates of the point
        land_mask_gdf: GeoDataFrame with land polygons
        
    Returns:
        Boolean indicating if the point is on land
    """
    point = Point(lon, lat)
    return land_mask_gdf.contains(point).any()

def standardize_features(df, feature_cols):
    """
    Standardize features to have zero mean and unit variance.
    
    Args:
        df: DataFrame with feature columns
        feature_cols: List of feature column names
        
    Returns:
        DataFrame with standardized features and scaling parameters
    """
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    df_std = df.copy()
    
    # Extract the features as a numpy array
    features = df[feature_cols].values
    
    # Fit the scaler and transform the features
    features_std = scaler.fit_transform(features)
    
    # Update the DataFrame with standardized values
    for i, col in enumerate(feature_cols):
        df_std[col] = features_std[:, i]
    
    # Return both the standardized DataFrame and the scaler for inverse transformation
    return df_std, scaler
