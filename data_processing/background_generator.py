"""Module for generating background (pseudo-absence) points."""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, box
import logging
from sklearn.cluster import KMeans
from tqdm import tqdm
import os
import requests
import zipfile
import io
import time

# Import helper functions from utils
from utils.helpers import (
    haversine_distance, 
    create_bounding_box, 
    points_to_gdf,
    is_on_land,
    standardize_features
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BackgroundGenerator:
    """Class for generating background (pseudo-absence) points."""
    
    def __init__(self, presence_file, output_file, params):
        """
        Initialize the background generator.
        
        Args:
            presence_file: Path to the presence data Excel file
            output_file: Path to save the generated background points
            params: Dictionary of parameters for background point generation
        """
        self.presence_file = presence_file
        self.output_file = output_file
        self.params = params
        
        # Ensure the method is valid
        valid_methods = ["buffer", "env_stratified"]
        if params["sampling_method"] not in valid_methods:
            raise ValueError(f"sampling_method must be one of {valid_methods}")
    
    def get_land_mask(self):
        """
        Get a land mask for filtering background points.
        
        Returns:
            GeoDataFrame with land polygons
        """
        logger.info("Obtaining land mask...")
        
        # URL for Natural Earth land shapefile (10m resolution)
        ne_url = "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip"
        
        try:
            # Create a temporary directory for downloaded files
            os.makedirs("temp", exist_ok=True)
            zip_path = "temp/ne_land.zip"
            
            # Download the file
            response = requests.get(ne_url)
            response.raise_for_status()
            
            # Save to a temporary zip file
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # Extract the shapefile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("temp")
            
            # Read the shapefile
            land_mask = gpd.read_file("temp/ne_10m_land.shp")
            
            logger.info("Land mask obtained successfully")
            return land_mask
            
        except Exception as e:
            logger.error(f"Error obtaining land mask: {str(e)}")
            logger.warning("Proceeding without land mask - background points may include water areas")
            return None
    
    def buffer_method(self, presence_df, species_name):
        """
        Generate background points using the buffer method.
        
        Args:
            presence_df: DataFrame with presence points for a species
            species_name: Name of the species
            
        Returns:
            DataFrame with generated background points
        """
        logger.info(f"Generating background points for {species_name} using buffer method")
        
        # Get parameters
        buffer_degree = self.params["buffer_degree"]
        n_background_ratio = self.params["n_background_ratio"]
        min_distance_km = self.params["min_distance_km"]
        
        # Create bounding box
        bbox = create_bounding_box(presence_df, buffer_degree)
        
        # Calculate number of background points to generate
        n_presence = len(presence_df)
        n_background = int(n_presence * n_background_ratio)
        
        logger.info(f"Target: {n_background} background points for {n_presence} presence points")
        
        # Get land mask for filtering
        land_mask = self.get_land_mask()
        
        # Generate random points within the bounding box
        # We'll generate more than needed to account for filtering
        safety_factor = 3  # Generate 3x more points than needed initially
        background_points = []
        points_generated = 0
        
        max_attempts = n_background * safety_factor * 2  # Limit attempts to avoid infinite loops
        attempts = 0
        
        with tqdm(total=n_background, desc=f"Generating points for {species_name}", dynamic_ncols=True) as pbar:
            while points_generated < n_background and attempts < max_attempts:
                attempts += 1
                
                # Generate a random point within the bounding box
                lat = np.random.uniform(bbox["min_lat"], bbox["max_lat"])
                lon = np.random.uniform(bbox["min_lon"], bbox["max_lon"])
                
                # Check if the point is far enough from any presence point
                is_far_enough = True
                for _, row in presence_df.iterrows():
                    distance = haversine_distance(
                        lat, lon, row["decimalLatitude"], row["decimalLongitude"]
                    )
                    if distance < min_distance_km:
                        is_far_enough = False
                        break
                
                # Check if the point is on land (if land mask is available)
                on_land = True
                if land_mask is not None:
                    point = Point(lon, lat)
                    on_land = land_mask.contains(point).any()
                
                # Add the point if it passes all filters
                if is_far_enough and on_land:
                    background_points.append({
                        "scientificName": species_name,
                        "decimalLatitude": lat,
                        "decimalLongitude": lon,
                        "Presence": 0  # 0 for background/absence
                    })
                    points_generated += 1
                    pbar.update(1)
                
                # Break if we have enough points
                if points_generated >= n_background:
                    break
        
        if points_generated < n_background:
            logger.warning(
                f"Could only generate {points_generated}/{n_background} background points "
                f"for {species_name} after {attempts} attempts"
            )
        
        # Convert to DataFrame
        if background_points:
            return pd.DataFrame(background_points)
        else:
            return pd.DataFrame(columns=["scientificName", "decimalLatitude", "decimalLongitude", "Presence"])
    
    def env_stratified_method(self, presence_df, species_name, bio_vars_df):
        """
        Generate background points using the environmental space stratified method.
        
        Args:
            presence_df: DataFrame with presence points for a species
            species_name: Name of the species
            bio_vars_df: DataFrame with bioclimatic variables for presence points
            
        Returns:
            DataFrame with generated background points
        """
        logger.info(f"Generating background points for {species_name} using environmental stratified method")
        
        # Get parameters
        n_clusters = self.params["env_n_clusters"]
        points_per_cluster = self.params["env_points_per_cluster"]
        n_background_ratio = self.params["n_background_ratio"]
        
        # Ensure we have bioclimatic variables
        if bio_vars_df is None or bio_vars_df.empty:
            logger.error(f"Bioclimatic variables not available for {species_name}")
            return pd.DataFrame()
        
        # Get presence points for this species with bioclimatic variables
        species_presence = pd.merge(
            presence_df[presence_df["scientificName"] == species_name],
            bio_vars_df,
            on=["decimalLatitude", "decimalLongitude"],
            how="inner"
        )
        
        if species_presence.empty:
            logger.error(f"No presence points with bioclimatic variables for {species_name}")
            return pd.DataFrame()
        
        # Get bioclimatic columns
        bio_cols = [col for col in species_presence.columns if col.startswith("BIO")]
        
        if not bio_cols:
            logger.error(f"No bioclimatic variables found in the dataset for {species_name}")
            return pd.DataFrame()
        
        # Standardize the environmental variables
        species_presence_std, scaler = standardize_features(species_presence, bio_cols)
        
        # Calculate number of background points to generate
        n_presence = len(species_presence)
        target_n_background = int(n_presence * n_background_ratio)
        
        # Adjust the number of clusters if necessary
        n_clusters = min(n_clusters, n_presence)
        
        # Adjust points per cluster to match target background points
        if n_clusters > 0:
            points_per_cluster = max(1, target_n_background // n_clusters)
        
        logger.info(
            f"Using {n_clusters} clusters with {points_per_cluster} points each "
            f"(target: {target_n_background} background points)"
        )
        
        if n_clusters == 0:
            logger.error(f"No clusters can be created for {species_name}")
            return pd.DataFrame()
        
        # Cluster the environmental points
        try:
            # Extract standardized environmental variables
            X = species_presence_std[bio_cols].values
            
            # Fit KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(X)
            
            # Get cluster centers
            centers = kmeans.cluster_centers_
            
            # Generate background points around each cluster center
            background_points = []
            
            for i, center in enumerate(centers):
                # Generate points with small Gaussian noise
                for _ in range(points_per_cluster):
                    # Add small noise to each dimension
                    noise = np.random.normal(0, 0.3, len(center))
                    point_env = center + noise
                    
                    # Create a record with the environmental variables
                    record = {"scientificName": species_name, "Presence": 0}
                    
                    # Add environmental variables
                    for j, col in enumerate(bio_cols):
                        record[col] = point_env[j]
                    
                    background_points.append(record)
            
            # Convert to DataFrame
            background_df = pd.DataFrame(background_points)
            
            # Inverse transform to get the original scale of environmental variables
            background_env = background_df[bio_cols].values
            background_env_orig = scaler.inverse_transform(background_env)
            
            # Update DataFrame with original scale values
            for i, col in enumerate(bio_cols):
                background_df[col] = background_env_orig[:, i]
            
            logger.info(f"Generated {len(background_df)} environmental background points for {species_name}")
            
            return background_df
        
        except Exception as e:
            logger.error(f"Error in environmental stratified method for {species_name}: {str(e)}")
            return pd.DataFrame()
    
    def generate_background_points(self, bio_vars_df=None):
        """
        Generate background points for all species.
        
        Args:
            bio_vars_df: Optional DataFrame with bioclimatic variables for presence points
        
        Returns:
            DataFrame with generated background points for all species
        """
        try:
            # Read presence data
            presence_df = pd.read_excel(self.presence_file)
            
            if presence_df.empty:
                logger.error("No presence data found")
                return pd.DataFrame()
            
            # Get unique species
            species_list = presence_df["scientificName"].unique()
            
            # Generate background points for each species
            all_background = []
            
            for species in species_list:
                # Get presence points for this species
                species_presence = presence_df[presence_df["scientificName"] == species]
                
                if self.params["sampling_method"] == "buffer":
                    # Buffer method
                    background_df = self.buffer_method(species_presence, species)
                else:
                    # Environmental stratified method
                    background_df = self.env_stratified_method(species_presence, species, bio_vars_df)
                
                if not background_df.empty:
                    all_background.append(background_df)
                    
                # Sleep briefly to avoid resource overuse
                time.sleep(0.5)
            
            # Combine all background points
            if all_background:
                combined_df = pd.concat(all_background, ignore_index=True)
                logger.info(f"Total background points generated: {len(combined_df)}")
                
                # Save to Excel
                combined_df.to_excel(self.output_file, index=False)
                logger.info(f"Background points saved to {self.output_file}")
                
                return combined_df
            else:
                logger.warning("No background points were generated")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error generating background points: {str(e)}")
            raise
