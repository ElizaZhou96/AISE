"""Module for extracting bioclimatic variables for points."""

import pandas as pd
import numpy as np
import rioxarray
import xarray as xr
import requests
import os
import time
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BioclimExtractor:
    """Class for extracting bioclimatic variables for points."""
    
    def __init__(self, input_file, output_file, worldclim_url):
        """
        Initialize the bioclimatic variable extractor.
        
        Args:
            input_file: Path to the input file with coordinates
            output_file: Path to save the output dataset
            worldclim_url: Base URL for WorldClim bioclimatic variables
        """
        self.input_file = input_file
        self.output_file = output_file
        self.worldclim_url = worldclim_url
        self.bioclim_vars = {}
    
    def download_bioclim_layers(self):
        """
        Download and load bioclimatic variable layers from WorldClim.
        
        Returns:
            Dictionary of xarray.DataArray objects for each bioclimatic variable
        """
        logger.info("Downloading and loading bioclimatic layers...")
        
        # Create a directory for cache if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        # Download and load each bioclimatic variable
        for i in tqdm(range(1, 20), desc="Downloading BIO layers", dynamic_ncols=True):
            var_name = f"BIO{i}"
            url = self.worldclim_url.format(i)
            cache_file = f"temp/wc2.1_2.5m_bio_{i}.tif"
            
            try:
                # Check if the file is already cached
                if os.path.exists(cache_file):
                    logger.info(f"Loading {var_name} from cache")
                    self.bioclim_vars[var_name] = rioxarray.open_rasterio(cache_file, masked=True)
                else:
                    # Download the file
                    logger.info(f"Downloading {var_name} from WorldClim")
                    response = requests.get(url)
                    response.raise_for_status()
                    
                    # Save to cache
                    with open(cache_file, 'wb') as f:
                        f.write(response.content)
                    
                    # Load the file
                    self.bioclim_vars[var_name] = rioxarray.open_rasterio(cache_file, masked=True)
                
                logger.info(f"Successfully loaded {var_name}")
                
                # Sleep briefly to avoid overwhelming the server
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error downloading/loading {var_name}: {str(e)}")
                
        logger.info(f"Loaded {len(self.bioclim_vars)}/19 bioclimatic variables")
        
        return self.bioclim_vars
    
    def extract_variables_for_point(self, lat, lon):
        """
        Extract bioclimatic variables for a single point.
        
        Args:
            lat: Latitude of the point
            lon: Longitude of the point
            
        Returns:
            Dictionary with bioclimatic variable values
        """
        result = {}
        
        for var_name, data_array in self.bioclim_vars.items():
            try:
                # The .sel() method selects data at the specific coordinates
                val = data_array.sel(x=lon, y=lat, method="nearest").values[0]
                
                # Convert to float and handle missing values
                if np.ma.is_masked(val):
                    result[var_name] = np.nan
                else:
                    result[var_name] = float(val)
                    
            except Exception as e:
                logger.error(f"Error extracting {var_name} at ({lat}, {lon}): {str(e)}")
                result[var_name] = np.nan
        
        return result
    
    def extract_variables_for_dataset(self):
        """
        Extract bioclimatic variables for all points in the dataset.
        
        Returns:
            DataFrame with points and their bioclimatic variables
        """
        try:
            # Read input data
            input_df = pd.read_excel(self.input_file)
            
            if input_df.empty:
                logger.error("Input file is empty")
                return pd.DataFrame()
            
            # Check if lat/lon columns exist
            lat_col = "decimalLatitude"
            lon_col = "decimalLongitude"
            
            if lat_col not in input_df.columns or lon_col not in input_df.columns:
                logger.error(f"Input file must contain {lat_col} and {lon_col} columns")
                return pd.DataFrame()
            
            logger.info(f"Processing {len(input_df)} points")
            
            # Download and load bioclimatic variables
            if not self.bioclim_vars:
                self.download_bioclim_layers()
            
            if not self.bioclim_vars:
                logger.error("No bioclimatic variables were loaded")
                return pd.DataFrame()
            
            # Extract variables for each point
            results = []
            
            for idx, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Extracting bioclimatic variables", dynamic_ncols=True):
                lat = row[lat_col]
                lon = row[lon_col]
                
                # Extract bioclimatic variables
                bio_vars = self.extract_variables_for_point(lat, lon)
                
                # Combine with original row data
                point_data = row.to_dict()
                point_data.update(bio_vars)
                
                results.append(point_data)
            
            # Convert to DataFrame
            result_df = pd.DataFrame(results)
            
            # Save to Excel
            result_df.to_excel(self.output_file, index=False)
            logger.info(f"Results saved to {self.output_file}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting bioclimatic variables: {str(e)}")
            raise
