"""Standalone script to extract bioclimatic variables for occurrence points."""

import pandas as pd
import numpy as np
import rioxarray
import requests
import os
import time
import logging
import argparse
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bioclim_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BioclimExtractor:
    """Class for extracting bioclimatic variables for points."""
    
    def __init__(self, input_file, output_file, worldclim_base_url):
        """
        Initialize the bioclimatic variable extractor.
        
        Args:
            input_file: Path to the input file with coordinates
            output_file: Path to save the output dataset
            worldclim_base_url: Base URL for WorldClim bioclimatic variables
        """
        self.input_file = input_file
        self.output_file = output_file
        self.worldclim_base_url = worldclim_base_url
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
        for i in tqdm(range(1, 20), desc="Downloading BIO layers"):
            var_name = f"BIO{i}"
            url = self.worldclim_base_url.format(i)
            cache_file = f"temp/wc2.1_2.5m_bio/wc2.1_2.5m_bio_{i}.tif"
            
            try:
                # Check if the file is already cached
                if os.path.exists(cache_file):
                    logger.info(f"Loading {var_name} from cache")
                    self.bioclim_vars[var_name] = rioxarray.open_rasterio(cache_file, masked=True)
                else:
                    # Download the file
                    logger.info(f"Downloading {var_name} from WorldClim")
                    try:
                        response = requests.get(url, timeout=15)
                        response.raise_for_status()
                        with open(cache_file, 'wb') as f:
                            f.write(response.content)
                        self.bioclim_vars[var_name] = rioxarray.open_rasterio(cache_file, masked=True)
                    except Exception as e:
                        logger.warning(f"Skipping {var_name} due to download failure: {e}")
                        continue
                    
                
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
            if self.input_file.endswith('.xlsx'):
                input_df = pd.read_excel(self.input_file)
            elif self.input_file.endswith('.csv'):
                input_df = pd.read_csv(self.input_file)
            else:
                logger.error("Input file must be .xlsx or .csv")
                return pd.DataFrame()
            
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
            
            for idx, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Extracting bioclimatic variables"):
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
            
            # Save to output file
            if self.output_file.endswith('.xlsx'):
                result_df.to_excel(self.output_file, index=False)
            elif self.output_file.endswith('.csv'):
                result_df.to_csv(self.output_file, index=False)
            else:
                # Default to Excel
                result_df.to_excel(self.output_file, index=False)
                
            logger.info(f"Results saved to {self.output_file}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting bioclimatic variables: {str(e)}")
            raise

def main():
    """Main function to extract bioclimatic variables."""
    parser = argparse.ArgumentParser(description="Extract Bioclimatic Variables for Occurrence Points")
    parser.add_argument("--input", required=True, help="Input file with coordinates (.xlsx or .csv)")
    parser.add_argument("--output", required=True, help="Output file for results (.xlsx or .csv)")
    
    args = parser.parse_args()
    
    # WorldClim URL template
    worldclim_base_url = "https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_2.5m_bio_{}.tif"
    
    try:
        logger.info("=== Extracting Bioclimatic Variables ===")
        logger.info(f"Input file: {args.input}")
        logger.info(f"Output file: {args.output}")
        
        # Initialize the bioclimatic variable extractor
        bioclim_extractor = BioclimExtractor(
            args.input,
            args.output,
            worldclim_base_url
        )
        
        # Extract variables for all points
        bioclim_extractor.extract_variables_for_dataset()
        
        logger.info("=== Bioclimatic Variable Extraction completed successfully ===")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()