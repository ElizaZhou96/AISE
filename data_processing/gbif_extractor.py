"""Module for extracting species occurrence data from GBIF."""

import pandas as pd
import numpy as np
from pygbif import occurrences
import time
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GBIFExtractor:
    """Class for extracting species occurrence data from GBIF."""
    
    def __init__(self, input_file, output_file, gbif_params):
        """
        Initialize the GBIF extractor.
        
        Args:
            input_file: Path to the input Excel file with species names
            output_file: Path to save the extracted occurrence data
            gbif_params: Dictionary of parameters for the GBIF API
        """
        self.input_file = input_file
        self.output_file = output_file
        self.gbif_params = gbif_params
        
    def read_species_list(self):
        """
        Read the list of species names from the input file.
        
        Returns:
            List of scientific names
        """
        try:
            # Read the Excel file
            df = pd.read_excel(self.input_file)
            
            # Check if the required column exists
            if "Scientific Name" not in df.columns:
                logger.error("Input file does not contain a 'Scientific Name' column")
                raise ValueError("Input file must contain a 'Scientific Name' column")
            
            # Extract unique species names
            species_list = df["Scientific Name"].unique().tolist()
            logger.info(f"Found {len(species_list)} unique species in the input file")
            
            return species_list
        except Exception as e:
            logger.error(f"Error reading species list: {str(e)}")
            raise
    
    def query_gbif(self, species_name):
        """
        Query the GBIF API for a single species.
        
        Args:
            species_name: Scientific name of the species
            
        Returns:
            DataFrame with occurrence records
        """
        try:
            logger.info(f"Querying GBIF for species: {species_name}")
            
            # Build query parameters
            params = {
                "scientificName": species_name,
                **self.gbif_params
            }
            
            # Make sure year is correctly formatted as a string with a comparison operator
            if "year" in params and isinstance(params["year"], int):
                params["year"] = f">{params['year']}"
            
            # Execute the query
            response = occurrences.search(**params)
            
            # Extract records
            records = []
            if "results" in response and response["results"]:
                for record in response["results"]:
                    # Check if the record has coordinates
                    if "decimalLatitude" in record and "decimalLongitude" in record:
                        # Extract required fields
                        extracted_record = {
                            "scientificName": record.get("scientificName", species_name),
                            "decimalLatitude": record.get("decimalLatitude"),
                            "decimalLongitude": record.get("decimalLongitude"),
                            "year": record.get("year"),
                            "country": record.get("country"),
                            "countryCode": record.get("countryCode"),
                            "elevation": record.get("elevation")
                        }
                        records.append(extracted_record)
            
            logger.info(f"Retrieved {len(records)} valid occurrence records for {species_name}")
            
            # Convert to DataFrame
            if records:
                return pd.DataFrame(records)
            else:
                return pd.DataFrame(columns=[
                    "scientificName", "decimalLatitude", "decimalLongitude", 
                    "year", "country", "countryCode", "elevation"
                ])
                
        except Exception as e:
            logger.error(f"Error querying GBIF for {species_name}: {str(e)}")
            # Return empty DataFrame on error
            return pd.DataFrame(columns=[
                "scientificName", "decimalLatitude", "decimalLongitude", 
                "year", "country", "countryCode", "elevation"
            ])
    
    def extract_all_species(self):
        """
        Extract occurrence records for all species in the list.
        
        Returns:
            DataFrame with combined occurrence records for all species
        """
        # Read species list
        species_list = self.read_species_list()
        
        if not species_list:
            logger.warning("No species found in the input file")
            return pd.DataFrame()
        
        # Query GBIF for each species
        all_records = []
        
        for species in tqdm(species_list, desc="Extracting species data", dynamic_ncols=True):
            # Query GBIF
            species_df = self.query_gbif(species)
            
            # Add to the list if not empty
            if not species_df.empty:
                all_records.append(species_df)
            
            # Sleep to avoid overwhelming the API
            time.sleep(1)
        
        # Combine all records
        if all_records:
            combined_df = pd.concat(all_records, ignore_index=True)
            logger.info(f"Total records extracted: {len(combined_df)}")
            return combined_df
        else:
            logger.warning("No valid occurrence records found for any species")
            return pd.DataFrame()
    
    def process_and_save(self):
        """
        Process all species and save the results to an Excel file.
        
        Returns:
            DataFrame with the processed results
        """
        try:
            # Extract all species data
            presence_df = self.extract_all_species()
            
            if presence_df.empty:
                logger.warning("No data to save")
                return presence_df
            
            # Add a presence column (1 for presence)
            presence_df["Presence"] = 1
            
            # Save to Excel
            presence_df.to_excel(self.output_file, index=False)
            logger.info(f"Results saved to {self.output_file}")
            
            return presence_df
        
        except Exception as e:
            logger.error(f"Error processing and saving data: {str(e)}")
            raise
