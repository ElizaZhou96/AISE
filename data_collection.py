"""Main module for the species occurrence pipeline."""

import argparse
import logging
import os
import pandas as pd
from config import GBIF_PARAMS, BACKGROUND_PARAMS, FILE_PATHS, WORLDCLIM_BASE_URL
from data_processing.gbif_extractor import GBIFExtractor
from data_processing.background_generator import BackgroundGenerator
from data_processing.bioclim_extractor import BioclimExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_pipeline(input_file=None, output_dir=None, sampling_method=None):
    """
    Run the complete data processing pipeline.
    
    Args:
        input_file: Path to the input Excel file with species names (overrides config)
        output_dir: Directory to save output files (overrides config)
        sampling_method: Method for background point generation (overrides config)
    """
    try:
        # Update file paths if custom input/output is provided
        if input_file:
            FILE_PATHS["input_file"] = input_file
            
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for key in FILE_PATHS:
                if key != "input_file":
                    FILE_PATHS[key] = os.path.join(output_dir, os.path.basename(FILE_PATHS[key]))
        
        # Update sampling method if provided
        if sampling_method:
            BACKGROUND_PARAMS["sampling_method"] = sampling_method
        
        sampling_method = BACKGROUND_PARAMS["sampling_method"]
        background_output = (
            FILE_PATHS["background_buffer_output"] 
            if sampling_method == "buffer" 
            else FILE_PATHS["background_env_stratified_output"]
        )
        
        logger.info("=== Starting Species Occurrence Pipeline ===")
        logger.info(f"Input file: {FILE_PATHS['input_file']}")
        logger.info(f"Sampling method: {sampling_method}")
        
        # Step 1: Extract Presence Records from GBIF
        logger.info("=== Step 1: Extracting Presence Records from GBIF ===")
        gbif_extractor = GBIFExtractor(
            FILE_PATHS["input_file"],
            FILE_PATHS["presence_output"],
            GBIF_PARAMS
        )
        presence_df = gbif_extractor.process_and_save()
        
        if presence_df.empty:
            logger.error("No presence data available. Pipeline cannot continue.")
            return
        
        # Step 2: Generate Background Points
        logger.info("=== Step 2: Generating Background Points ===")
        
        background_generator = BackgroundGenerator(
            FILE_PATHS["presence_output"],
            background_output,
            BACKGROUND_PARAMS
        )
        
        if sampling_method == "buffer":
            # Buffer method doesn't need bioclimatic variables
            background_df = background_generator.generate_background_points()
        else:
            # For env_stratified method, we need bioclimatic variables for presence points first
            logger.info("Extracting bioclimatic variables for presence points (needed for env_stratified method)")
            
            presence_bioclim_extractor = BioclimExtractor(
                FILE_PATHS["presence_output"],
                "temp_presence_with_bio.xlsx",
                WORLDCLIM_BASE_URL
            )
            presence_bio_df = presence_bioclim_extractor.extract_variables_for_dataset()
            
            # Now generate background points using the environmental variables
            background_df = background_generator.generate_background_points(presence_bio_df)
        
        if background_df.empty:
            logger.error("No background points were generated. Pipeline cannot continue.")
            return
        
        # Step 3: Extract Bioclimatic Variables for all points
        logger.info("=== Step 3: Extracting Bioclimatic Variables ===")
        
        # If we're using env_stratified, we may already have BIO variables for background points
        if sampling_method == "buffer" or not any(col.startswith("BIO") for col in background_df.columns):
            bioclim_extractor = BioclimExtractor(
                background_output,
                FILE_PATHS["final_dataset_output"],
                WORLDCLIM_BASE_URL
            )
            bioclim_extractor.extract_variables_for_dataset()
        else:
            # Combine presence and background points
            logger.info("Combining presence and background datasets")
            combined_df = pd.concat([presence_bio_df, background_df], ignore_index=True)
            combined_df.to_excel(FILE_PATHS["final_dataset_output"], index=False)
            logger.info(f"Combined dataset saved to {FILE_PATHS['final_dataset_output']}")
        
        logger.info("=== Pipeline completed successfully ===")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Species Occurrence Data Processing Pipeline")
    parser.add_argument("--input", help="Input Excel file with species names")
    parser.add_argument("--output", help="Output directory for results")
    parser.add_argument(
        "--method", 
        choices=["buffer", "env_stratified"],
        help="Method for background point generation"
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        input_file=args.input,
        output_dir=args.output,
        sampling_method=args.method
    )
