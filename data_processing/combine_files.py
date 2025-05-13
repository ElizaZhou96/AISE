"""Script to combine presence and background points before extracting bioclimatic variables."""

import pandas as pd
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def combine_files(presence_file, background_file, output_file):
    """
    Combine presence and background points into a single file.
    
    Args:
        presence_file: Path to the presence points file
        background_file: Path to the background points file
        output_file: Path to save the combined file
    """
    try:
        # Read the input files
        presence_df = pd.read_excel(presence_file)
        background_df = pd.read_excel(background_file)
        
        logger.info(f"Read {len(presence_df)} presence points and {len(background_df)} background points")
        
        # Ensure both dataframes have the same columns
        common_columns = list(set(presence_df.columns) & set(background_df.columns))
        if len(common_columns) < len(presence_df.columns) or len(common_columns) < len(background_df.columns):
            logger.warning("The two files have different columns. Only common columns will be kept.")
            logger.info(f"Common columns: {common_columns}")
            
            presence_df = presence_df[common_columns]
            background_df = background_df[common_columns]
        
        # Combine the dataframes
        combined_df = pd.concat([presence_df, background_df], ignore_index=True)
        
        # Save to output file
        combined_df.to_excel(output_file, index=False)
        
        logger.info(f"Combined file saved to {output_file} with {len(combined_df)} points")
        
        return combined_df
        
    except Exception as e:
        logger.error(f"Error combining files: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine presence and background points")
    parser.add_argument("--presence", required=True, help="Presence points file (xxx1_presence.xlsx)")
    parser.add_argument("--background", required=True, help="Background points file (xxx2_background_buffer.xlsx)")
    parser.add_argument("--output", required=True, help="Output combined file")
    
    args = parser.parse_args()
    
    combine_files(args.presence, args.background, args.output)