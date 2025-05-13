"""Configuration settings for the species occurrence pipeline."""

# GBIF API parameters
GBIF_PARAMS = {
    "hasCoordinate": True,
    "basisOfRecord": "HUMAN_OBSERVATION",
    "year": "2000,2025",  # Records after this year
    "limit": 1000
}

# Background points generation parameters
BACKGROUND_PARAMS = {
    "sampling_method": "buffer",  # Options: "buffer" or "env_stratified"
    "n_background_ratio": 2.0,    # Number of background points relative to presence points
    "buffer_degree": 2.0,         # Geographic expansion from bounding box (degrees)
    "min_distance_km": 5.0,       # Minimum distance from any presence point (km)
    "env_n_clusters": 10,         # Number of clusters for KMeans (stratified sampling)
    "env_points_per_cluster": 50  # Number of background points per cluster center
}

# File paths and naming templates
FILE_PATHS = {
    "input_file": "EU_SpeciesList.xlsx",
    "presence_output": "EU_Species_Presence.xlsx",
    "background_buffer_output": "EU_Species_background_buffer.xlsx",
    "background_env_stratified_output": "EU_Species_background_env_stratified.xlsx",
    "final_dataset_output": "EU_Species_training_dataset.xlsx"
}

# WorldClim data
WORLDCLIM_BASE_URL = "https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_2.5m_bio_{}.tif"
WORLDCLIM_RESOLUTION = "2.5m"  # 2.5 arc-minutes (~5km)
