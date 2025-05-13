"""
explain_rule.py

This module reads causal effect estimates from DoWhy (e.g., from a file like 'dowhy_results.xlsx')
and generates structured natural language explanations for each species and each bioclimatic variable (BIO).

It uses a rule-based interpretation system based on effect size and direction, and can output plain text files
per species. Designed for integration into biodiversity causal inference pipelines.
"""

import pandas as pd
import os

# Human-readable descriptions for each BIO variable (can be extended)
BIO_DESC = {
    "BIO1": "Annual Mean Temperature",
    "BIO2": "Mean Diurnal Range (Mean of monthly max temp - min temp)",
    "BIO3": "Isothermality (BIO2/BIO7)*100",
    "BIO4": "Temperature Seasonality (standard deviation *100)",
    "BIO5": "Max Temperature of Warmest Month",
    "BIO6": "Min Temperature of Coldest Month",
    "BIO7": "Temperature Annual Range (BIO5-BIO6)",
    "BIO8": "Mean Temperature of Wettest Quarter",
    "BIO9": "Mean Temperature of Driest Quarter",
    "BIO10": "Mean Temperature of Warmest Quarter",
    "BIO11": "Mean Temperature of Coldest Quarter",
    "BIO12": "Annual Precipitation",
    "BIO13": "Precipitation of Wettest Month",
    "BIO14": "Precipitation of Driest Month",
    "BIO15": "Precipitation Seasonality (Coefficient of Variation)",
    "BIO16": "Precipitation of Wettest Quarter",
    "BIO17": "Precipitation of Driest Quarter",
    "BIO18": "Precipitation of Warmest Quarter",
    "BIO19": "Precipitation of Coldest Quarter"
}

def explain_bio(treatment, effect, controls):
    """
    Generate a structured explanation string for one BIO variable's causal effect.

    Parameters:
    - treatment (str): Name of the BIO variable (e.g., 'BIO11')
    - effect (float): Estimated causal effect (from DoWhy)
    - controls (list or str): Common cause variables controlled in the model

    Returns:
    - str: Explanation block in plain text format
    """
    direction = "✅" if abs(effect) > 0.05 else "❌"  # Use checkmark for significant effect
    effect_str = f"{effect:+.4f}"  # Format effect with sign and 4 decimal places
    ctrl_text = ", ".join(controls) if isinstance(controls, list) else controls or "None"
    bio_name = BIO_DESC.get(treatment, treatment)  # Fall back to raw name if not found

    # Rule-based explanation generation
    if abs(effect) > 0.05:
        if effect > 0:
            interpretation = f"Higher {bio_name} increases the probability of species occurrence. " \
                             f"This may reflect habitat preference or physiological suitability."
        else:
            interpretation = f"Higher {bio_name} decreases the probability of species occurrence. " \
                             f"This could indicate climatic stress or ecological limitations."
    else:
        interpretation = f"{bio_name} shows weak or negligible causal effect on species presence."

    return f"""{treatment} — {bio_name}
{direction} Effect: {effect_str} | Controls: {ctrl_text}
Explanation: {interpretation}
"""

def generate_explanation_per_species(species_df, species_name):
    """
    Generate full explanation text for one species from its DoWhy results.

    Parameters:
    - species_df (pd.DataFrame): DataFrame with DoWhy results (one species only)
    - species_name (str): Name of the species

    Returns:
    - str: Multi-line text block of explanation
    """
    lines = [f"\n\u2705 Species: {species_name}\n"]

    # Iterate through all estimated treatment effects for this species
    for _, row in species_df.iterrows():
        treatment = row['Treatment']
        effect = row['Effect']
        # Parse controls if stored as stringified list
        if isinstance(row['CommonCauses'], str):
            try:
                controls = eval(row['CommonCauses'])
            except:
                controls = []
        else:
            controls = row['CommonCauses']

        explanation = explain_bio(treatment, effect, controls)
        lines.append(explanation)

    # Optional: Add a summary of most impactful variables
    top = species_df.iloc[species_df['Effect'].abs().argsort()[::-1][:2]]
    top_vars = ", ".join(top['Treatment'].tolist())
    summary = f"\n📈 Summary: {species_name}'s occurrence is primarily influenced by {top_vars}."
    lines.append(summary)

    return "\n".join(lines)

def generate_explanations_from_file(input_path, output_dir="outputs/species_explanations"):
    """
    Batch process an Excel file of DoWhy results and write structured explanations for each species.

    Parameters:
    - input_path (str): Path to input Excel file (e.g., 'outputs/dowhy_results.xlsx')
    - output_dir (str): Folder to write text files
    """
    df = pd.read_excel(input_path)
    os.makedirs(output_dir, exist_ok=True)

    for species in df["Species"].unique():
        species_df = df[df["Species"] == species].copy()
        text = generate_explanation_per_species(species_df, species)

        file_name = f"{species.replace(' ', '_')}_rule1_explanation.txt"
        with open(os.path.join(output_dir, file_name), "w", encoding="utf-8") as f:
            f.write(text)

        print(f"✅ Explanation written for: {species}")

generate_explanations_from_file("outputs/dowhy_results.xlsx")