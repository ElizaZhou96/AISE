import pandas as pd
from dowhy import CausalModel
from notears_linear import run_notears_l_and_plot_colored 
from notears_nonlinear import run_notears_n_and_plot_colored 
import warnings
from select_candidates import get_combined_bio_candidates
import time
import torch
import os

# Make sure outputs directory exists
os.makedirs("outputs", exist_ok=True)  # Fix syntax error in the quotes

def extract_common_causes(G, treatment):
    if treatment not in G:
        raise ValueError(f"[Error] Treatment variable '{treatment}' not found in DAG.")
    return list(G.predecessors(treatment))

def run_analysis(method, species_name, df_all):
    run_func = run_notears_l_and_plot_colored if method == "Linear" else run_notears_n_and_plot_colored
    print(f"\n🔧 Running {method} NOTEARS for {species_name}...")

    # NOTEARS
    fig_path, top_edges, explanations, dag = run_func(species_name)
    print(f"📌 DAG saved to: {fig_path}")
    print(f"🔗 Top 5 edges ({method}):\n" + "\n".join(f"  - {e}" for e in top_edges))
    print("\n🧠 Explanations:\n" + "\n".join(f"  - {line}" for line in explanations))

    # Candidate Treatment
    bios_combined = get_combined_bio_candidates(df_all, dag, top_k=10)
    results = []
    for treatment in bios_combined:
        try:
            common_causes = extract_common_causes(dag, treatment)
            model = CausalModel(
                data=df_all,
                treatment=treatment,
                outcome="Presence",
                common_causes=common_causes
            )
            
            try:
                estimate = model.estimate_effect(
                    model.identify_effect(),
                    method_name="econml.dml.ForestDML",
                    control_value=0,
                    treatment_value=1
                )
                print(f"  ✅ Used ForestDML for {treatment}")
            except Exception as e:
                print(f"  ⚠️ ForestDML failed for {treatment}: {e}")
                # fallback only if treatment is binary
                if df_all[treatment].nunique() != 2:
                    print(f"  ⚠️ Skipping {treatment} — not binary")
                    continue
                estimate = model.estimate_effect(
                    model.identify_effect(),
                    method_name="backdoor.propensity_score_matching",
                    control_value=0,
                    treatment_value=1
                )
                print(f"  🔁 Fallback to propensity_score_matching for {treatment}")
        
            results.append({
                "Treatment": treatment,
                "Effect": estimate.value,
                "CommonCauses": ", ".join(common_causes) if common_causes else "None",
                "Species": species_name,
                "Method": method
            })
            print(f"  ✅ {treatment} ({method}): effect = {estimate.value:.4f} | controls = {common_causes}")
        except Exception as e:
            print(f"  ❌ {treatment} ({method}): error - {str(e)}")
    
    df_result = pd.DataFrame(results)
    
    output_excel = f"outputs/{species_name.replace(' ', '_')}_{method}_results.xlsx"
    if not df_result.empty:
        df_result.to_excel(output_excel, index=False)
        print(f"📁 Saved {method} results to: {output_excel}")
    
    return df_result, top_edges, explanations, fig_path  # Return all 4 values to match the unpacking

def get_species_df(species_name, data_path="Data/Bee_Flower.xlsx"):
    df = pd.read_excel(data_path)
    df.columns = df.columns.str.replace('"', '').str.strip()
    
    if species_name not in df["Species"].unique():
        print(f"[Error] Species '{species_name}' not found.")
        return pd.DataFrame()
    
    df_species = df[df["Species"] == species_name].copy()
    bio_cols = [f"BIO{i}" for i in range(1, 20)]
    cols = bio_cols + ["Presence"]
    return df_species[cols].dropna().copy()

def run_dowhy(species_list_path="Species_Bee_Flower.xlsx", save_path="outputs/dowhy_results.xlsx"):
    species_df = pd.read_excel(species_list_path)
    species_list = species_df["Scientific Name"].dropna().unique()
    
    all_results = []
    for sp in species_list:
        print(f"\n📦 Running analysis for: {sp}")
        try:
            # Call run_analysis but only use the first return value (the DataFrame)
            df_species = get_species_df(sp)
            if not df_species.empty:
                linear_df, _, _, _ = run_analysis("Linear", sp, df_species)
                nonlinear_df, _, _, _ = run_analysis("Nonlinear", sp, df_species)
                
                # Add results if not empty
                if not linear_df.empty:
                    all_results.append(linear_df)
                if not nonlinear_df.empty:
                    all_results.append(nonlinear_df)
            else:
                print(f"⚠️ No data found for species {sp}")
        except Exception as e:
            print(f"❌ Failed for {sp}: {str(e)}")
        time.sleep(1)
    
    # Save combined results
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        # Ensure only the required columns are in the final output
        if "Method" in final_df.columns:
            final_df = final_df[["Treatment", "Effect", "CommonCauses", "Species"]]
        final_df.to_excel(save_path, index=False)
        print(f"\n✅ All results saved to {save_path}")
    else:
        print("\n⚠️ No results to save.")
        
    return final_df if all_results else None

if __name__ == "__main__":
    run_dowhy()