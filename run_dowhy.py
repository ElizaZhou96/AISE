import pandas as pd
from dowhy import CausalModel
from notears_linear import run_notears_l_and_plot_colored 
from notears_nonlinear import run_notears_n_and_plot_colored 
import warnings
from select_candidates import (
    get_combined_bio_candidates,
    get_top_outdegree_nodes,
    get_top_correlated_bios
)
import time

import torch
torch.set_default_dtype(torch.double)

warnings.filterwarnings("ignore")

def extract_common_causes(G, treatment):
    """
    Extract parent nodes of the treatment variable from the DAG.
    """
    if treatment not in G:
        raise ValueError(f"[Error] Treatment variable '{treatment}' not found in DAG.")
    return list(G.predecessors(treatment))  # incoming edges = direct causes

def run_dowhy_analysis(species_name, data_path="Data/Bee_Flower.xlsx"):

    df = pd.read_excel(data_path)
    df.columns = df.columns.str.replace('"', '').str.strip()

    if species_name not in df["Species"].unique():
        print(f"[Error] Species '{species_name}' not found.")
        return

    df_species = df[df["Species"] == species_name].copy()
    bio_cols = [f"BIO{i}" for i in range(1, 20)]
    cols = bio_cols + ["Presence"]
    df_all = df_species[cols].dropna().copy()

    method_results = {}

    for method_name, run_func in {
        "Linear": run_notears_l_and_plot_colored,
        "Nonlinear": run_notears_n_and_plot_colored
    }.items():
        print(f"\n🔧 Running {method_name} NOTEARS for {species_name}...")

        try:
            fig_path, top_edges, explanations, dag = run_func(species_name)

            print(f"📌 DAG saved to: {fig_path}")
            print(f"🔗 Top 5 edges ({method_name}):\n" + "\n".join(f"  - {e}" for e in top_edges))
            print("\n🧠 Explanations:\n" + "\n".join(f"  - {line}" for line in explanations))

            # Use DAG structure to select treatment variables
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
                    identified_model = model.identify_effect()
                    estimate = model.estimate_effect(identified_model, method_name="backdoor.random_forest")

                    results.append({
                        "Treatment": treatment,
                        "Effect": estimate.value,
                        "CommonCauses": common_causes
                    })
                    print(f"  ✅ {treatment} ({method_name}): effect = {estimate.value:.4f} | controls = {common_causes}")
                except Exception as e:
                    print(f"  ❌ {treatment} ({method_name}): error - {str(e)}")

            df_result = pd.DataFrame(results)
            df_result["Species"] = species_name
            df_result["Method"] = method_name
            method_results[method_name] = df_result

        except Exception as e:
            print(f"❌ Error running {method_name} NOTEARS for {species_name}: {str(e)}")

    if method_results:
        return pd.concat(method_results.values(), ignore_index=True)
    else:
        return None


def run_dowhy(species_list_path="Species_Bee_Flower.xlsx", save_path="outputs/dowhy_results.xlsx"):
    """
    Run run_dowhy_analysis() for all species listed in the Excel file under column 'Scientific Name'
    and save the result table to an output Excel file.
    """
    species_df = pd.read_excel(species_list_path)
    species_list = species_df["Scientific Name"].dropna().unique()

    all_results = []
    for sp in species_list:
        print(f"\n📦 Running analysis for: {sp}")
        try:
            result_df = run_dowhy_analysis(sp)
            if result_df is not None and not result_df.empty:
                result_df["Species"] = sp
                all_results.append(result_df)
        except Exception as e:
            print(f"❌ Failed for {sp}: {e}")
        time.sleep(1)  

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_excel(save_path, index=False)
        print(f"\n✅ All results saved to {save_path}")
    else:
        print("\n⚠️ No results to save.")

# if __name__ == "__main__":
#     species = "Osmia parietina"  # or "Ajuga reptans"
#     result_df = run_dowhy_analysis(species)
#     print("\n📊 Summary Table:")
#     print(result_df)

if __name__ == "__main__":
    run_dowhy()