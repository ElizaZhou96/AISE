import pandas as pd
from dowhy import CausalModel
from notears_linear import run_notears_l_and_plot_colored 
from notears_nonlinear import run_notears_n_and_plot_colored 
import warnings
from select_candidates import get_combined_bio_candidates
import time
import torch

torch.set_default_dtype(torch.double)
warnings.filterwarnings("ignore")

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
            # try:
            #     estimate = model.estimate_effect(
            #         model.identify_effect(),
            #         method_name="econml.dml.ForestDML",
            #         control_value=0,
            #         treatment_value=1
            #     )
            # except ImportError:
            #     # fallback to simpler method
            #     estimate = model.estimate_effect(
            #         model.identify_effect(),
            #         method_name="backdoor.propensity_score_matching",
            #         control_value=0,
            #         treatment_value=1
            #     )
            
            try:
                estimate = model.estimate_effect(
                    model.identify_effect(),
                    method_name="econml.dml.ForestDML",
                    control_value=0,
                    treatment_value=1
                )
                print(f"  ✅ Used ForestDML for {treatment}")
            except Exception as e:  # unexpected
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
                "CommonCauses": ", ".join(common_causes) if common_causes else "None"
            })
            print(f"  ✅ {treatment} ({method}): effect = {estimate.value:.4f} | controls = {common_causes}")
        except Exception as e:
            print(f"  ❌ {treatment} ({method}): error - {str(e)}")

    
    df_result = pd.DataFrame(results)

    output_excel = f"outputs/{species_name.replace(' ', '_')}_{method}_results.xlsx"
    if not df_result.empty:
        df_result.to_excel(output_excel, index=False)
        print(f"📁 Saved {method} results to: {output_excel}")

    return df_result, top_edges, explanations, fig_path


def run_dowhy_analysis(species_name, data_path="Data/Bee_Flower.xlsx"):
    import os
    df = pd.read_excel(data_path)
    df.columns = df.columns.str.replace('"', '').str.strip()
    
    if species_name not in df["Species"].unique():
        print(f"[Error] Species '{species_name}' not found.")
        return

    df_species = df[df["Species"] == species_name].copy()
    bio_cols = [f"BIO{i}" for i in range(1, 20)]
    cols = bio_cols + ["Presence"]
    df_all = df_species[cols].dropna().copy()

    all_results = []
    output_lines = [] 

    output_lines.append(f"{'='*60}")
    output_lines.append(f"🔍 Causal Analysis for: {species_name}")
    output_lines.append(f"{'='*60}")

    for method in ["Linear", "Nonlinear"]:
        try:
            output_lines.append(f"\n📊 [{method.upper()} NOTEARS Results]")
            output_lines.append(f"{'-'*60}")

            df_result, top_edges, explanations, dag_path = run_analysis(method, species_name, df_all)
            all_results.append(df_result)

            if dag_path:
                output_lines.append(f"📌 DAG saved to: {dag_path}")
            if top_edges is not None and len(top_edges) > 0:
                output_lines.append("🔗 Top 5 edges:")
                output_lines.extend([f"  - {edge}" for edge in top_edges])
            if explanations:
                output_lines.append("\n🧠 Explanations:")
                output_lines.extend([f"  - {line}" for line in explanations])
            else:
                output_lines.append("⚠️  No significant edges or explanations.")
        except Exception as e:
            msg = f"❌ Error running {method} NOTEARS for {species_name}: {str(e)}"
            output_lines.append(msg)
            print(msg)

    # report
    report_text = "\n".join(output_lines)
    print(report_text)

    # save as txt
    os.makedirs("outputs", exist_ok=True)
    txt_path = f"outputs/{species_name.replace(' ', '_')}_causal_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n📝 Report saved to: {txt_path}")

    return pd.concat(all_results, ignore_index=True) if all_results else None


def run_dowhy(species_list_path="Species_Bee_Flower.xlsx", save_path="outputs/dowhy_results.xlsx"):
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

if __name__ == "__main__":
    run_dowhy()
