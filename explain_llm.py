"""
llm_explainer_local.py

This script loads a local large language model (LLM) such as Mistral or Mixtral using llama-cpp-python,
and generates natural language explanations for species-BIO causal effects estimated by DoWhy.
It processes an input Excel file (e.g., batch_results.xlsx) and outputs enriched explanations.

Ensure you have:
- Installed llama-cpp-python: pip install llama-cpp-python
- Downloaded a GGUF model (e.g., Mistral-7B-Instruct or Mixtral-8x7B-Instruct from TheBloke on HuggingFace)
"""

import pandas as pd
import ollama
import os

# Editable: path to your downloaded GGUF model (Q4 recommended)
MODEL_PATH = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"  # Change this to your downloaded file

# Human-friendly descriptions of BIO variables
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

def build_prompt(species, treatment, bio_name, effect, controls):
    """
    Construct a natural language prompt to pass to the local LLM for explanation generation.
    """
    direction = "positive" if effect > 0 else "negative"
    effect_str = f"{effect:+.3f}"
    ctrl_text = ", ".join(controls) if controls else "None"

    prompt = f"""
    You are an ecological scientist with expertise in species distribution modeling and climate-driven habitat analysis.

    Given the following causal inference result, explain in detail why this specific bioclimatic variable may causally influence the presence of the species.
    Base your explanation on ecological principles, including physiological tolerances, climatic constraints, seasonal dependencies, or habitat specialization.
    Ensure the explanation is realistic, grounded in ecological reasoning, and free from vague generalizations.

    Species: {species}
    BIO Variable: {treatment} ({bio_name})
    Estimated Causal Effect: {effect_str} ({direction})
    Controlled Variables: {ctrl_text}

    Write 3-5 sentences explaining the most likely ecological mechanism behind this causal link. Use precise ecological terminology and avoid speculative or non-scientific language.
    Explanation:

    """
    return prompt.strip()

class OllamaLLM:
    def __init__(self, model_name="deepseek-coder-v2"):
        self.model_name = model_name

    def __call__(self, prompt, max_tokens=200, stop=None):
        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_predict": max_tokens,
                "stop": stop
            } if max_tokens or stop else None
        )
        return response['message']['content']

def load_llm(model_name="deepseek-coder-v2"):
    """
    Load LLM using local Ollama backend.
    """
    return OllamaLLM(model_name)

def generate_explanations_local(input_path="outputs/dowhy_results.xlsx", output_dir="outputs/species_explanations_llm"):
    """
    Load causal effect results, pass them through the local LLM, and write explanations to individual text files.
    """
    model_name="deepseek-coder-v2"
    
    print("\n🔍 Loading model...")
    llm = load_llm()

    print("📄 Reading DoWhy results from:", input_path)
    df = pd.read_excel(input_path)

    os.makedirs(output_dir, exist_ok=True)

    for species in df["Species"].unique():
        species_df = df[df["Species"] == species].copy()
        lines = [f"\n✅ Species: {species}\n"]

        explanation_blocks = []  # For summary generation

        for _, row in species_df.iterrows():
            treatment = row["Treatment"]
            bio_name = BIO_DESC.get(treatment, treatment)
            effect = row["Effect"]
            controls = eval(row["CommonCauses"]) if isinstance(row["CommonCauses"], str) else row["CommonCauses"]

            prompt = build_prompt(species, treatment, bio_name, effect, controls)
            response = llm(prompt, max_tokens=300)
            response = response.strip()

            # Split Explanation and Summary if provided
            if "Summary:" in response:
                parts = response.split("Summary:")
                explanation = parts[0].strip()
                indiv_summary = parts[1].strip()
            else:
                explanation = response
                indiv_summary = ""

            explanation_text = (
                f"{treatment} — {bio_name}\n"
                f"Effect: {effect:+.4f} | Controls: {', '.join(controls) if controls else 'None'}\n"
                f"Explanation: {explanation}\n"
            )
            lines.append(explanation_text)
            explanation_blocks.append(f"{treatment} — {bio_name}: {explanation}")

        # Now generate overall summary based on all explanations
        summary_prompt = (
            f"You are an ecological scientist. Based on the following explanations for species '{species}', "
            f"write a concise summary (3–5 sentences) that synthesizes the main ecological insights.\n\n"
            + "\n".join(explanation_blocks) +
            "\n\nSummary:"
        )
        summary_response = llm(summary_prompt, max_tokens=250).strip()

        lines.append("\n📌 Overall Summary:\n" + summary_response + "\n")

        # Save explanation as text file
        file_name = f"{species.replace(' ', '_')}_{model_name}_explanation.txt"
        with open(os.path.join(output_dir, file_name), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"✅ Explanation written for: {species}")


if __name__ == "__main__":
    generate_explanations_local()
