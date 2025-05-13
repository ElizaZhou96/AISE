# Remove torch and rerun the function without importing it

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import matplotlib.patches as mpatches
from notears.linear import notears_linear

bio_type_colors = {
    "Temperature-related": "#EEA9A9",
    "Precipitation-related": "#A5DEE4",
    "Derived": "#B481BB"
}

def get_bio_type(bio_name):
    num = int(bio_name.replace("BIO", ""))
    if num in [1, 2, 5, 6, 8, 9, 10, 11]:
        return "Temperature-related"
    elif num in [12, 13, 14, 16, 17, 18, 19]:
        return "Precipitation-related"
    else:
        return "Derived"

def run_notears_l_and_plot_colored(target_species):
    df = pd.read_excel("Data/Bee_Flower.xlsx")
    df.columns = df.columns.str.replace('"', '').str.strip()

    if target_species not in df["Species"].unique():
        print(f"[Error] Species '{target_species}' not found in dataset!")
        return

    df_sub = df[df["Species"] == target_species].copy()
    bio_cols = [f'BIO{i}' for i in range(1, 20)]
    df_presence_only = df_sub[df_sub["Presence"] == 1].copy()
    df_clean = df_presence_only[bio_cols + ['Presence']].dropna()
    df_clean = df_clean.loc[:, df_clean.std() > 0]

    if df_clean.shape[0] < 50:
        print(f"[Warning] Only {df_clean.shape[0]} samples after cleaning — DAG may be unstable.")

    X = df_clean.to_numpy()
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    var_names = df_clean.columns.tolist()

    W_est = notears_linear(X, lambda1=0.1, loss_type='l2')

    G = nx.DiGraph()
    edge_list = []
    for i in range(len(var_names)):
        for j in range(len(var_names)):
            weight = W_est[i, j]
            if abs(weight) > 0.01:
                G.add_edge(var_names[j], var_names[i], weight=weight)
                edge_list.append((var_names[j], var_names[i], weight))

    edge_list_sorted = sorted(edge_list, key=lambda x: abs(x[2]), reverse=True)[:5]
    top_edges = [f"{src} → {tgt} (weight = {w:.2f})" for src, tgt, w in edge_list_sorted]
    explanations = [
        f"{src} {'positively' if w > 0 else 'negatively'} influences {tgt} with a strength of {abs(w):.2f}."
        for src, tgt, w in edge_list_sorted
    ]

    var_names_sorted = sorted(var_names, key=lambda x: int(x.replace("BIO", ""))) 
    pos = nx.shell_layout(G, nlist=[var_names_sorted])
    pos = nx.rescale_layout_dict(pos, scale=3)

    node_colors = [bio_type_colors[get_bio_type(var)] for var in var_names_sorted]

    plt.figure(figsize=(13, 11))
    nx.draw(G, pos, with_labels=True, node_size=2000,
            node_color=node_colors, font_size=10, arrowsize=20)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {k: f'{v:.2f}' for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(f"Causal DAG for {target_species} (NOTEARS)")

    legend_handles = [
        mpatches.Patch(color=color, label=label.capitalize())
        for label, color in bio_type_colors.items()
    ]
    plt.legend(handles=legend_handles, title="BIO Variable Types", loc="lower right",
               fontsize=9, title_fontsize=10)

    explanation_text = "\n".join([
        "Top 5 strongest causal edges:"] +
        [f"  - {line}" for line in top_edges] +
        ["", "Natural language explanation:"] +
        [f"  - {line}" for line in explanations]
    )
    plt.gcf().text(0.01, 0.01, explanation_text, fontsize=9, va='bottom', ha='left', wrap=True)

    os.makedirs("outputs", exist_ok=True)
    fig_path = f"outputs/{target_species.replace(' ', '_')}_L_DAG_colored.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    return fig_path, top_edges, explanations, G

run_notears_l_and_plot_colored("Osmia parietina") #Osmia parietina & Ajuga reptans
