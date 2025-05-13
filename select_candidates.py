def get_top_outdegree_nodes(G, top_k=5):
    """
    Return top_k nodes with highest out-degree from a DAG (excluding 'Presence' if present).
    """
    candidates = [node for node in G.nodes if node.startswith("BIO")]
    degrees = {node: G.out_degree(node) for node in candidates}
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    return [node for node, deg in sorted_nodes[:top_k]]


def get_top_correlated_bios(df, top_k=5):
    """
    Return top_k BIO variables most correlated (by abs value) with Presence.
    """
    bio_cols = [col for col in df.columns if col.startswith("BIO")]
    if "Presence" not in df.columns:
        raise ValueError("Presence column not found in dataframe.")
    corrs = df[bio_cols + ['Presence']].corr()['Presence'].drop('Presence')
    sorted_bios = corrs.abs().sort_values(ascending=False).head(top_k).index.tolist()
    return sorted_bios


def get_combined_bio_candidates(df, G, top_k=10):
    """
    Return high-confidence BIOs by intersecting top outdegree and top correlated variables.
    """
    outdeg_bios = set(get_top_outdegree_nodes(G, top_k))
    corr_bios = set(get_top_correlated_bios(df, top_k))
    return list(outdeg_bios & corr_bios)
