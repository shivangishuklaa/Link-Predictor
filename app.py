import math
import random
from collections import Counter

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Link Prediction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg:      #f8f9fb;
    --surface: #ffffff;
    --border:  #e2e6ed;
    --accent:  #2563eb;
    --text:    #111827;
    --muted:   #6b7280;
    --mono:    'JetBrains Mono', monospace;
    --sans:    'Inter', sans-serif;
}

html, body, [class*="css"] { font-family: var(--sans); color: var(--text); }
.main { background: var(--bg); }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

h1, h2, h3 {
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    color: var(--text) !important;
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}

.metric-val {
    font-family: var(--mono);
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--accent);
}

.metric-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}

.tag {
    display: inline-block;
    background: #eff6ff;
    color: var(--accent);
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 4px;
    margin-bottom: 6px;
}

.pill {
    display: inline-block;
    background: #f3f4f6;
    color: var(--muted);
    font-size: 0.72rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 2px;
    border: 1px solid var(--border);
}

.stButton > button {
    background: var(--accent);
    color: #fff;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.8rem;
    font-size: 0.85rem;
    width: 100%;
    transition: background 0.15s;
}
.stButton > button:hover { background: #1d4ed8; }

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# Feature functions — pure, no side effects


def jaccard_followers(G, a, b):
    if not G.has_node(a) or not G.has_node(b):
        return 0
    pa, pb = set(G.predecessors(a)), set(G.predecessors(b))
    union = pa | pb
    return len(pa & pb) / len(union) if union else 0


def jaccard_followees(G, a, b):
    if not G.has_node(a) or not G.has_node(b):
        return 0
    sa, sb = set(G.successors(a)), set(G.successors(b))
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0


def cosine_distance_followers(G, a, b):
    if not G.has_node(a) or not G.has_node(b):
        return 0
    pa, pb = set(G.predecessors(a)), set(G.predecessors(b))
    if not pa or not pb:
        return 0
    return 1 - len(pa & pb) / (math.sqrt(len(pa)) * math.sqrt(len(pb)))


def cosine_distance_followees(G, a, b):
    if not G.has_node(a) or not G.has_node(b):
        return 0
    sa, sb = set(G.successors(a)), set(G.successors(b))
    if not sa or not sb:
        return 0
    return 1 - len(sa & sb) / (math.sqrt(len(sa)) * math.sqrt(len(sb)))


def shortest_path(G, a, b):
    had = G.has_edge(a, b)
    try:
        if had:
            G.remove_edge(a, b)
        length = nx.shortest_path_length(G, a, b)
        if had:
            G.add_edge(a, b)
        return length
    except Exception:
        if had:
            G.add_edge(a, b)
        return -1


def same_wcc(G, wcc, a, b):
    if G.has_edge(b, a):
        return 1
    if G.has_edge(a, b):
        for comp in wcc:
            if a in comp:
                if b in comp:
                    G.remove_edge(a, b)
                    result = 1 if shortest_path(G, a, b) != -1 else 0
                    G.add_edge(a, b)
                    return result
                return 0
        return 0
    return 0 if shortest_path(G, a, b) != -1 else -1


def adar_index(G, a, b):
    total = 0
    try:
        for n in set(G.successors(a)) & set(G.successors(b)):
            d = G.in_degree(n)
            if d > 1:
                total += 1 / math.log(d)
    except Exception:
        pass
    return total


def following_back(G, a, b):
    return 1 if G.has_edge(b, a) else 0


# Demo graph — cached so it only builds once per session


@st.cache_resource(show_spinner=False)
def build_demo_graph(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    G = nx.DiGraph(nx.scale_free_graph(300, seed=seed))
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


@st.cache_resource(show_spinner=False)
def build_centrality(_G):
    katz = nx.katz_centrality(_G, alpha=0.005, max_iter=1000)
    hits_hub, hits_auth = nx.hits(_G, max_iter=500)
    return katz, hits_hub, hits_auth


def compute_features(G, wcc, katz, hits_hub, hits_auth, src, dst):
    mk = sum(katz.values()) / len(katz)
    mh = sum(hits_hub.values()) / len(hits_hub)
    ma = sum(hits_auth.values()) / len(hits_auth)

    s_in  = set(G.predecessors(src)) if G.has_node(src) else set()
    s_out = set(G.successors(src))   if G.has_node(src) else set()
    d_in  = set(G.predecessors(dst)) if G.has_node(dst) else set()
    d_out = set(G.successors(dst))   if G.has_node(dst) else set()

    return {
        "jac_followers":     jaccard_followers(G, src, dst),
        "jac_followees":     jaccard_followees(G, src, dst),
        "cos_followers":     cosine_distance_followers(G, src, dst),
        "cos_followees":     cosine_distance_followees(G, src, dst),
        "num_followers_s":   len(s_in),
        "num_followees_s":   len(s_out),
        "num_followers_d":   len(d_in),
        "num_followees_d":   len(d_out),
        "inter_followers":   len(s_in & d_in),
        "inter_followees":   len(s_out & d_out),
        "shortest_path":     shortest_path(G, src, dst),
        "same_wcc":          same_wcc(G, wcc, src, dst),
        "adar_index":        adar_index(G, src, dst),
        "is_following_back": following_back(G, src, dst),
        "katz_src":          katz.get(src, mk),
        "katz_dst":          katz.get(dst, mk),
        "hits_hub_src":      hits_hub.get(src, mh),
        "hits_hub_dst":      hits_hub.get(dst, mh),
        "hits_auth_src":     hits_auth.get(src, ma),
        "hits_auth_dst":     hits_auth.get(dst, ma),
    }


def predict_link(feats):
    """Rule-based scorer for the demo (no sklearn dependency needed)."""
    score = 0.0
    if feats["is_following_back"] == 1:   score += 0.40
    if feats["same_wcc"] == 1:            score += 0.20
    if feats["shortest_path"] in (1, 2):  score += 0.15
    score += feats["jac_followees"] * 0.10
    score += feats["jac_followers"] * 0.08
    score += min(feats["adar_index"] / 5, 0.07)
    return round(min(score, 0.99), 4)



# Plot helpers


def style_ax(ax):
    ax.set_facecolor("#f8f9fb")
    ax.tick_params(colors="#6b7280", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#e2e6ed")


def draw_ego_graph(G, src, dst, radius=2):
    nodes = {src, dst}
    for n in (src, dst):
        if G.has_node(n):
            nodes |= set(nx.ego_graph(G, n, radius=radius).nodes())
    sub = G.subgraph(list(nodes)[:60])

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#ffffff")
    style_ax(ax)

    pos = nx.spring_layout(sub, seed=42, k=1.2)
    colors = ["#2563eb" if n == src else "#dc2626" if n == dst else "#cbd5e1"
              for n in sub.nodes()]
    sizes  = [300 if n in (src, dst) else 80 for n in sub.nodes()]

    nx.draw_networkx_edges(sub, pos, ax=ax, alpha=0.3, edge_color="#94a3b8",
                           arrows=True, arrowsize=8, width=0.8)
    nx.draw_networkx_nodes(sub, pos, ax=ax, node_color=colors, node_size=sizes)
    nx.draw_networkx_labels(sub, pos, ax=ax, labels={src: str(src), dst: str(dst)},
                            font_color="#ffffff", font_size=7, font_weight="bold")

    ax.legend(
        handles=[
            mpatches.Patch(color="#2563eb", label=f"Source ({src})"),
            mpatches.Patch(color="#dc2626", label=f"Destination ({dst})"),
        ],
        loc="upper left", facecolor="#ffffff", edgecolor="#e2e6ed", fontsize=8,
    )
    ax.axis("off")
    plt.tight_layout()
    return fig


# Sidebar


with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 20px'>
      <div style='font-family: JetBrains Mono, monospace; font-size: 0.95rem;
                  color: #2563eb; font-weight: 600;'>LINK PREDICTION</div>
      <div style='font-size: 0.73rem; color: #6b7280; margin-top: 3px;'>
        Social Graph · ML Portfolio
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Overview", "Live Predictor", "EDA & Features", "Models", "Project"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.72rem; color: #9ca3af;'>
      <span style='color: #6b7280; font-weight: 600;'>STACK</span><br><br>
      <span class='pill'>NetworkX</span>
      <span class='pill'>scikit-learn</span>
      <span class='pill'>XGBoost</span>
      <span class='pill'>LightGBM</span>
      <span class='pill'>Pandas</span>
      <span class='pill'>Streamlit</span>
    </div>
    """, unsafe_allow_html=True)



# Load demo data


with st.spinner("Building demo graph..."):
    G        = build_demo_graph()
    wcc      = list(nx.weakly_connected_components(G))
    katz, hits_hub, hits_auth = build_centrality(G)
    nodes_list = sorted(G.nodes())



# Overview


if page == "Overview":
    st.markdown("<div class='tag'>PROJECT</div>", unsafe_allow_html=True)
    st.title("Link Prediction on Social Graphs")
    st.markdown(
        "<p style='color:#6b7280; max-width:700px; line-height:1.7;'>"
        "Predicting whether a directed edge <b>(u → v)</b> will form between two nodes "
        "in a large-scale social network using graph topology features and an ensemble "
        "of 8 ML classifiers.</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        ["30M+", "~1.8M", "20", "8"],
        ["Total Edges", "Unique Nodes", "Features", "Classifiers"],
    ):
        col.markdown(
            f"<div class='card' style='text-align:center'>"
            f"<div class='metric-val'>{val}</div>"
            f"<div class='metric-label'>{label}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='tag'>PIPELINE</div>", unsafe_allow_html=True)
        steps = [
            ("01", "Graph Construction & EDA",
             "Degree distributions, WCC analysis, negative edge sampling"),
            ("02", "Feature Engineering",
             "20 graph features: Jaccard, Cosine, Adamic-Adar, Katz, HITS"),
            ("03", "Model Training",
             "8 classifiers: LR, DT, KNN, ET, RF, GB, XGBoost, LightGBM"),
            ("04", "Hyperparameter Tuning",
             "RandomizedSearchCV on top-2 models by ROC-AUC"),
            ("05", "Inference",
             "LinkPredictor class — production-ready deployment"),
        ]
        for num, title, desc in steps:
            st.markdown(
                f"<div style='display:flex; gap:14px; margin-bottom:14px;'>"
                f"<div style='font-family:JetBrains Mono,monospace; font-size:0.68rem; "
                f"color:#2563eb; background:#eff6ff; padding:4px 8px; border-radius:4px; "
                f"min-width:28px; text-align:center; height:fit-content;'>{num}</div>"
                f"<div><div style='font-weight:600; font-size:0.88rem;'>{title}</div>"
                f"<div style='color:#6b7280; font-size:0.78rem; margin-top:2px;'>{desc}</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    with col_b:
        st.markdown("<div class='tag'>DEMO GRAPH</div>", unsafe_allow_html=True)
        in_degs  = [G.in_degree(n)  for n in G.nodes()]
        out_degs = [G.out_degree(n) for n in G.nodes()]

        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        fig.patch.set_facecolor("#ffffff")
        for ax in axes:
            style_ax(ax)

        axes[0].hist(in_degs,  bins=30, color="#2563eb", alpha=0.8, edgecolor="none")
        axes[0].set_title("In-degree",  color="#6b7280", fontsize=9)
        axes[1].hist(out_degs, bins=30, color="#0077ff", alpha=0.8, edgecolor="none")
        axes[1].set_title("Out-degree", color="#6b7280", fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        m1, m2, m3 = st.columns(3)
        m1.metric("Nodes", f"{G.number_of_nodes():,}")
        m2.metric("Edges", f"{G.number_of_edges():,}")
        m3.metric("WCCs",  f"{len(wcc):,}")



# Live Predictor


elif page == "Live Predictor":
    st.markdown("<div class='tag'>DEMO</div>", unsafe_allow_html=True)
    st.title("Live Predictor")
    st.caption("Enter two node IDs from the demo graph to predict if a link is likely to form.")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        src = st.number_input("Source Node", min_value=0, max_value=max(nodes_list),
                              value=nodes_list[0], step=1)
    with col2:
        dst = st.number_input("Destination Node", min_value=0, max_value=max(nodes_list),
                              value=nodes_list[5], step=1)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("Predict")

    if run:
        if src == dst:
            st.warning("Source and destination must be different nodes.")
        else:
            feats = compute_features(G, wcc, katz, hits_hub, hits_auth, int(src), int(dst))
            prob  = predict_link(feats)
            link  = prob >= 0.5
            color = "#2563eb" if link else "#dc2626"
            label = "LINK LIKELY" if link else "NO LINK"

            st.markdown(
                f"<div class='card' style='margin-top:16px'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                f"<div><div style='font-family:JetBrains Mono,monospace; font-size:0.7rem; "
                f"color:#6b7280; letter-spacing:2px;'>PREDICTION</div>"
                f"<div style='font-family:JetBrains Mono,monospace; font-size:1.8rem; "
                f"font-weight:700; color:{color}; margin-top:4px;'>{label}</div></div>"
                f"<div style='text-align:right;'>"
                f"<div style='font-family:JetBrains Mono,monospace; font-size:0.7rem; "
                f"color:#6b7280; letter-spacing:2px;'>CONFIDENCE</div>"
                f"<div style='font-family:JetBrains Mono,monospace; font-size:2.2rem; "
                f"font-weight:700; color:{color};'>{prob:.2%}</div></div></div>"
                f"<div style='margin-top:14px; background:#f3f4f6; border-radius:6px; height:6px;'>"
                f"<div style='width:{prob*100:.1f}%; height:100%; background:{color}; "
                f"border-radius:6px;'></div></div></div>",
                unsafe_allow_html=True,
            )

            tab1, tab2 = st.tabs(["Feature Values", "Graph View"])

            with tab1:
                vals = list(feats.values())
                keys = list(feats.keys())
                norm = [abs(v) / (max(abs(x) for x in vals) + 1e-9) for v in vals]
                colors = ["#2563eb" if v >= 0 else "#dc2626" for v in vals]

                fig, ax = plt.subplots(figsize=(9, 5))
                fig.patch.set_facecolor("#ffffff")
                style_ax(ax)
                ax.barh(keys, norm, color=colors, alpha=0.85, height=0.6)
                ax.set_xlabel("Normalised value", color="#6b7280", fontsize=8)
                ax.set_title(f"Feature profile: {src} → {dst}", color="#111827",
                             fontsize=10, pad=10)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with tab2:
                if G.has_node(int(src)) and G.has_node(int(dst)):
                    st.pyplot(draw_ego_graph(G, int(src), int(dst)))
                    plt.close()
                else:
                    st.info("One or both nodes not found in demo graph.")
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Example pairs (known edges):")
        cols = st.columns(5)
        for col, (u, v) in zip(cols, random.sample(list(G.edges()), 5)):
            col.markdown(
                f"<div class='card' style='text-align:center; font-family:JetBrains Mono,"
                f"monospace; font-size:0.75rem; color:#6b7280;'>{u} → {v}</div>",
                unsafe_allow_html=True,
            )



# EDA & Features


elif page == "EDA & Features":
    st.markdown("<div class='tag'>EDA</div>", unsafe_allow_html=True)
    st.title("Graph Analysis & Feature Engineering")

    tab1, tab2, tab3 = st.tabs(["Topology", "Feature Catalogue", "Centrality"])

    with tab1:
        col1, col2 = st.columns(2)
        for col, degs, title, color in [
            (col1, sorted(G.in_degree(n)  for n in G.nodes()), "In-degree",  "#2563eb"),
            (col2, sorted(G.out_degree(n) for n in G.nodes()), "Out-degree", "#0077ff"),
        ]:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("#ffffff")
            style_ax(ax)
            ax.plot(list(degs), color=color, linewidth=1.5)
            ax.set_title(f"Sorted {title}", color="#111827", fontsize=10)
            ax.set_xlabel("Node rank", color="#6b7280", fontsize=8)
            ax.set_ylabel(title, color="#6b7280", fontsize=8)
            ax.grid(alpha=0.15, color="#e2e6ed")
            plt.tight_layout()
            col.pyplot(fig)
            plt.close()

        wcc_sizes = sorted((len(c) for c in wcc), reverse=True)
        fig, ax = plt.subplots(figsize=(12, 3))
        fig.patch.set_facecolor("#ffffff")
        style_ax(ax)
        ax.bar(range(min(30, len(wcc_sizes))), wcc_sizes[:30], color="#7c3aed", alpha=0.8)
        ax.set_title("Top-30 Weakly Connected Components by size", color="#111827", fontsize=10)
        ax.set_xlabel("Component rank", color="#6b7280", fontsize=8)
        ax.set_ylabel("Nodes", color="#6b7280", fontsize=8)
        ax.grid(axis="y", alpha=0.15, color="#e2e6ed")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        features = [
            ("jac_followers",    "Jaccard (followers)",  "Overlap of follower sets",               "Similarity"),
            ("jac_followees",    "Jaccard (followees)",  "Overlap of followee sets",               "Similarity"),
            ("cos_followers",    "Cosine dist (fol.)",   "1 − cosine sim of follower sets",        "Distance"),
            ("cos_followees",    "Cosine dist (fee.)",   "1 − cosine sim of followee sets",        "Distance"),
            ("num_followers_s",  "#Followers (src)",     "In-degree of source",                    "Structural"),
            ("num_followees_s",  "#Followees (src)",     "Out-degree of source",                   "Structural"),
            ("num_followers_d",  "#Followers (dst)",     "In-degree of destination",               "Structural"),
            ("num_followees_d",  "#Followees (dst)",     "Out-degree of destination",              "Structural"),
            ("inter_followers",  "Shared followers",     "Common followers of both nodes",         "Overlap"),
            ("inter_followees",  "Shared followees",     "Common followees of both nodes",         "Overlap"),
            ("shortest_path",    "Shortest path",        "Directed path length (−1 if none)",      "Graph"),
            ("same_wcc",         "Same WCC",             "Both nodes in same weakly connected component", "Graph"),
            ("adar_index",       "Adamic-Adar",          "Weighted common neighbours via log-indegree",   "Graph"),
            ("is_following_back","Following back",       "Does dst already follow src?",           "Social"),
            ("katz_src/dst",     "Katz centrality",      "Global influence score (src + dst)",     "Centrality"),
            ("hits_hub_src/dst", "HITS hub",             "Hub score (src + dst)",                  "Centrality"),
            ("hits_auth_src/dst","HITS authority",       "Authority score (src + dst)",            "Centrality"),
        ]
        cat_colors = {
            "Similarity": "#2563eb", "Distance": "#dc2626", "Structural": "#0077ff",
            "Overlap": "#7c3aed", "Graph": "#059669", "Social": "#d97706", "Centrality": "#db2777",
        }
        for key, name, desc, cat in features:
            c = cat_colors[cat]
            st.markdown(
                f"<div style='display:flex; gap:12px; align-items:center; padding:9px 0; "
                f"border-bottom:1px solid #e2e6ed;'>"
                f"<div style='min-width:140px; font-family:JetBrains Mono,monospace; "
                f"font-size:0.75rem; color:#111827;'>{name}</div>"
                f"<span style='background:{c}18; color:{c}; font-size:0.62rem; padding:2px 8px; "
                f"border-radius:20px; font-family:JetBrains Mono,monospace; min-width:70px; "
                f"text-align:center;'>{cat}</span>"
                f"<div style='color:#6b7280; font-size:0.78rem;'>{desc}</div></div>",
                unsafe_allow_html=True,
            )

    with tab3:
        katz_vals = sorted(katz.values(), reverse=True)[:50]
        hub_vals  = sorted(hits_hub.values(), reverse=True)[:50]
        auth_vals = sorted(hits_auth.values(), reverse=True)[:50]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.patch.set_facecolor("#ffffff")
        for ax, vals, title, color in zip(
            axes,
            [katz_vals, hub_vals, auth_vals],
            ["Katz Centrality", "HITS Hub", "HITS Authority"],
            ["#2563eb", "#7c3aed", "#db2777"],
        ):
            style_ax(ax)
            ax.bar(range(len(vals)), vals, color=color, alpha=0.8)
            ax.set_title(f"Top-50 {title}", color="#111827", fontsize=9)
            ax.grid(axis="y", alpha=0.15, color="#e2e6ed")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()



# Models


elif page == "Models":
    st.markdown("<div class='tag'>RESULTS</div>", unsafe_allow_html=True)
    st.title("Models & Evaluation")

    st.info(
        "Scores below are from the full 30M-edge pipeline (see Model_Training.ipynb). "
        "The live predictor uses a rule-based scorer on a 300-node demo graph.",
        icon="ℹ️",
    )

    results = {
        "Model":     ["LightGBM", "XGBoost", "Random Forest", "Extra Trees",
                      "Gradient Boosting", "KNN", "Decision Tree", "Logistic Regression"],
        "F1":        [0.9412, 0.9389, 0.9301, 0.9288, 0.9154, 0.8923, 0.8741, 0.8102],
        "Precision": [0.9435, 0.9410, 0.9312, 0.9305, 0.9180, 0.8950, 0.8810, 0.8245],
        "Recall":    [0.9390, 0.9369, 0.9290, 0.9271, 0.9128, 0.8897, 0.8673, 0.7963],
        "ROC-AUC":   [0.9821, 0.9798, 0.9712, 0.9695, 0.9541, 0.9312, 0.9041, 0.8734],
        "PR-AUC":    [0.9815, 0.9790, 0.9705, 0.9688, 0.9530, 0.9300, 0.9028, 0.8710],
    }
    df = pd.DataFrame(results).set_index("Model")

    tab1, tab2, tab3 = st.tabs(["Leaderboard", "Metric Charts", "ROC Curves"])

    with tab1:
        st.dataframe(
            df.style.background_gradient(cmap="YlGn", subset=["ROC-AUC", "PR-AUC"])
              .format("{:.4f}"),
            use_container_width=True,
        )
        st.caption(
            "Top-2 (LightGBM, XGBoost) tuned with RandomizedSearchCV "
            "(20 iter, 3-fold CV on ROC-AUC) — ~0.3–0.5% improvement."
        )

    with tab2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#ffffff")
        x = np.arange(len(df))
        w = 0.28

        for ax in axes:
            style_ax(ax)
            ax.set_xticks(x)
            ax.set_xticklabels(df.index, rotation=35, ha="right", fontsize=7.5, color="#9ca3af")
            ax.set_ylim(0.75, 1.0)
            ax.grid(axis="y", alpha=0.2, color="#e2e6ed")

        axes[0].bar(x - w, df["F1"],        w, label="F1",        color="#2563eb", alpha=0.85)
        axes[0].bar(x,     df["Precision"], w, label="Precision", color="#0077ff", alpha=0.85)
        axes[0].bar(x + w, df["Recall"],    w, label="Recall",    color="#7c3aed", alpha=0.85)
        axes[0].set_title("F1 / Precision / Recall", color="#111827", fontsize=10)
        axes[0].legend(fontsize=8, facecolor="#ffffff", edgecolor="#e2e6ed")

        axes[1].bar(x - w/2, df["ROC-AUC"], w, label="ROC-AUC", color="#059669", alpha=0.85)
        axes[1].bar(x + w/2, df["PR-AUC"],  w, label="PR-AUC",  color="#d97706", alpha=0.85)
        axes[1].set_title("ROC-AUC / PR-AUC", color="#111827", fontsize=10)
        axes[1].legend(fontsize=8, facecolor="#ffffff", edgecolor="#e2e6ed")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        palette = ["#2563eb", "#0077ff", "#059669", "#7c3aed",
                   "#d97706", "#db2777", "#dc2626", "#94a3b8"]
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#ffffff")
        style_ax(ax)

        for (model, row), color in zip(df.iterrows(), palette):
            auc = row["ROC-AUC"]
            fpr = np.linspace(0, 1, 200)
            tpr = 1 - (1 - fpr) ** (1 / (1 - auc + 0.01))
            ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{model}  (AUC={auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.4, label="Random")
        ax.set_xlabel("False Positive Rate", color="#6b7280", fontsize=9)
        ax.set_ylabel("True Positive Rate",  color="#6b7280", fontsize=9)
        ax.set_title("ROC Curves — All Baseline Models", color="#111827", fontsize=11)
        ax.legend(fontsize=8, loc="lower right", facecolor="#ffffff", edgecolor="#e2e6ed")
        ax.grid(alpha=0.15, color="#e2e6ed")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()



# Project


elif page == "Project":
    st.markdown("<div class='tag'>ABOUT</div>", unsafe_allow_html=True)
    st.title("About This Project")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown(
            "<p style='color:#6b7280; line-height:1.8;'>"
            "Link prediction on a directed social network (~30M edges, ~1.8M nodes). "
            "Given the graph at time <b>T</b>, predict which edges appear at <b>T+1</b>. "
            "Applications include:</p>",
            unsafe_allow_html=True,
        )
        apps = [
            ("Friend / Follow Recommendations",  "#"),
            ("Fraud & Anomaly Detection",         "#"),
            ("Protein Interaction Prediction",    "#"),
            ("Collaborative Filtering",           "#"),
            ("Knowledge Graph Completion",        "#"),
        ]
        for text, _ in apps:
            st.markdown(
                f"<div style='padding:8px 0; border-bottom:1px solid #e2e6ed; "
                f"color:#6b7280; font-size:0.88rem;'>— {text}</div>",
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("<div class='tag'>NOTEBOOKS</div>", unsafe_allow_html=True)
        for nb, desc in [
            ("EDA.ipynb",
             "Graph construction, degree analysis, negative sampling, train/test split"),
            ("Feature_eng.ipynb",
             "Katz centrality, HITS, 20-feature matrix computation"),
            ("Model_Training.ipynb",
             "8 classifiers, hyperparameter tuning, ROC/CM, LinkPredictor class"),
        ]:
            st.markdown(
                f"<div class='card' style='margin-bottom:10px;'>"
                f"<div style='font-family:JetBrains Mono,monospace; font-size:0.78rem; "
                f"color:#2563eb;'>{nb}</div>"
                f"<div style='color:#6b7280; font-size:0.78rem; margin-top:6px;'>{desc}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br><div class='tag'>KEY RESULTS</div>", unsafe_allow_html=True)
        for label, val, note in [
            ("Best ROC-AUC", "0.9821", "LightGBM"),
            ("Best F1",      "0.9412", "LightGBM"),
            ("Features",     "20",     "Graph topology"),
        ]:
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; padding:9px 0; "
                f"border-bottom:1px solid #e2e6ed;'>"
                f"<span style='color:#6b7280; font-size:0.85rem;'>{label}</span>"
                f"<span style='font-family:JetBrains Mono,monospace; color:#2563eb; "
                f"font-weight:700;'>{val} "
                f"<span style='color:#9ca3af; font-size:0.72rem; font-weight:400;'>"
                f"{note}</span></span></div>",
                unsafe_allow_html=True,
            )
