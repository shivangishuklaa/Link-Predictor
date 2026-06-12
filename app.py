import streamlit as st
import joblib
import pickle
import os
import math
import numpy as np
import networkx as nx

# ── LinkPredictor class (must be defined before loading predictor.pkl) ─────────
class LinkPredictor:
    RANK_COLS = ['katz_src', 'katz_dst',
                 'hits_hub_src', 'hits_hub_dst',
                 'hits_auth_src', 'hits_auth_dst']

    def __init__(self, model, graph, wcc, katz, hits_hub, hits_auth,
                 mean_katz, mean_hub, mean_auth, feature_cols, rank_bounds):
        self.model        = model
        self.graph        = graph
        self.wcc          = wcc
        self.katz         = katz
        self.hits_hub     = hits_hub
        self.hits_auth    = hits_auth
        self.mean_katz    = mean_katz
        self.mean_hub     = mean_hub
        self.mean_auth    = mean_auth
        self.feature_cols = feature_cols
        self.rank_bounds  = rank_bounds

    def _rank_normalize(self, col, value):
        mn, mx = self.rank_bounds[col]
        span = mx - mn if mx != mn else 1.0
        return float(np.clip((value - mn) / span, 0, 1))

    def _jaccard_followers(self, a, b):
        try:
            if not self.graph.has_node(a) or not self.graph.has_node(b): return 0
            pa, pb = set(self.graph.predecessors(a)), set(self.graph.predecessors(b))
            union = pa | pb
            return len(pa & pb) / len(union) if union else 0
        except: return 0

    def _jaccard_followees(self, a, b):
        try:
            if not self.graph.has_node(a) or not self.graph.has_node(b): return 0
            sa, sb = set(self.graph.successors(a)), set(self.graph.successors(b))
            union = sa | sb
            return len(sa & sb) / len(union) if union else 0
        except: return 0

    def _cosine_followers(self, a, b):
        try:
            if not self.graph.has_node(a) or not self.graph.has_node(b): return 0
            pa, pb = set(self.graph.predecessors(a)), set(self.graph.predecessors(b))
            if not pa or not pb: return 0
            return len(pa & pb) / (math.sqrt(len(pa)) * math.sqrt(len(pb)))
        except: return 0

    def _cosine_followees(self, a, b):
        try:
            if not self.graph.has_node(a) or not self.graph.has_node(b): return 0
            sa, sb = set(self.graph.successors(a)), set(self.graph.successors(b))
            if not sa or not sb: return 0
            return len(sa & sb) / (math.sqrt(len(sa)) * math.sqrt(len(sb)))
        except: return 0

    def _shortest_path(self, a, b):
        edge_existed = self.graph.has_edge(a, b)
        try:
            if edge_existed: self.graph.remove_edge(a, b)
            length = nx.shortest_path_length(self.graph, source=a, target=b)
            if edge_existed: self.graph.add_edge(a, b)
            return length
        except:
            if edge_existed: self.graph.add_edge(a, b)
            return -1

    def _same_wcc(self, a, b):
        """Check if nodes a and b are in the same weakly connected component."""
        try:
            return 1 if self._shortest_path(a, b) != -1 else 0
        except: return 0

    def _adar(self, a, b):
        total = 0
        try:
            common = set(self.graph.successors(a)) & set(self.graph.successors(b))
            for node in common:
                in_deg = self.graph.in_degree(node)
                if in_deg > 1:
                    total += 1 / math.log(in_deg)
        except: pass
        return total

    def predict(self, source_node, destination_node, threshold=0.5):
        src, dst = source_node, destination_node
        src_known = self.graph.has_node(src)
        dst_known = self.graph.has_node(dst)

        warning = None
        if not src_known and not dst_known:
            warning = f'Both nodes {src} and {dst} are unseen — features will be zero'
        elif not src_known:
            warning = f'Source node {src} not in training graph'
        elif not dst_known:
            warning = f'Destination node {dst} not in training graph'

        s_in  = set(self.graph.predecessors(src)) if src_known else set()
        s_out = set(self.graph.successors(src))   if src_known else set()
        d_in  = set(self.graph.predecessors(dst)) if dst_known else set()
        d_out = set(self.graph.successors(dst))   if dst_known else set()

        feat_dict = {
            'jac_followers':   self._jaccard_followers(src, dst),
            'jac_followees':   self._jaccard_followees(src, dst),
            'cos_followers':   self._cosine_followers(src, dst),
            'cos_followees':   self._cosine_followees(src, dst),
            'num_followers_s': len(s_in),
            'num_followees_s': len(s_out),
            'num_followers_d': len(d_in),
            'num_followees_d': len(d_out),
            'inter_followers': len(s_in & d_in),
            'inter_followees': len(s_out & d_out),
            'shortest_path':   self._shortest_path(src, dst),
            'same_wcc':        self._same_wcc(src, dst),
            'adar_index':      self._adar(src, dst),
            'katz_src':        self.katz.get(src, self.mean_katz),
            'katz_dst':        self.katz.get(dst, self.mean_katz),
            'hits_hub_src':    self.hits_hub.get(src, self.mean_hub),
            'hits_hub_dst':    self.hits_hub.get(dst, self.mean_hub),
            'hits_auth_src':   self.hits_auth.get(src, self.mean_auth),
            'hits_auth_dst':   self.hits_auth.get(dst, self.mean_auth),
        }

        for col in self.RANK_COLS:
            feat_dict[col] = self._rank_normalize(col, feat_dict[col])

        features = [feat_dict[col] for col in self.feature_cols]
        prob = self.model.predict_proba([features])[0][1]
        pred = int(prob >= threshold)
        return {'probability': prob, 'prediction': pred, 'warning': warning}


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Link Predictor", page_icon="🔗", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #f6f8fa; color: #1a1a2e; }
.hero { text-align: center; padding: 2.5rem 0 1rem 0; }
.hero h1 { font-size: 2.2rem; font-weight: 700; letter-spacing: -0.02em; color: #1a1a2e; margin-bottom: 0.3rem; }
.hero .subtitle { font-size: 0.9rem; color: #6e7781; font-family: 'JetBrains Mono', monospace; }
.accent { color: #0969da; }
.stats-bar { display: flex; justify-content: center; gap: 2rem; margin: 0.8rem 0 1.2rem; flex-wrap: wrap; }
.stat-item { text-align: center; }
.stat-label { font-size: 0.65rem; color: #6e7781; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: 0.08em; }
.stat-value-blue  { font-size: 0.9rem; font-weight: 700; color: #0969da; }
.stat-value-green { font-size: 0.9rem; font-weight: 700; color: #1a7f37; }
.divider { border: none; border-top: 1px solid #d0d7de; margin: 1.2rem 0; }
.result-card { border-radius: 12px; padding: 1.5rem 2rem; margin-top: 1.2rem; text-align: center; }
.result-link   { background: #ddf4e4; border: 1px solid #82cfaa; }
.result-nolink { background: #ffebe9; border: 1px solid #ffcfc9; }
.prob-label { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: #57606a; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; }
.prob-value { font-size: 3.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; line-height: 1; margin-bottom: 0.6rem; }
.prob-link   { color: #0969da; }
.prob-nolink { color: #cf222e; }
.verdict { font-size: 1.05rem; font-weight: 600; margin-bottom: 0.3rem; }
.verdict-link   { color: #1a7f37; }
.verdict-nolink { color: #cf222e; }
.verdict-sub { font-size: 0.82rem; color: #57606a; }
.warn-box { background: #fff8c5; border: 1px solid #d4a72c; border-radius: 8px; padding: 0.7rem 1rem; margin-top: 1rem; font-size: 0.82rem; color: #7d4e00; font-family: 'JetBrains Mono', monospace; }
.feat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.6rem; margin-top: 1rem; }
.feat-item { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 8px; padding: 0.5rem 0.7rem; }
.feat-item .fname { color: #6e7781; font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; margin-bottom: 0.1rem; }
.feat-item .fval  { color: #1a1a2e; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }
div[data-testid="stNumberInput"] input { background: #ffffff !important; border: 1px solid #d0d7de !important; color: #1a1a2e !important; border-radius: 8px !important; font-family: 'JetBrains Mono', monospace !important; font-size: 1.05rem !important; }
div[data-testid="stButton"] > button { background: #0969da !important; color: white !important; border: none !important; border-radius: 8px !important; font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important; font-size: 1rem !important; padding: 0.6rem 2rem !important; width: 100%; }
div[data-testid="stButton"] > button:hover { background: #0860c8 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    path = "data/predictor.pkl"
    if not os.path.exists(path):
        return None, f"predictor.pkl not found at `{path}`"
    try:
        return joblib.load(path), None
    except Exception as e:
        return None, str(e)


predictor, load_error = load_predictor()

st.markdown("""
<div class="hero">
    <h1>🔗 Link <span class="accent">Predictor</span></h1>
    <div class="subtitle">social graph · directed edge prediction</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stats-bar">
  <div class="stat-item"><div class="stat-label">Best Model</div><div class="stat-value-blue">Extra Trees</div></div>
  <div class="stat-item"><div class="stat-label">ROC-AUC</div><div class="stat-value-green">0.9581</div></div>
  <div class="stat-item"><div class="stat-label">F1 Score</div><div class="stat-value-green">0.9025</div></div>
  <div class="stat-item"><div class="stat-label">Precision</div><div class="stat-value-green">0.9925</div></div>
  <div class="stat-item"><div class="stat-label">PR-AUC</div><div class="stat-value-green">0.9698</div></div>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.error(f"**Model load failed:** {load_error}\n\nMake sure `data/predictor.pkl` exists in the same folder as `app.py`.")
    st.stop()

st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("##### Enter Node Pair")
st.caption("Predict whether a directed follow edge exists: Source → Destination")

col1, col2 = st.columns(2)
with col1:
    source = st.number_input("Source Node ID", min_value=0, value=273084, step=1, format="%d")
with col2:
    dest = st.number_input("Destination Node ID", min_value=0, value=1505602, step=1, format="%d")

threshold = st.slider("Decision Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                       help="Probability above this → Link predicted")

predict_btn = st.button("Predict Link →")

if predict_btn:
    if source == dest:
        st.warning("Source and destination nodes are the same.")
    else:
        with st.spinner("Computing graph features…"):
            try:
                result  = predictor.predict(int(source), int(dest), threshold=threshold)
                prob    = result["probability"]
                pred    = result["prediction"]
                warning = result.get("warning")

                link_class  = "result-link"        if pred else "result-nolink"
                prob_class  = "prob-link"           if pred else "prob-nolink"
                verd_class  = "verdict-link"        if pred else "verdict-nolink"
                verdict     = "Link likely exists"  if pred else "No link expected"
                verdict_sub = (f"Probability {prob:.1%} exceeds threshold {threshold:.0%}"
                               if pred else
                               f"Probability {prob:.1%} below threshold {threshold:.0%}")

                st.markdown(f"""
                <div class="result-card {link_class}">
                    <div class="prob-label">Edge probability</div>
                    <div class="prob-value {prob_class}">{prob:.1%}</div>
                    <div class="verdict {verd_class}">{verdict}</div>
                    <div class="verdict-sub">{verdict_sub}</div>
                </div>
                """, unsafe_allow_html=True)

                if warning:
                    st.markdown(f'<div class="warn-box">⚠ {warning}</div>', unsafe_allow_html=True)

                with st.expander("📊 Feature breakdown (all 19 features)", expanded=False):
                    g = predictor.graph
                    src = int(source)
                    dst = int(dest)
                    src_known = g.has_node(src)
                    dst_known = g.has_node(dst)

                    # Recompute all features for display
                    s_in  = set(g.predecessors(src)) if src_known else set()
                    s_out = set(g.successors(src))   if src_known else set()
                    d_in  = set(g.predecessors(dst)) if dst_known else set()
                    d_out = set(g.successors(dst))   if dst_known else set()

                    def _jac(sa, sb):
                        u = sa | sb
                        return len(sa & sb) / len(u) if u else 0.0
                    
                    def _cos(sa, sb):
                        if not sa or not sb: return 0.0
                        return len(sa & sb) / (math.sqrt(len(sa)) * math.sqrt(len(sb)))

                    features_display = {
                        "jac_followers":     f"{_jac(s_in, d_in):.4f}",
                        "jac_followees":     f"{_jac(s_out, d_out):.4f}",
                        "cos_followers":     f"{_cos(s_in, d_in):.4f}",
                        "cos_followees":     f"{_cos(s_out, d_out):.4f}",
                        "num_followers_s":   len(s_in),
                        "num_followees_s":   len(s_out),
                        "num_followers_d":   len(d_in),
                        "num_followees_d":   len(d_out),
                        "inter_followers":   len(s_in & d_in),
                        "inter_followees":   len(s_out & d_out),
                        "shortest_path":     predictor._shortest_path(src, dst),
                        "same_wcc":          predictor._same_wcc(src, dst),
                        "adar_index":        f"{predictor._adar(src, dst):.4f}",
                        "katz_src":          f"{predictor.katz.get(src, predictor.mean_katz):.6f}",
                        "katz_dst":          f"{predictor.katz.get(dst, predictor.mean_katz):.6f}",
                        "hits_hub_src":      f"{predictor.hits_hub.get(src, predictor.mean_hub):.6f}",
                        "hits_hub_dst":      f"{predictor.hits_hub.get(dst, predictor.mean_hub):.6f}",
                        "hits_auth_src":     f"{predictor.hits_auth.get(src, predictor.mean_auth):.6f}",
                        "hits_auth_dst":     f"{predictor.hits_auth.get(dst, predictor.mean_auth):.6f}",
                    }

                    items_html = "".join([
                        f'<div class="feat-item"><div class="fname">{k}</div><div class="fval">{v}</div></div>'
                        for k, v in features_display.items()
                    ])
                    st.markdown(f'<div class="feat-grid">{items_html}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown(   
    "<p style='text-align:center; color:#aaa; font-size:0.75rem; font-family:JetBrains Mono,monospace;'>"
    "link-prediction · extra trees · 19 graph features · ROC-AUC 0.9581"
    "</p>",
    unsafe_allow_html=True,
)