"""Streamlit demo: HRI via Voice + Gesture.  Run:  streamlit run app.py"""
import glob
import io
import json
import os
import time

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import signal
from scipy.io import wavfile

import pipeline as P

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(ROOT, "assets", "samples")
ACCENT, DIM, BG, FG = "#7CFFB2", "#6b7480", "#0b0d10", "#d8dee6"
AMBER = "#FFB454"

st.set_page_config(page_title="HRI/CMD", page_icon="▮", layout="wide")
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'IBM Plex Sans', sans-serif; background:{BG}; color:{FG}; }}
h1,h2,h3,h4, code, .mono {{ font-family: 'IBM Plex Mono', monospace !important; font-weight:500; letter-spacing:-0.01em; }}
.stApp header {{ background: transparent; }}
.block-container {{ padding-top: 2rem; max-width: 1200px; }}
.hdr {{ font-family:'IBM Plex Mono'; color:{ACCENT}; font-size:1.05rem; border-bottom:1px solid #1f242b; padding-bottom:.6rem; margin-bottom:1rem; }}
.hdr span {{ color:{DIM}; }}
.big {{ font-family:'IBM Plex Mono'; font-size:2.2rem; color:{ACCENT}; line-height:1.1; }}
.lbl {{ font-family:'IBM Plex Mono'; font-size:.72rem; color:{DIM}; text-transform:uppercase; letter-spacing:.12em; }}
.stTabs [data-baseweb="tab-list"] {{ gap:1.5rem; border-bottom:1px solid #1f242b; }}
.stTabs [data-baseweb="tab"] {{ font-family:'IBM Plex Mono'; font-size:.85rem; background:transparent; }}
.stTabs [aria-selected="true"] {{ color:{ACCENT} !important; }}
div[data-testid="stMetric"] {{ border-left:2px solid #1f242b; padding-left:.8rem; }}
.stButton>button {{ border-radius:0; border:1px solid {ACCENT}; background:transparent; color:{ACCENT}; font-family:'IBM Plex Mono'; }}
.stButton>button:hover {{ background:{ACCENT}; color:{BG}; }}
.foot {{ color:{DIM}; font-size:.75rem; border-top:1px solid #1f242b; margin-top:2rem; padding-top:.6rem; }}
</style>""", unsafe_allow_html=True)


def style(fig, h=260):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor=BG, plot_bgcolor=BG,
                      font=dict(family="IBM Plex Mono", color=FG, size=11), legend=dict(orientation="h", y=1.15))
    fig.update_xaxes(gridcolor="#1a1f26", zeroline=False)
    fig.update_yaxes(gridcolor="#1a1f26", zeroline=False)
    return fig


@st.cache_resource
def load_models():
    return (joblib.load(os.path.join(P.MODEL_DIR, "voice.joblib")),
            joblib.load(os.path.join(P.MODEL_DIR, "gesture.joblib")))


@st.cache_data
def load_results():
    with open(os.path.join(P.MODEL_DIR, "results.json")) as f:
        return json.load(f)


@st.cache_data
def load_emg_samples():
    d = np.load(os.path.join(SAMPLES, "emg_samples.npz"))
    return d["windows"], d["labels"]


def read_wav(src):
    sr, w = wavfile.read(io.BytesIO(src) if isinstance(src, bytes) else src)
    w = w.astype(np.float32)
    if w.ndim > 1:
        w = w.mean(1)
    w /= max(np.abs(w).max(), 1) if w.dtype == np.float32 and np.abs(w).max() > 1.5 else 1
    if sr != P.SR:
        w = signal.resample_poly(w, P.SR, sr).astype(np.float32)
    return P.fix_length(w, P.SR)


res, (mv, mg) = load_results(), load_models()
CMDS = res["commands"]
rng = np.random.default_rng(int(time.time()) % 1000)

st.markdown('<div class="hdr">HRI/CMD <span>· voice + gesture command fusion · MLSP PBL</span></div>', unsafe_allow_html=True)
t_live, t_sig, t_mod, t_fus = st.tabs(["live", "signal chain", "models", "fusion study"])

# ----------------------------------------------------------------- LIVE
with t_live:
    if "pos" not in st.session_state:
        st.session_state.pos, st.session_state.trail, st.session_state.dir = (3, 3), [(3, 3)], (0, 1)
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown('<div class="lbl">1 · voice input</div>', unsafe_allow_html=True)
        mic = st.audio_input("say: go / stop / left / right / up / down", label_visibility="collapsed")
        wavs = sorted(glob.glob(os.path.join(SAMPLES, "*.wav")))
        pick = st.selectbox("…or use a dataset clip", ["—"] + [os.path.basename(w) for w in wavs])
        v_snr = st.select_slider("voice noise (SNR dB)", ["clean", 20, 10, 5, 0], "clean")
        st.markdown('<div class="lbl" style="margin-top:1rem">2 · gesture input (EMG replay)</div>', unsafe_allow_html=True)
        W, Y = load_emg_samples()
        gi = st.selectbox("EMG window (held-out subject)", range(len(W)),
                          format_func=lambda i: f"#{i}  · performed: {P.GESTURE_NAMES[CMDS[Y[i]]]}")
        g_snr = st.select_slider("EMG noise (SNR dB)", ["clean", 20, 10, 5, 0], "clean")
        run = st.button("execute command ▸")

    if run:
        snr = lambda s: None if s == "clean" else s
        if mic is not None:
            xv = read_wav(mic.getvalue())
        elif pick != "—":
            xv = read_wav(os.path.join(SAMPLES, pick))
        else:
            xv = None
        t0 = time.perf_counter()
        pg = mg.predict_proba(P.emg_features(P.add_noise(W[gi], snr(g_snr), rng))[None])
        if xv is not None:
            pv = mv.predict_proba(P.voice_features(P.add_noise(xv, snr(v_snr), rng))[None])
        else:
            pv = np.ones_like(pg) / len(CMDS)
        pf = P.fuse(pv, pg, "confidence")
        ms = (time.perf_counter() - t0) * 1000
        cmd = CMDS[int(pf.argmax())]
        st.session_state.last = dict(pv=pv[0], pg=pg[0], pf=pf[0], ms=ms, cmd=cmd, xv=xv)
        x, y = st.session_state.pos
        dx, dy = {"left": (-1, 0), "right": (1, 0), "up": (0, 1), "down": (0, -1)}.get(cmd, (0, 0))
        if cmd in ("left", "right", "up", "down"):
            st.session_state.dir = (dx, dy)
        if cmd == "go":
            dx, dy = st.session_state.dir
        x, y = min(6, max(0, x + dx)), min(6, max(0, y + dy))
        st.session_state.pos = (x, y)
        st.session_state.trail = (st.session_state.trail + [(x, y)])[-12:]

    with right:
        L = st.session_state.get("last")
        st.markdown('<div class="lbl">fused command</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="big">{L["cmd"].upper() if L else "—"}</div>', unsafe_allow_html=True)
        if L:
            c1, c2, c3 = st.columns(3)
            c1.metric("voice", CMDS[int(L["pv"].argmax())], f'{L["pv"].max():.0%}')
            c2.metric("gesture", CMDS[int(L["pg"].argmax())], f'{L["pg"].max():.0%}')
            c3.metric("latency", f'{L["ms"]:.0f} ms')
            fig = go.Figure()
            for name, key, col in [("voice", "pv", AMBER), ("gesture", "pg", "#6fa8ff"), ("fused", "pf", ACCENT)]:
                fig.add_bar(x=CMDS, y=L[key], name=name, marker_color=col)
            fig.update_layout(barmode="group")
            st.plotly_chart(style(fig, 220), use_container_width=True)
        tx, ty = zip(*st.session_state.trail)
        fig = go.Figure()
        fig.add_scatter(x=tx, y=ty, mode="lines", line=dict(color=DIM, width=1, dash="dot"), showlegend=False)
        fig.add_scatter(x=[st.session_state.pos[0]], y=[st.session_state.pos[1]], mode="markers",
                        marker=dict(symbol="square", size=26, color=ACCENT), showlegend=False)
        fig.update_xaxes(range=[-.5, 6.5], dtick=1, showticklabels=False)
        fig.update_yaxes(range=[-.5, 6.5], dtick=1, showticklabels=False, scaleanchor="x")
        st.markdown('<div class="lbl">robot (7×7 grid)</div>', unsafe_allow_html=True)
        st.plotly_chart(style(fig, 280), use_container_width=True)
        if st.button("reset robot"):
            st.session_state.pos, st.session_state.trail = (3, 3), [(3, 3)]
            st.rerun()

# ----------------------------------------------------------------- SIGNAL CHAIN
with t_sig:
    use_f = st.toggle("apply DSP chain", True)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("#### voice")
        st.caption("pre-emphasis → band-pass 80–4000 Hz → energy VAD → STFT → mel → log → DCT (MFCC)")
        vf = st.selectbox("clip", [os.path.basename(w) for w in wavs], key="vclip")
        raw = read_wav(os.path.join(SAMPLES, vf))
        proc = P.voice_preprocess(raw, use_f)
        t = np.arange(P.SR) / P.SR
        fig = go.Figure()
        fig.add_scatter(x=t, y=raw, name="raw", line=dict(color=DIM, width=1))
        fig.add_scatter(x=t, y=proc, name="processed", line=dict(color=ACCENT, width=1))
        st.plotly_chart(style(fig.update_xaxes(title="s")), use_container_width=True)
        m, mel = P.mfcc(proc)
        st.plotly_chart(style(go.Figure(go.Heatmap(z=mel.T, colorscale="Viridis", showscale=False)).update_layout(title="log-mel spectrogram")), use_container_width=True)
        st.plotly_chart(style(go.Figure(go.Heatmap(z=m.T, colorscale="Magma", showscale=False)).update_layout(title="MFCC (13 × frames)")), use_container_width=True)
    with b:
        st.markdown("#### gesture (EMG)")
        st.caption("band-pass 10–90 Hz → 50 Hz notch → rectify + RMS envelope → time/spectral features")
        ei = st.selectbox("window", range(len(W)), key="eclip", format_func=lambda i: f"#{i} · {P.GESTURE_NAMES[CMDS[Y[i]]]}")
        w = W[ei]
        wf = P.emg_filter(w) if use_f else w - w.mean(0)
        te = np.arange(P.EMG_WIN) / P.EMG_FS
        ch = st.slider("channel", 1, 8, 1) - 1
        fig = go.Figure()
        fig.add_scatter(x=te, y=w[:, ch] * 1e3, name="raw", line=dict(color=DIM, width=1))
        fig.add_scatter(x=te, y=wf[:, ch] * 1e3, name="filtered", line=dict(color=ACCENT, width=1))
        st.plotly_chart(style(fig.update_xaxes(title="s")), use_container_width=True)
        env = P.emg_envelope(wf)
        st.plotly_chart(style(go.Figure(go.Heatmap(z=(env * 1e3).T, colorscale="Viridis", showscale=False)).update_layout(title="RMS envelope · 8 channels")), use_container_width=True)
        f0, p0 = signal.welch(w[:, ch], P.EMG_FS, nperseg=64)
        f1, p1 = signal.welch(wf[:, ch], P.EMG_FS, nperseg=64)
        fig = go.Figure()
        fig.add_scatter(x=f0, y=p0, name="raw", line=dict(color=DIM))
        fig.add_scatter(x=f1, y=p1, name="filtered", line=dict(color=ACCENT))
        st.plotly_chart(style(fig.update_yaxes(type="log").update_layout(title="PSD (Welch)")), use_container_width=True)

# ----------------------------------------------------------------- MODELS
with t_mod:
    for mod, label in [("voice", "voice · Speech Commands v0.02"), ("gesture", "gesture · UCI EMG (subject-independent)")]:
        st.markdown(f"#### {label}")
        R = res[mod]["models"]
        rows = [{"model": k, "accuracy": v["accuracy"], "precision": v["precision"], "recall": v["recall"],
                 "F1": v["f1"], "latency ms": v["latency_ms"],
                 "acc w/o DSP": res[mod]["ablation"][k]["without_dsp"]} for k, v in R.items()]
        st.dataframe(rows, hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in rows[0] if c != "model"})
        c1, c2 = st.columns(2, gap="large")
        fig = go.Figure()
        for met, col in [("accuracy", ACCENT), ("f1", AMBER)]:
            fig.add_bar(x=list(R), y=[R[k][met] for k in R], name=met, marker_color=col)
        fig.update_layout(barmode="group", title="with DSP")
        c1.plotly_chart(style(fig), use_container_width=True)
        best = res["best"][mod]
        cm = np.array(R[best]["confusion"])
        fig = go.Figure(go.Heatmap(z=cm, x=CMDS, y=CMDS, colorscale=[[0, BG], [1, ACCENT]], showscale=False))
        fig.update_layout(title=f"confusion · {best}")
        fig.update_yaxes(autorange="reversed")
        c2.plotly_chart(style(fig), use_container_width=True)
    st.caption("Best model per modality (by macro-F1) is used in the live demo.")

# ----------------------------------------------------------------- FUSION
with t_fus:
    F = res["fusion"]
    st.markdown("#### confidence-weighted late fusion under noise")
    st.caption("w = 1 − normalised entropy of each modality's posterior; fused = (w_v·p_v + w_g·p_g)/(w_v + w_g). "
               "Voice/gesture clips are paired by command label (synthetic pairing). Models trained on clean data only.")
    c1, c2 = st.columns(2, gap="large")
    for col, key, title in [(c1, "voice_noisy", "noise on VOICE (gesture clean)"), (c2, "gesture_noisy", "noise on EMG (voice clean)")]:
        fig = go.Figure()
        for mode, name, colr in [("voice", "voice only", AMBER), ("gesture", "gesture only", "#6fa8ff"),
                                 ("average", "fused · plain average", DIM), ("confidence", "fused · confidence-weighted", ACCENT)]:
            fig.add_scatter(x=[str(s) for s in F["snr"]], y=F[key][mode], name=name, mode="lines+markers",
                            line=dict(color=colr, width=3 if mode == "confidence" else 1.5))
        fig.update_layout(title=title).update_xaxes(title="SNR dB (clean → noisy)").update_yaxes(title="accuracy", range=[0, 1])
        col.plotly_chart(style(fig, 340), use_container_width=True)
    vn, gn = F["voice_noisy"], F["gesture_noisy"]
    st.markdown(f"**Finding.** Clean: voice {vn['voice'][0]:.1%}, gesture {gn['gesture'][0]:.1%}, fused {vn['confidence'][0]:.1%}. "
                f"At 10 dB voice noise single-voice falls to {vn['voice'][2]:.1%} while fusion holds {vn['confidence'][2]:.1%}; "
                f"at 5 dB EMG noise gesture-only drops to {gn['gesture'][3]:.1%} vs fused {gn['confidence'][3]:.1%}.")

st.markdown('<div class="foot">Data: Google Speech Commands v0.02 (Warden 2018) · UCI "EMG data for gestures" (Lobov et al. 2018). '
            'Models: SVM / Random Forest / k-NN (scikit-learn). Gesture input is a replay of held-out recordings.</div>', unsafe_allow_html=True)
