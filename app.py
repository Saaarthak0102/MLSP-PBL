"""Streamlit demo: HRI via Voice (how many units) + Gesture (which direction).
Run:  streamlit run app.py
Left half = controls / measured stats.  Right half = robot simulation.
"""
import base64
import glob
import io
import json
import os
import time

import cv2
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from scipy import signal
from scipy.io import wavfile

import pipeline as P
import vision as V

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(ROOT, "assets", "samples")
ACCENT, DIM, BG, FG = "#7CFFB2", "#6b7480", "#0b0d10", "#d8dee6"
AMBER, BLUE = "#FFB454", "#6fa8ff"
GRID = 9
START = (GRID // 2, GRID // 2)
ARROW = {"left": "←", "right": "→", "up": "↑", "down": "↓"}
VEC = {"left": (-1, 0), "right": (1, 0), "up": (0, 1), "down": (0, -1)}

st.set_page_config(page_title="GestVox · voice + gesture robot control", page_icon="▮", layout="wide")
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'IBM Plex Sans', sans-serif; background:{BG}; color:{FG}; }}
h1,h2,h3,h4, code, .mono {{ font-family: 'IBM Plex Mono', monospace !important; font-weight:500; letter-spacing:-0.01em; }}
.stApp header {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; max-width: 1700px; }}
.hdr {{ font-family:'IBM Plex Mono'; color:{ACCENT}; font-size:1.05rem; border-bottom:1px solid #1f242b; padding-bottom:.6rem; margin-bottom:1rem; }}
.hdr span {{ color:{DIM}; }}
.big {{ font-family:'IBM Plex Mono'; font-size:2.1rem; color:{ACCENT}; line-height:1.15; }}
.lbl {{ font-family:'IBM Plex Mono'; font-size:.72rem; color:{DIM}; text-transform:uppercase; letter-spacing:.12em; }}
.stTabs [data-baseweb="tab-list"] {{ gap:1.2rem; border-bottom:1px solid #1f242b; }}
.stTabs [data-baseweb="tab"] {{ font-family:'IBM Plex Mono'; font-size:.82rem; background:transparent; }}
.stTabs [aria-selected="true"] {{ color:{ACCENT} !important; }}
div[data-testid="stMetric"] {{ border-left:2px solid #1f242b; padding-left:.8rem; }}
.stButton>button {{ border-radius:0; border:1px solid {ACCENT}; background:transparent; color:{ACCENT}; font-family:'IBM Plex Mono'; }}
.stButton>button:hover {{ background:{ACCENT}; color:{BG}; }}
.log {{ font-family:'IBM Plex Mono'; font-size:.8rem; color:{DIM}; line-height:1.6; }}
.log b {{ color:{FG}; font-weight:500; }}
.foot {{ color:{DIM}; font-size:.75rem; border-top:1px solid #1f242b; margin-top:2rem; padding-top:.6rem; }}
</style>""", unsafe_allow_html=True)


def style(fig, h=260):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor=BG, plot_bgcolor=BG,
                      font=dict(family="IBM Plex Mono", color=FG, size=11), legend=dict(orientation="h", y=1.15))
    fig.update_xaxes(gridcolor="#1a1f26", zeroline=False)
    fig.update_yaxes(gridcolor="#1a1f26", zeroline=False)
    return fig


def _mt(*names):
    """Modification-time key so a retrain is picked up without restarting the server."""
    return tuple(os.path.getmtime(os.path.join(P.MODEL_DIR, n)) for n in names if os.path.exists(os.path.join(P.MODEL_DIR, n)))


@st.cache_resource
def _load_models(_k):
    return (joblib.load(os.path.join(P.MODEL_DIR, "voice.joblib")),
            joblib.load(os.path.join(P.MODEL_DIR, "gesture.joblib")))


@st.cache_resource
def _load_cam(_k):
    return joblib.load(os.path.join(P.MODEL_DIR, "cam_gesture.joblib"))


@st.cache_data
def _load_results(_k):
    with open(os.path.join(P.MODEL_DIR, "results.json")) as f:
        return json.load(f)


def load_models():
    return _load_models(_mt("voice.joblib", "gesture.joblib"))


def load_cam():
    return _load_cam(_mt("cam_gesture.joblib"))


def load_results():
    return _load_results(_mt("results.json"))


@st.cache_data
def load_emg_samples():
    d = np.load(os.path.join(SAMPLES, "emg_samples.npz"))
    return d["windows"], d["labels"]


def read_wav(src):
    sr, w = wavfile.read(io.BytesIO(src) if isinstance(src, bytes) else src)
    w = w.astype(np.float32)
    if w.ndim > 1:
        w = w.mean(1)
    if np.abs(w).max() > 1.5:
        w /= 32768.0
    if sr != P.SR:
        w = signal.resample_poly(w, P.SR, sr).astype(np.float32)
    return P.fix_length(w, P.SR)


capture = components.declare_component("capture", path=os.path.join(ROOT, "components", "capture"))
res, (mv, mg) = load_results(), load_models()
UNITS, DIRS = res["units"], res["dirs"]
rng = np.random.default_rng(int(time.time()) % 1000)

# ----------------------------------------------------------------------------- robot state
ss = st.session_state
if "pos" not in ss:
    ss.pos, ss.trail, ss.rlog, ss.pending, ss.history = START, [START], [], None, []


def plan_move(units, direction):
    """Return list of cells visited (excluding start) and whether the wall blocked it."""
    dx, dy = VEC[direction]
    x, y = ss.pos
    path = []
    for _ in range(units):
        nx, ny = x + dx, y + dy
        if not (0 <= nx < GRID and 0 <= ny < GRID):
            return path, True
        x, y = nx, ny
        path.append((x, y))
    return path, False


def robot_fig(pos, trail, target=None, h=520):
    fig = go.Figure()
    if len(trail) > 1:
        tx, ty = zip(*trail)
        fig.add_scatter(x=tx, y=ty, mode="lines", line=dict(color=ACCENT, width=2, dash="dot"), opacity=.5, showlegend=False, hoverinfo="skip")
    if target is not None:
        fig.add_scatter(x=[target[0]], y=[target[1]], mode="markers", showlegend=False, hoverinfo="skip",
                        marker=dict(symbol="square-open", size=34, color=AMBER, line=dict(width=2)))
    fig.add_scatter(x=[pos[0]], y=[pos[1]], mode="markers", showlegend=False,
                    hovertemplate=f"({pos[0]}, {pos[1]})<extra></extra>",
                    marker=dict(symbol="square", size=34, color=ACCENT))
    fig.update_xaxes(range=[-.5, GRID - .5], tickmode="array", tickvals=list(range(GRID)), gridcolor="#1f242b", showline=True, linecolor="#1f242b", zeroline=False)
    fig.update_yaxes(range=[-.5, GRID - .5], tickmode="array", tickvals=list(range(GRID)), gridcolor="#1f242b", showline=True, linecolor="#1f242b",
                     zeroline=False, scaleanchor="x")
    fig.update_layout(height=h, margin=dict(l=24, r=10, t=10, b=24), paper_bgcolor=BG, plot_bgcolor="#0e1115",
                      font=dict(family="IBM Plex Mono", color=DIM, size=10))
    return fig


# ----------------------------------------------------------------------------- audio / session processing
def audio_stats(x):
    fl, hop = 400, 160
    n = 1 + (len(x) - fl) // hop
    idx = np.arange(fl)[None] + hop * np.arange(n)[:, None]
    en = (x[idx] ** 2).mean(1) + 1e-12
    noise = np.percentile(en, 10)
    voiced = en > max(noise * 8, 0.1 * en.max())
    sig = en[voiced].mean() if voiced.any() else noise
    cs = np.r_[0, np.cumsum(en)]
    w = P.SR // hop
    best = int(np.argmax(cs[w:] - cs[:-w])) if n > w else 0
    start = min(best * hop, max(len(x) - P.SR, 0))
    return dict(duration=len(x) / P.SR, rms=float(np.sqrt((x ** 2).mean())), peak=float(np.abs(x).max()),
                voiced_pct=float(voiced.mean()), snr_db=float(10 * np.log10(sig / noise)), start=start)


def decide(pv, pg, v_ok, g_ok, th):
    """Complementary fusion + confidence gating: units from voice, direction from gesture."""
    if not v_ok or not g_ok:
        return None, "missing " + ("speech" if not v_ok else "hand")
    cv_, cg_ = float(pv.max()), float(pg.max())
    if cv_ < th or cg_ < th:
        return None, f"low confidence ({'voice' if cv_ < th else 'gesture'} {min(cv_, cg_):.0%} < {th:.0%})"
    return (int(pv.argmax()) + 1, DIRS[int(pg.argmax())]), "accepted"


def process_session(data):
    T = {}
    t = time.perf_counter()
    x = np.frombuffer(base64.b64decode(data["audio"]), dtype=np.int16).astype(np.float32) / 32768.0
    frames = V.decode_jpegs([base64.b64decode(f) for f in data["frames"]])
    T["decode"] = time.perf_counter() - t
    t = time.perf_counter()
    A = audio_stats(x)
    win = P.fix_length(x[A["start"]:A["start"] + P.SR], P.SR)
    proc = P.voice_preprocess(win, True)
    v_ok = A["peak"] > 0.02 and A["voiced_pct"] > 0.03
    pv = mv.predict_proba(P.voice_features(win)[None])[0] if v_ok else np.ones(len(UNITS)) / len(UNITS)
    T["voice DSP + model"] = time.perf_counter() - t
    t = time.perf_counter()
    if len(frames) >= 5:
        feat, S, over = V.analyze(frames)
    else:
        feat, S, over = np.zeros(len(V.FEATURE_NAMES), np.float32), None, []
    T["OpenCV analysis"] = time.perf_counter() - t
    t = time.perf_counter()
    cam = load_cam()
    g_ok = S is not None and S["detected"].mean() >= 0.2
    per_model = {n: m.predict_proba(feat[None])[0] for n, m in cam.items()}
    best_cam = load_results()["camera"]["best"]
    pg = per_model[best_cam] if g_ok else np.ones(len(DIRS)) / len(DIRS)
    T["classify"] = time.perf_counter() - t
    return dict(A=A, x=x, proc=proc, pv=pv, v_ok=v_ok, S=S, over=over, feat=feat, pg=pg, g_ok=g_ok, per_model=per_model,
                best_cam=best_cam, T=T, n_frames=len(frames), elapsed=data["elapsed"], id=data["id"])


def queue_command(source, pv, pg, v_ok, g_ok, th, extra=None):
    cmd, why = decide(pv, pg, v_ok, g_ok, th)
    rec = dict(run=len(ss.history) + 1, source=source,
               units=UNITS[int(pv.argmax())] if v_ok else "—", v_conf=round(float(pv.max()), 2) if v_ok else None,
               direction=DIRS[int(pg.argmax())] if g_ok else "—", g_conf=round(float(pg.max()), 2) if g_ok else None,
               executed=f"{cmd[0]} {cmd[1]}" if cmd else f"rejected · {why}")
    if extra:
        rec.update(extra)
    ss.history.append(rec)
    ss.decision = dict(cmd=cmd, why=why, pv=pv, pg=pg, v_ok=v_ok, g_ok=g_ok)
    if cmd:
        ss.pending = cmd


# ----------------------------------------------------------------------------- rendering of one session
def render_decision(D, T=None):
    cmd = D["cmd"]
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown('<div class="lbl">command</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="big">{f"{cmd[0]} {ARROW[cmd[1]]} {cmd[1]}" if cmd else "REJECTED"}</div>', unsafe_allow_html=True)
        said = f"Heard **{UNITS[int(D['pv'].argmax())]}** ({D['pv'].max():.0%})" if D["v_ok"] else "No speech detected"
        saw = f"saw swipe **{DIRS[int(D['pg'].argmax())]}** ({D['pg'].max():.0%})" if D["g_ok"] else "no hand detected"
        tail = f" → robot moves **{cmd[0]}** cell{'s' if cmd[0] > 1 else ''} **{cmd[1]}**." if cmd else f" → not executed: {D['why']}. Repeat the command."
        st.markdown(f"{said}, {saw}{tail}")
    with c2:
        if T:
            st.metric("total processing", f"{sum(T.values()) * 1000:.0f} ms")
            st.dataframe(pd.DataFrame({"stage": list(T), "ms": [v * 1000 for v in T.values()]}).set_index("stage").style.format("{:.1f}"),
                         use_container_width=True)
    a, b = st.columns(2)
    for col, title, labels, p, colr in [(a, "voice → units", UNITS, D["pv"], AMBER), (b, "gesture → direction", DIRS, D["pg"], BLUE)]:
        fig = go.Figure(go.Bar(x=labels, y=p, marker_color=colr))
        fig.update_layout(title=title).update_yaxes(range=[0, 1])
        col.plotly_chart(style(fig, 200), use_container_width=True, key=f"{title}_{len(ss.history)}")


def render_session(R):
    A, S = R["A"], R["S"]
    st.markdown("#### voice · measured")
    m = st.columns(5)
    m[0].metric("captured", f"{A['duration']:.2f} s")
    m[1].metric("peak", f"{20 * np.log10(A['peak'] + 1e-9):.1f} dBFS")
    m[2].metric("RMS", f"{A['rms']:.4f}")
    m[3].metric("voiced", f"{A['voiced_pct']:.0%}")
    m[4].metric("est. SNR", f"{A['snr_db']:.1f} dB")
    t5 = np.arange(len(R["x"])) / P.SR
    fig = go.Figure()
    fig.add_scatter(x=t5[::8], y=R["x"][::8], line=dict(color=DIM, width=1))
    fig.add_vrect(x0=A["start"] / P.SR, x1=A["start"] / P.SR + 1, fillcolor=ACCENT, opacity=.15, line_width=0)
    fig.update_layout(title="capture · shaded = 1 s window classified", showlegend=False).update_xaxes(title="s")
    a, b = st.columns(2)
    a.plotly_chart(style(fig, 230), use_container_width=True)
    mf, mel = P.mfcc(R["proc"])
    b.plotly_chart(style(go.Figure(go.Heatmap(z=mel.T, colorscale="Viridis", showscale=False)).update_layout(title="log-mel (band-pass + VAD)"), 230), use_container_width=True)
    if not R["v_ok"]:
        st.warning("Audio too quiet / no speech found.")

    st.markdown("#### gesture · measured (OpenCV)")
    if S is None:
        st.warning("Too few video frames received.")
        return
    m = st.columns(5)
    m[0].metric("frames", R["n_frames"])
    m[1].metric("effective fps", f"{R['n_frames'] / max(R['elapsed'], 1e-6):.1f}")
    m[2].metric("hand detected", f"{S['detected'].mean():.0%}")
    m[3].metric("net dx", f"{R['feat'][0]:+.2f}")
    m[4].metric("net dy", f"{R['feat'][1]:+.2f}")
    pick = np.linspace(0, len(R["over"]) - 1, 5).astype(int)
    st.image(np.hstack([cv2.cvtColor(R["over"][i], cv2.COLOR_BGR2RGB) for i in pick]),
             caption="face (grey) · hand contour (blue) · convex hull (green) · centroid (red)", width="stretch")
    st.image(np.hstack([S["masks"][i] for i in pick]), caption="skin mask after morphology", width="stretch")
    a, b = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=S["cx"], y=-S["cy"], mode="lines+markers", line=dict(color=ACCENT),
                    marker=dict(size=4, color=np.arange(len(S["cx"])), colorscale="Viridis"))
    fig.update_layout(title="hand trajectory (colour = time)", showlegend=False).update_xaxes(range=[0, 1], title="x").update_yaxes(range=[-1, 0], title="y")
    a.plotly_chart(style(fig, 260), use_container_width=True)
    fig = go.Figure()
    fig.add_scatter(y=S["area"] * 100, name="area %", line=dict(color=BLUE))
    fig.add_scatter(y=S["motion"], name="motion energy", line=dict(color=DIM))
    fig.update_layout(title="per-frame series").update_xaxes(title="frame")
    b.plotly_chart(style(fig, 260), use_container_width=True)
    rows = [{"model": n + (" ★" if n == R["best_cam"] else ""), "direction": DIRS[int(p.argmax())], "confidence": float(p.max())}
            for n, p in R["per_model"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={"confidence": st.column_config.NumberColumn(format="%.3f")})
    if not R["g_ok"]:
        st.warning("Hand not found in enough frames. Try brighter light, a plain background, and keep your hand in front of your face.")

    with st.expander("improve the gesture model with your own data"):
        lab = st.selectbox("swipe you just performed", V.GESTURES)
        c1, c2 = st.columns(2)
        if c1.button("save this clip as a training sample"):
            fp = P.CAM_REAL
            if os.path.exists(fp):
                d = np.load(fp)
                X0, y0 = list(d["X"]), list(d["y"])
            else:
                X0, y0 = [], []
            X0.append(R["feat"]); y0.append(V.GESTURES.index(lab))
            np.savez_compressed(fp, X=np.array(X0), y=np.array(y0))
            st.success(f"saved ({len(y0)} real clips total)")
        if c2.button("retrain camera models now"):
            with st.spinner("training…"):
                P.train_cam(n_per_class=100)
            _load_cam.clear(); _load_results.clear()
            st.success("retrained — record again to use the new models")


# ============================================================================= LAYOUT
st.markdown('<div class="hdr">GESTVOX <span>· speak the distance · gesture the direction · robot obeys</span></div>', unsafe_allow_html=True)
left, right = st.columns(2, gap="large")

with left:
    t_sess, t_rep, t_sig, t_mod, t_fus = st.tabs(["live session", "dataset replay", "signal chain", "models", "joint study"])

    # ------------------------------------------------------------- LIVE SESSION
    with t_sess:
        st.markdown('<div class="lbl">say one number (one–five) while swiping your hand left / right / up / down</div>', unsafe_allow_html=True)
        th = st.slider("confidence gate (reject command if either input is below)", 0.2, 0.9, 0.45, 0.05)
        data = capture(key="cap", default=None)
        if data and data.get("id") != ss.get("last_id"):
            ss.last_id = data["id"]
            with st.status("processing capture…") as status:
                R = process_session(data)
                ss.sess = R
                queue_command("camera+mic", R["pv"], R["pg"], R["v_ok"], R["g_ok"], th,
                              dict(snr_db=round(R["A"]["snr_db"], 1), hand_pct=round(float(R["S"]["detected"].mean()) if R["S"] else 0, 2)))
                status.update(label="done", state="complete")
        if ss.get("sess"):
            render_decision(ss.decision, ss.sess["T"])
            render_session(ss.sess)
        else:
            st.info("No capture yet. Press **start** — the browser asks for camera + microphone permission the first time.")

    # ------------------------------------------------------------- DATASET REPLAY
    with t_rep:
        st.caption("Offline version: a dataset voice clip gives the units, a held-out UCI-EMG recording gives the direction.")
        wavs = sorted(glob.glob(os.path.join(SAMPLES, "*.wav")))
        W, Y = load_emg_samples()
        pick = st.selectbox("voice clip (Speech Commands)", [os.path.basename(w) for w in wavs], key="rep_clip")
        v_snr = st.select_slider("voice noise (SNR dB)", ["clean", 20, 10, 5, 0], "clean", key="rep_vs")
        gi = st.selectbox("EMG window (held-out subject)", range(len(W)),
                          format_func=lambda i: f"#{i} · performed: {P.GESTURE_NAMES[DIRS[Y[i]]]} → {DIRS[Y[i]]}", key="rep_g")
        g_snr = st.select_slider("EMG noise (SNR dB)", ["clean", 20, 10, 5, 0], "clean", key="rep_gs")
        if st.button("execute ▸", key="rep_run"):
            snr = lambda s: None if s == "clean" else s
            xv = read_wav(os.path.join(SAMPLES, pick))
            pv = mv.predict_proba(P.voice_features(P.add_noise(xv, snr(v_snr), rng))[None])[0]
            pg = mg.predict_proba(P.emg_features(P.add_noise(W[gi], snr(g_snr), rng))[None])[0]
            queue_command("dataset", pv, pg, True, True, 0.45)
            ss.rep_shown = True
        if ss.get("rep_shown") and ss.get("decision") and ss.history and ss.history[-1]["source"] == "dataset":
            render_decision(ss.decision)

    # ------------------------------------------------------------- SIGNAL CHAIN
    with t_sig:
        use_f = st.toggle("apply DSP chain", True)
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
        m_, mel = P.mfcc(proc)
        a, b = st.columns(2)
        a.plotly_chart(style(go.Figure(go.Heatmap(z=mel.T, colorscale="Viridis", showscale=False)).update_layout(title="log-mel")), use_container_width=True)
        b.plotly_chart(style(go.Figure(go.Heatmap(z=m_.T, colorscale="Magma", showscale=False)).update_layout(title="MFCC")), use_container_width=True)
        st.markdown("#### gesture (EMG)")
        st.caption("band-pass 10–90 Hz → 50 Hz notch → rectify + RMS envelope → time/spectral features")
        ei = st.selectbox("window", range(len(W)), key="eclip", format_func=lambda i: f"#{i} · {DIRS[Y[i]]}")
        w = W[ei]
        wf = P.emg_filter(w) if use_f else w - w.mean(0)
        te = np.arange(P.EMG_WIN) / P.EMG_FS
        ch = st.slider("channel", 1, 8, 1) - 1
        fig = go.Figure()
        fig.add_scatter(x=te, y=w[:, ch] * 1e3, name="raw", line=dict(color=DIM, width=1))
        fig.add_scatter(x=te, y=wf[:, ch] * 1e3, name="filtered", line=dict(color=ACCENT, width=1))
        st.plotly_chart(style(fig.update_xaxes(title="s")), use_container_width=True)
        a, b = st.columns(2)
        env = P.emg_envelope(wf)
        a.plotly_chart(style(go.Figure(go.Heatmap(z=(env * 1e3).T, colorscale="Viridis", showscale=False)).update_layout(title="RMS envelope · 8 ch")), use_container_width=True)
        f0, p0 = signal.welch(w[:, ch], P.EMG_FS, nperseg=64)
        f1, p1 = signal.welch(wf[:, ch], P.EMG_FS, nperseg=64)
        fig = go.Figure()
        fig.add_scatter(x=f0, y=p0, name="raw", line=dict(color=DIM))
        fig.add_scatter(x=f1, y=p1, name="filtered", line=dict(color=ACCENT))
        b.plotly_chart(style(fig.update_yaxes(type="log").update_layout(title="PSD (Welch)")), use_container_width=True)

    # ------------------------------------------------------------- MODELS
    with t_mod:
        blocks = [("voice", "voice → units · Speech Commands v0.02", UNITS), ("gesture", "gesture → direction · UCI EMG (subject-independent)", DIRS)]
        if "camera" in res:
            blocks.append(("camera", "gesture → direction · OpenCV camera (synthetic-trained)", DIRS))
        for mod, label, labs in blocks:
            st.markdown(f"#### {label}")
            R_ = res[mod]["models"]
            rows = [{"model": k, "accuracy": v["accuracy"], "precision": v["precision"], "recall": v["recall"],
                     "F1": v["f1"], "latency ms": v["latency_ms"]} for k, v in R_.items()]
            if "ablation" in res[mod]:
                for r in rows:
                    r["acc w/o DSP"] = res[mod]["ablation"][r["model"]]["without_dsp"]
            st.dataframe(rows, hide_index=True, use_container_width=True,
                         column_config={c: st.column_config.NumberColumn(format="%.3f") for c in rows[0] if c != "model"})
            c1, c2 = st.columns(2)
            fig = go.Figure()
            for met, col in [("accuracy", ACCENT), ("f1", AMBER)]:
                fig.add_bar(x=list(R_), y=[R_[k][met] for k in R_], name=met, marker_color=col)
            fig.update_layout(barmode="group")
            c1.plotly_chart(style(fig), use_container_width=True, key=f"bar_{mod}")
            best = res["best"][mod] if mod in res["best"] else res[mod]["best"]
            fig = go.Figure(go.Heatmap(z=np.array(R_[best]["confusion"]), x=labs, y=labs, colorscale=[[0, BG], [1, ACCENT]], showscale=False))
            fig.update_layout(title=f"confusion · {best}").update_yaxes(autorange="reversed")
            c2.plotly_chart(style(fig), use_container_width=True, key=f"cm_{mod}")

    # ------------------------------------------------------------- JOINT STUDY
    with t_fus:
        F, G = res["fusion"], res["gating"]
        st.markdown("#### complementary fusion: units (voice) + direction (gesture)")
        st.caption("A command is correct only if BOTH slots are right (joint accuracy). Offline, voice from Speech Commands and gesture from "
                   "UCI-EMG are paired at random (synthetic pairing). Models trained on clean data only.")
        c1, c2 = st.columns(2)
        for col, key, title in [(c1, "voice_noisy", "noise on VOICE"), (c2, "gesture_noisy", "noise on EMG")]:
            fig = go.Figure()
            for k, name, colr in [("units_only", "units only", AMBER), ("direction_only", "direction only", BLUE), ("joint", "joint command", ACCENT)]:
                fig.add_scatter(x=[str(s) for s in F["snr"]], y=F[key][k], name=name, mode="lines+markers",
                                line=dict(color=colr, width=3 if k == "joint" else 1.5))
            fig.update_layout(title=title).update_xaxes(title="SNR dB").update_yaxes(range=[0, 1], title="accuracy")
            col.plotly_chart(style(fig, 300), use_container_width=True, key=f"fus_{key}")
        st.markdown("#### confidence gating — refuse to move when unsure")
        fig = go.Figure()
        fig.add_scatter(x=G["threshold"], y=G["accuracy_accepted"], name="accuracy of executed commands", line=dict(color=ACCENT, width=3))
        fig.add_scatter(x=G["threshold"], y=G["coverage"], name="coverage (commands executed)", line=dict(color=AMBER))
        fig.add_scatter(x=G["threshold"], y=G["wrong_moves_avoided"], name="wrong moves avoided", line=dict(color=BLUE))
        fig.update_layout(title="gate threshold sweep (clean data)").update_xaxes(title="threshold on each modality's top probability").update_yaxes(range=[0, 1])
        st.plotly_chart(style(fig, 320), use_container_width=True, key="gate")
        i = min(range(len(G["threshold"])), key=lambda k: abs(G["threshold"][k] - 0.45))
        st.markdown(f"**Finding.** Without gating the joint command is right {res['joint_clean']:.1%} of the time. A gate at "
                    f"{G['threshold'][i]:.2f} executes {G['coverage'][i]:.0%} of commands at {G['accuracy_accepted'][i]:.1%} accuracy and prevents "
                    f"{G['wrong_moves_avoided'][i]:.0%} of wrong moves.")

# ============================================================================= RIGHT: SIMULATION
with right:
    st.markdown('<div class="lbl">robot simulation · 9 × 9 grid · each spoken unit = one cell</div>', unsafe_allow_html=True)
    plot_ph = st.empty()
    info_ph = st.empty()
    if ss.pending:
        units, direction = ss.pending
        ss.pending = None
        path, blocked = plan_move(units, direction)
        start = ss.pos
        target = path[-1] if path else start
        pos = start
        plot_ph.plotly_chart(robot_fig(pos, ss.trail, target), use_container_width=True, key="sim_0")
        for k, cell in enumerate(path):
            time.sleep(0.35)
            pos = cell
            ss.trail.append(cell)
            plot_ph.plotly_chart(robot_fig(pos, ss.trail[-16:], target), use_container_width=True, key=f"sim_{k + 1}")
        ss.pos = pos
        ss.trail = ss.trail[-16:]
        ss.rlog.append(f"<b>{units} {ARROW[direction]}</b> {direction}: {start} → {pos}" + (" · <span style='color:#ff7a7a'>wall</span>" if blocked else ""))
    else:
        plot_ph.plotly_chart(robot_fig(ss.pos, ss.trail), use_container_width=True, key="sim_idle")
    c1, c2 = st.columns([3, 1])
    c1.markdown(f'<div class="log">position <b>{ss.pos}</b><br>' + "<br>".join(ss.rlog[-6:][::-1]) + "</div>", unsafe_allow_html=True)
    if c2.button("reset robot"):
        ss.pos, ss.trail, ss.rlog, ss.pending = START, [START], [], None
        st.rerun()
    if ss.history:
        with st.expander(f"session log · {len(ss.history)} commands"):
            hd = pd.DataFrame(ss.history)
            st.dataframe(hd, hide_index=True, use_container_width=True)
            ok = hd[hd["units"] != "—"]
            ok = ok[ok["direction"] != "—"]
            if len(ok):
                ct = pd.crosstab(ok["units"], ok["direction"]).reindex(index=UNITS, columns=DIRS, fill_value=0)
                fig = go.Figure(go.Heatmap(z=ct.values, x=list(ct.columns), y=list(ct.index), text=ct.values, texttemplate="%{text}",
                                           colorscale=[[0, BG], [1, ACCENT]], showscale=False))
                fig.update_yaxes(autorange="reversed", title="units (voice)").update_xaxes(title="direction (gesture)")
                st.plotly_chart(style(fig, 260), use_container_width=True, key="cmdmat")

st.markdown('<div class="foot">Data: Google Speech Commands v0.02 (Warden 2018) · UCI "EMG data for gestures" (Lobov et al. 2018). '
            'Models: SVM / Random Forest / k-NN (scikit-learn) · OpenCV hand tracking. Camera-gesture models start synthetic-trained; add your own clips to adapt them.</div>', unsafe_allow_html=True)
