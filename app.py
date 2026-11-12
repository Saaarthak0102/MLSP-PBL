"""Streamlit demo: HRI via Voice + Gesture.  Run:  streamlit run app.py"""
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


capture = components.declare_component("capture", path=os.path.join(ROOT, "components", "capture"))


@st.cache_resource
def load_cam():
    return joblib.load(os.path.join(P.MODEL_DIR, "cam_gesture.joblib"))


def move_robot(cmd):
    x, y = st.session_state.pos
    dx, dy = {"left": (-1, 0), "right": (1, 0), "up": (0, 1), "down": (0, -1)}.get(cmd, (0, 0))
    if cmd in ("left", "right", "up", "down"):
        st.session_state.dir = (dx, dy)
    if cmd == "go":
        dx, dy = st.session_state.dir
    x, y = min(6, max(0, x + dx)), min(6, max(0, y + dy))
    st.session_state.pos = (x, y)
    st.session_state.trail = (st.session_state.trail + [(x, y)])[-12:]


def robot_fig():
    tx, ty = zip(*st.session_state.trail)
    fig = go.Figure()
    fig.add_scatter(x=tx, y=ty, mode="lines", line=dict(color=DIM, width=1, dash="dot"), showlegend=False)
    fig.add_scatter(x=[st.session_state.pos[0]], y=[st.session_state.pos[1]], mode="markers",
                    marker=dict(symbol="square", size=26, color=ACCENT), showlegend=False)
    fig.update_xaxes(range=[-.5, 6.5], dtick=1, showticklabels=False)
    fig.update_yaxes(range=[-.5, 6.5], dtick=1, showticklabels=False, scaleanchor="x")
    return style(fig, 280)


def audio_stats(x):
    """Real measurements on the captured audio + start of the most energetic 1 s window."""
    fl, hop = 400, 160
    n = 1 + (len(x) - fl) // hop
    idx = np.arange(fl)[None] + hop * np.arange(n)[:, None]
    en = (x[idx] ** 2).mean(1) + 1e-12
    noise = np.percentile(en, 10)
    thr = max(noise * 8, 0.1 * en.max())
    voiced = en > thr
    sig = en[voiced].mean() if voiced.any() else noise
    cs = np.r_[0, np.cumsum(en)]
    w = P.SR // hop
    best = int(np.argmax(cs[w:] - cs[:-w])) if n > w else 0
    start = min(best * hop, max(len(x) - P.SR, 0))
    return dict(duration=len(x) / P.SR, rms=float(np.sqrt((x ** 2).mean())), peak=float(np.abs(x).max()),
                voiced_pct=float(voiced.mean()), snr_db=float(10 * np.log10(sig / noise)), start=start)


def process_session(data):
    T = {}
    t = time.perf_counter()
    x = np.frombuffer(base64.b64decode(data["audio"]), dtype=np.int16).astype(np.float32) / 32768.0
    frames = V.decode_jpegs([base64.b64decode(f) for f in data["frames"]])
    T["decode"] = time.perf_counter() - t
    # ---- voice: locate word, band-pass/VAD, MFCC, classify
    t = time.perf_counter()
    A = audio_stats(x)
    win = P.fix_length(x[A["start"]:A["start"] + P.SR], P.SR)
    proc = P.voice_preprocess(win, True)
    pv_ok = A["peak"] > 0.02 and A["voiced_pct"] > 0.03
    pv = mv.predict_proba(P.voice_features(win)[None])[0] if pv_ok else np.ones(len(CMDS)) / len(CMDS)
    T["voice DSP + model"] = time.perf_counter() - t
    # ---- gesture: OpenCV
    t = time.perf_counter()
    if len(frames) >= 5:
        feat, S, over = V.analyze(frames)
    else:
        feat, S, over = np.zeros(len(V.FEATURE_NAMES), np.float32), None, []
    T["OpenCV analysis"] = time.perf_counter() - t
    t = time.perf_counter()
    cam = load_cam()
    g_ok = S is not None and S["detected"].mean() >= 0.2
    per_model = {}
    for name, m in cam.items():
        pr = m.predict_proba(feat[None])[0]
        pc = np.zeros(len(CMDS))
        for gi, g in enumerate(V.GESTURES):
            pc[CMDS.index(V.GESTURE_TO_CMD[g])] = pr[gi]
        per_model[name] = pc
    best_cam = load_results()["camera"]["best"]
    pg = per_model[best_cam] if g_ok else np.ones(len(CMDS)) / len(CMDS)
    pf = P.fuse(pv[None], pg[None], "confidence")[0]
    T["classify + fuse"] = time.perf_counter() - t
    return dict(A=A, x=x, win=win, proc=proc, pv=pv, pv_ok=pv_ok, S=S, over=over, feat=feat, pg=pg, g_ok=g_ok,
                per_model=per_model, best_cam=best_cam, pf=pf, T=T, n_frames=len(frames), elapsed=data["elapsed"], id=data["id"])


def render_session(R):
    A, S = R["A"], R["S"]
    vcmd, gcmd = CMDS[int(R["pv"].argmax())], CMDS[int(R["pg"].argmax())]
    none = not R["pv_ok"] and not R["g_ok"]
    fcmd = None if none else CMDS[int(R["pf"].argmax())]
    st.markdown("### result")
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown(f'<div class="lbl">fused command</div><div class="big">{fcmd.upper() if fcmd else "NO INPUT"}</div>', unsafe_allow_html=True)
        parts = [f"Heard **{vcmd}** ({R['pv'].max():.0%})" if R["pv_ok"] else "No speech detected (audio too quiet)",
                 f"saw a gesture for **{gcmd}** ({R['pg'].max():.0%})" if R["g_ok"] else "no hand detected"]
        st.markdown(", ".join(parts) + (f" → **executing {fcmd.upper()}**." if fcmd else "."))
        wv, wg = P.entropy_weight(R["pv"][None])[0], P.entropy_weight(R["pg"][None])[0]
        if R["pv_ok"] and R["g_ok"]:
            if vcmd == gcmd:
                st.markdown("✓ voice and gesture **agree**.")
            else:
                st.markdown(f"⚠ they **disagree** — fusion trusted **{'voice' if wv > wg else 'gesture'}** more (higher confidence).")
        df = pd.DataFrame({"command": CMDS, "voice": R["pv"], "gesture": R["pg"], "fused": R["pf"]}).set_index("command")
        st.dataframe(df, use_container_width=True, column_config={
            c: st.column_config.ProgressColumn(c, min_value=0.0, max_value=1.0, format="%.3f") for c in df.columns})
        st.caption(f"confidence weights (1 − normalised entropy): voice {wv:.2f} · gesture {wg:.2f}")
    with c2:
        st.metric("total processing", f"{sum(R['T'].values()) * 1000:.0f} ms")
        st.dataframe(pd.DataFrame({"stage": list(R["T"]), "ms": [v * 1000 for v in R["T"].values()]}).set_index("stage").style.format("{:.1f}"),
                     use_container_width=True)
        st.plotly_chart(robot_fig(), use_container_width=True, key="robot_sess")

    Hh = st.session_state.get("history", [])
    if Hh:
        st.markdown("#### agreement matrix · all runs this session")
        hd = pd.DataFrame(Hh)
        ct = pd.crosstab(hd["voice"], hd["gesture"])
        ct = ct.reindex(index=CMDS + ["—"], columns=CMDS + ["—"], fill_value=0)
        fig = go.Figure(go.Heatmap(z=ct.values, x=list(ct.columns), y=list(ct.index), text=ct.values, texttemplate="%{text}",
                                   colorscale=[[0, BG], [1, ACCENT]], showscale=False))
        fig.update_yaxes(autorange="reversed", title="voice").update_xaxes(title="gesture")
        st.plotly_chart(style(fig, 320), use_container_width=True, key="agree_mat")
        st.caption("rows = voice prediction · columns = gesture prediction · diagonal = agreement")
        st.dataframe(hd, hide_index=True, use_container_width=True)

    st.markdown("### voice · measured")
    m = st.columns(5)
    m[0].metric("captured", f"{A['duration']:.2f} s")
    m[1].metric("peak", f"{20 * np.log10(A['peak'] + 1e-9):.1f} dBFS")
    m[2].metric("RMS", f"{A['rms']:.4f}")
    m[3].metric("voiced", f"{A['voiced_pct']:.0%}")
    m[4].metric("est. SNR", f"{A['snr_db']:.1f} dB")
    t5 = np.arange(len(R["x"])) / P.SR
    fig = go.Figure()
    fig.add_scatter(x=t5[::8], y=R["x"][::8], line=dict(color=DIM, width=1), name="captured")
    fig.add_vrect(x0=A["start"] / P.SR, x1=A["start"] / P.SR + 1, fillcolor=ACCENT, opacity=.15, line_width=0)
    fig.update_layout(title="capture · shaded = 1 s window classified").update_xaxes(title="s")
    a, b, c = st.columns(3)
    a.plotly_chart(style(fig, 240), use_container_width=True)
    mf, mel = P.mfcc(R["proc"])
    b.plotly_chart(style(go.Figure(go.Heatmap(z=mel.T, colorscale="Viridis", showscale=False)).update_layout(title="log-mel (band-pass + VAD)"), 240), use_container_width=True)
    c.plotly_chart(style(go.Figure(go.Heatmap(z=mf.T, colorscale="Magma", showscale=False)).update_layout(title="MFCC"), 240), use_container_width=True)
    if not R["pv_ok"]:
        st.warning("Audio too quiet / no speech found — voice ignored in fusion.")

    st.markdown("### gesture · measured (OpenCV)")
    if S is None:
        st.warning("Too few video frames received.")
        return
    m = st.columns(5)
    m[0].metric("frames", R["n_frames"])
    m[1].metric("effective fps", f"{R['n_frames'] / max(R['elapsed'], 1e-6):.1f}")
    m[2].metric("hand detected", f"{S['detected'].mean():.0%}")
    m[3].metric("net motion", f"dx {R['feat'][0]:+.2f} dy {R['feat'][1]:+.2f}")
    m[4].metric("mean fingers", f"{R['feat'][3]:.1f}")
    pick = np.linspace(0, len(R["over"]) - 1, 6).astype(int)
    st.image(np.hstack([cv2.cvtColor(R["over"][i], cv2.COLOR_BGR2RGB) for i in pick]),
             caption="face (grey) · hand contour (blue) · convex hull (green) · centroid (red)", width="stretch")
    st.image(np.hstack([S["masks"][i] for i in pick]), caption="skin mask after morphology", width="stretch")
    a, b = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=S["cx"], y=-S["cy"], mode="lines+markers", line=dict(color=ACCENT),
                    marker=dict(size=4, color=np.arange(len(S["cx"])), colorscale="Viridis"))
    fig.update_layout(title="hand centroid trajectory (colour = time)").update_xaxes(range=[0, 1], title="x").update_yaxes(range=[-1, 0], title="y")
    a.plotly_chart(style(fig, 300), use_container_width=True)
    fig = go.Figure()
    fig.add_scatter(y=S["fingers"], name="fingers", line=dict(color=AMBER))
    fig.add_scatter(y=S["area"] * 100, name="area %", line=dict(color="#6fa8ff"))
    fig.add_scatter(y=S["motion"], name="motion energy", line=dict(color=DIM))
    fig.update_layout(title="per-frame series").update_xaxes(title="frame")
    b.plotly_chart(style(fig, 300), use_container_width=True)
    st.markdown("**gesture classifiers on this clip**")
    rows = [{"model": n + (" ★" if n == R["best_cam"] else ""), "command": CMDS[int(pc.argmax())], "confidence": float(pc.max())}
            for n, pc in R["per_model"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={"confidence": st.column_config.NumberColumn(format="%.3f")})
    st.dataframe(pd.DataFrame([R["feat"]], columns=V.FEATURE_NAMES).style.format("{:.3f}"), hide_index=True, use_container_width=True)
    if not R["g_ok"]:
        st.warning("Hand not found in enough frames — gesture ignored in fusion. Try brighter light and a plain background.")

    with st.expander("improve the gesture model with your own data"):
        lab = st.selectbox("gesture you just performed", V.GESTURES)
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
            load_cam.clear(); load_results.clear()
            st.success("retrained — record again to use the new models")


res, (mv, mg) = load_results(), load_models()
CMDS = res["commands"]
rng = np.random.default_rng(int(time.time()) % 1000)

st.markdown('<div class="hdr">HRI/CMD <span>· voice + gesture command fusion · MLSP PBL</span></div>', unsafe_allow_html=True)
t_sess, t_live, t_sig, t_mod, t_fus = st.tabs(["live session", "dataset replay", "signal chain", "models", "fusion study"])

if "pos" not in st.session_state:
    st.session_state.pos, st.session_state.trail, st.session_state.dir = (3, 3), [(3, 3)], (0, 1)

# ----------------------------------------------------------------- LIVE SESSION (camera + mic)
with t_sess:
    data = capture(key="cap", default=None)
    if data and data.get("id") != st.session_state.get("last_id"):
        st.session_state.last_id = data["id"]
        with st.status("processing capture…") as status:
            R = process_session(data)
            st.session_state.sess = R
            fc = None if (not R["pv_ok"] and not R["g_ok"]) else CMDS[int(R["pf"].argmax())]
            if fc:
                move_robot(fc)
            hist = st.session_state.setdefault("history", [])
            hist.append(dict(run=len(hist) + 1, voice=CMDS[int(R["pv"].argmax())] if R["pv_ok"] else "—",
                             gesture=CMDS[int(R["pg"].argmax())] if R["g_ok"] else "—", executed=fc or "—",
                             snr_db=round(R["A"]["snr_db"], 1),
                             hand_pct=round(float(R["S"]["detected"].mean()) if R["S"] else 0, 2)))
            status.update(label="done", state="complete")
    if st.session_state.get("sess"):
        render_session(st.session_state.sess)
    else:
        st.info("No capture yet. Press **start** above — the browser asks for camera + microphone permission the first time.")

# ----------------------------------------------------------------- DATASET REPLAY
with t_live:
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
        move_robot(cmd)

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
        st.markdown('<div class="lbl">robot (7×7 grid)</div>', unsafe_allow_html=True)
        st.plotly_chart(robot_fig(), use_container_width=True, key="robot_live")
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
