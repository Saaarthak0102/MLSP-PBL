"""
Human-Robot Interaction using Voice + Gesture Recognition  (MLSP PBL)
=====================================================================
Single-file implementation: datasets -> signal processing -> features ->
ML models -> comparison -> confidence-weighted late fusion -> noise study.

Datasets (public)
  * VOICE   : Google Speech Commands v0.02 (Warden, 2018)
              http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz
  * GESTURE : EMG data for gestures, UCI ML Repository (Lobov et al., 2018)
              https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures
              (8-channel Myo armband, 200 Hz, 36 subjects)

Signal-processing techniques
  Voice   : pre-emphasis, Butterworth band-pass, energy-based VAD (silence trim),
            STFT -> mel filterbank -> log -> DCT (MFCC) + delta/delta-delta
  Gesture : Butterworth band-pass, 50 Hz notch, windowing/segmentation,
            time-domain (MAV, RMS, WL, ZC, SSC) + spectral (mean/median freq) features

ML models  : SVM (RBF), Random Forest, k-NN  (compared per modality)
Research   : confidence-weighted late fusion of voice+gesture vs single modality
             under increasing noise (SNR sweep).

Usage
  python pipeline.py --download     # fetch datasets into ./data
  python pipeline.py --train        # extract features, train, evaluate, save ./models
"""
import argparse
import glob
import io
import json
import os
import tarfile
import time
import urllib.request
import zipfile

import numpy as np
from scipy import signal
from scipy.fft import dct
from scipy.io import wavfile

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "models")

COMMANDS = ["go", "stop", "left", "right", "up", "down"]
# EMG class id -> command  (2 fist, 3 wrist flexion, 4 wrist extension,
#                           5 radial deviation, 6 ulnar deviation, 7 extended palm)
GESTURE_TO_CMD = {7: "go", 2: "stop", 5: "left", 6: "right", 4: "up", 3: "down"}
GESTURE_NAMES = {"go": "open palm", "stop": "fist", "left": "radial dev.",
                 "right": "ulnar dev.", "up": "wrist ext.", "down": "wrist flex."}

SR = 16000          # voice sample rate
EMG_FS = 200        # gesture sample rate
EMG_WIN = 200       # 1 s window
EMG_STRIDE = 400

URL_SPEECH = "http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz"
URL_EMG = "https://archive.ics.uci.edu/static/public/481/emg+data+for+gestures.zip"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def download_data():
    sp = os.path.join(DATA_DIR, "speech")
    if not os.path.isdir(os.path.join(sp, "go")):
        print("downloading Speech Commands test set (~110MB)...")
        raw = urllib.request.urlopen(URL_SPEECH).read()
        os.makedirs(sp, exist_ok=True)
        tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz").extractall(sp)
    em = os.path.join(DATA_DIR, "emg")
    if not os.path.isdir(em):
        print("downloading UCI EMG gestures...")
        raw = urllib.request.urlopen(URL_EMG).read()
        os.makedirs(em, exist_ok=True)
        zipfile.ZipFile(io.BytesIO(raw)).extractall(em)


def load_voice_dataset(max_per_class=400):
    X, y = [], []
    for ci, cmd in enumerate(COMMANDS):
        files = sorted(glob.glob(os.path.join(DATA_DIR, "speech", cmd, "*.wav")))[:max_per_class]
        for f in files:
            sr, w = wavfile.read(f)
            w = w.astype(np.float32) / 32768.0
            X.append(fix_length(w, SR))
            y.append(ci)
    return np.array(X, dtype=np.float32), np.array(y)


def load_emg_dataset(max_per_class_per_subject=60):
    """Return raw windows (N, EMG_WIN, 8), labels (command idx), subject ids."""
    import pandas as pd
    base = glob.glob(os.path.join(DATA_DIR, "emg", "*", "*"))
    subj_dirs = sorted(d for d in base if os.path.isdir(d))
    if not subj_dirs:
        subj_dirs = sorted(d for d in glob.glob(os.path.join(DATA_DIR, "emg", "*")) if os.path.isdir(d))
    W, y, subj = [], [], []
    for sd in subj_dirs:
        sid = int(os.path.basename(sd))
        counts = {}
        for f in sorted(glob.glob(os.path.join(sd, "*.txt"))):
            df = pd.read_csv(f, sep="\t").dropna()
            sig = df[[f"channel{i}" for i in range(1, 9)]].values.astype(np.float32)
            lab = df["class"].values
            # contiguous runs of the same class
            edges = np.flatnonzero(np.diff(lab)) + 1
            for s, e in zip(np.r_[0, edges], np.r_[edges, len(lab)]):
                c = int(lab[s])
                if c not in GESTURE_TO_CMD:
                    continue
                cmd_i = COMMANDS.index(GESTURE_TO_CMD[c])
                for a in range(s, e - EMG_WIN + 1, EMG_STRIDE):
                    if counts.get(c, 0) >= max_per_class_per_subject:
                        break
                    W.append(sig[a:a + EMG_WIN]); y.append(cmd_i); subj.append(sid)
                    counts[c] = counts.get(c, 0) + 1
    return np.array(W, dtype=np.float32), np.array(y), np.array(subj)


def fix_length(w, n):
    return np.pad(w, (0, n - len(w))) if len(w) < n else w[:n]


# --------------------------------------------------------------------------
# Signal processing - VOICE
# --------------------------------------------------------------------------
def pre_emphasis(x, a=0.97):
    return np.append(x[0], x[1:] - a * x[:-1])


def bandpass(x, lo, hi, fs, order=4):
    sos = signal.butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def vad_trim(x, sr=SR, frame=400, hop=160, thr_ratio=0.08, margin=3):
    """Energy-based voice activity detection: keep the voiced region, re-centre in 1 s."""
    if len(x) < frame:
        return x
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None] + hop * np.arange(n)[:, None]
    en = (x[idx] ** 2).mean(1)
    act = np.flatnonzero(en > thr_ratio * en.max())
    if len(act) == 0:
        return x
    a = max(act[0] - margin, 0) * hop
    b = min(act[-1] + margin, n) * hop + frame
    seg = x[a:b]
    out = np.zeros(sr, dtype=x.dtype)
    seg = seg[:sr]
    s = (sr - len(seg)) // 2
    out[s:s + len(seg)] = seg
    return out


def voice_preprocess(x, use_filter=True):
    if not use_filter:
        return x
    x = pre_emphasis(x)
    x = bandpass(x, 80, 4000, SR)
    return vad_trim(x)


def mel_filterbank(n_mels=26, n_fft=512, sr=SR, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = imel(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        l, c, r = bins[m - 1], bins[m], bins[m + 1]
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / max(c - l, 1)
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / max(r - c, 1)
    return fb


_FB = mel_filterbank()


def mfcc(x, n_mfcc=13, frame=400, hop=160, n_fft=512):
    """STFT -> power spectrum -> mel filterbank -> log -> DCT. Returns (frames, n_mfcc) and mel spec."""
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None] + hop * np.arange(n)[:, None]
    frames = x[idx] * np.hamming(frame)
    power = np.abs(np.fft.rfft(frames, n_fft)) ** 2 / n_fft
    mel_spec = np.log(power @ _FB.T + 1e-10)
    return dct(mel_spec, type=2, axis=1, norm="ortho")[:, :n_mfcc], mel_spec


def voice_features(raw, use_filter=True):
    x = voice_preprocess(raw, use_filter)
    m, _ = mfcc(x)
    d1 = np.gradient(m, axis=0)
    d2 = np.gradient(d1, axis=0)
    f = np.concatenate([m, d1, d2], axis=1)
    return np.concatenate([f.mean(0), f.std(0)]).astype(np.float32)


# --------------------------------------------------------------------------
# Signal processing - GESTURE (EMG)
# --------------------------------------------------------------------------
_EMG_BP = signal.butter(4, [10, 90], btype="band", fs=EMG_FS, output="sos")
_EMG_NOTCH = signal.iirnotch(50, 30, fs=EMG_FS)


def emg_filter(w):
    """w: (T, 8). Band-pass 10-90 Hz then 50 Hz mains notch (zero-phase)."""
    y = signal.sosfiltfilt(_EMG_BP, w, axis=0)
    return signal.filtfilt(*_EMG_NOTCH, y, axis=0)


def emg_envelope(w, win=20):
    """Rectify + moving RMS envelope (for visualisation)."""
    k = np.ones(win) / win
    return np.sqrt(np.apply_along_axis(lambda c: np.convolve(c ** 2, k, "same"), 0, w))


def emg_features(w, use_filter=True):
    x = emg_filter(w) if use_filter else w - w.mean(0)
    x = x * 1e3
    mav = np.abs(x).mean(0)
    rms = np.sqrt((x ** 2).mean(0))
    wl = np.abs(np.diff(x, axis=0)).sum(0)
    thr = 0.01 * rms.max() + 1e-9
    zc = ((x[:-1] * x[1:] < 0) & (np.abs(x[:-1] - x[1:]) > thr)).sum(0)
    d = np.diff(x, axis=0)
    ssc = ((d[:-1] * d[1:] < 0) & ((np.abs(d[:-1]) > thr) | (np.abs(d[1:]) > thr))).sum(0)
    f, p = signal.welch(x, fs=EMG_FS, nperseg=64, axis=0)
    mnf = (f[:, None] * p).sum(0) / (p.sum(0) + 1e-12)
    cum = np.cumsum(p, 0)
    mdf = f[np.argmax(cum >= cum[-1] / 2, axis=0)]
    rel = rms / (rms.sum() + 1e-9)          # spatial muscle-activation pattern (scale invariant)
    return np.concatenate([mav, rms, wl, zc, ssc, mnf, mdf, rel]).astype(np.float32)


# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------
def add_noise(x, snr_db, rng):
    if snr_db is None:
        return x
    p = np.mean(x ** 2) + 1e-12
    n = rng.standard_normal(x.shape).astype(x.dtype) * np.sqrt(p / 10 ** (snr_db / 10))
    return x + n


# --------------------------------------------------------------------------
# Models, metrics, fusion
# --------------------------------------------------------------------------
def make_models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.calibration import CalibratedClassifierCV
    return {
        "SVM (RBF)": make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(C=10, gamma="scale"), cv=3)),
        "Random Forest": make_pipeline(StandardScaler(), RandomForestClassifier(300, n_jobs=-1, random_state=0)),
        "k-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(7, weights="distance")),
    }


def evaluate(model, Xte, yte):
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    t = time.perf_counter()
    pred = model.predict(Xte)
    ms = (time.perf_counter() - t) / len(Xte) * 1000
    p, r, f, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(yte, pred)), "precision": float(p), "recall": float(r),
            "f1": float(f), "latency_ms": float(ms),
            "confusion": confusion_matrix(yte, pred, labels=range(len(COMMANDS))).tolist()}


def entropy_weight(p):
    """Confidence in [0,1]: 1 - normalised entropy. Rows of p are probability vectors."""
    h = -(p * np.log(p + 1e-12)).sum(1) / np.log(p.shape[1])
    return 1 - h


def fuse(pv, pg, mode="confidence"):
    if mode == "voice":
        return pv
    if mode == "gesture":
        return pg
    if mode == "average":
        return (pv + pg) / 2
    wv, wg = entropy_weight(pv)[:, None], entropy_weight(pg)[:, None]
    return (wv * pv + wg * pg) / (wv + wg + 1e-9)


def acc(p, y):
    return float((p.argmax(1) == y).mean())


# --------------------------------------------------------------------------
# Train everything
# --------------------------------------------------------------------------
def train():
    from sklearn.model_selection import train_test_split
    import joblib
    os.makedirs(MODEL_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    results = {"commands": COMMANDS, "gesture_names": GESTURE_NAMES}

    # ---- VOICE ----
    print("[voice] loading + feature extraction")
    Xr, yv = load_voice_dataset()
    idx = np.arange(len(yv))
    tr, te = train_test_split(idx, test_size=0.25, stratify=yv, random_state=0)
    Fv_tr = np.array([voice_features(Xr[i]) for i in tr])
    Fv_te = np.array([voice_features(Xr[i]) for i in te])
    Fv_te_nof = np.array([voice_features(Xr[i], use_filter=False) for i in te])
    Fv_tr_nof = np.array([voice_features(Xr[i], use_filter=False) for i in tr])
    results["voice"] = {"n_train": len(tr), "n_test": len(te), "models": {}, "ablation": {}}
    vmodels = make_models()
    for name, m in vmodels.items():
        m.fit(Fv_tr, yv[tr])
        results["voice"]["models"][name] = evaluate(m, Fv_te, yv[te])
        m2 = make_models()[name].fit(Fv_tr_nof, yv[tr])
        results["voice"]["ablation"][name] = {"with_dsp": results["voice"]["models"][name]["accuracy"],
                                              "without_dsp": evaluate(m2, Fv_te_nof, yv[te])["accuracy"]}
        print(f"  {name:14s} acc={results['voice']['models'][name]['accuracy']:.3f}  "
              f"(no DSP {results['voice']['ablation'][name]['without_dsp']:.3f})")

    # ---- GESTURE ----
    print("[gesture] loading + feature extraction")
    Wr, yg, subj = load_emg_dataset()
    tr_g = subj <= 26
    te_g = subj >= 27       # subject-independent split
    Fg = np.array([emg_features(w) for w in Wr])
    Fg_nof = np.array([emg_features(w, use_filter=False) for w in Wr])
    results["gesture"] = {"n_train": int(tr_g.sum()), "n_test": int(te_g.sum()),
                          "split": "subject-independent (subjects 1-26 train, 27-36 test)",
                          "models": {}, "ablation": {}}
    gmodels = make_models()
    for name, m in gmodels.items():
        m.fit(Fg[tr_g], yg[tr_g])
        results["gesture"]["models"][name] = evaluate(m, Fg[te_g], yg[te_g])
        m2 = make_models()[name].fit(Fg_nof[tr_g], yg[tr_g])
        results["gesture"]["ablation"][name] = {"with_dsp": results["gesture"]["models"][name]["accuracy"],
                                                "without_dsp": evaluate(m2, Fg_nof[te_g], yg[te_g])["accuracy"]}
        print(f"  {name:14s} acc={results['gesture']['models'][name]['accuracy']:.3f}  "
              f"(no DSP {results['gesture']['ablation'][name]['without_dsp']:.3f})")

    # ---- pick best per modality, save ----
    best_v = max(results["voice"]["models"], key=lambda k: results["voice"]["models"][k]["f1"])
    best_g = max(results["gesture"]["models"], key=lambda k: results["gesture"]["models"][k]["f1"])
    results["best"] = {"voice": best_v, "gesture": best_g}
    joblib.dump(vmodels[best_v], os.path.join(MODEL_DIR, "voice.joblib"), compress=3)
    joblib.dump(gmodels[best_g], os.path.join(MODEL_DIR, "gesture.joblib"), compress=3)
    print(f"best voice={best_v}  best gesture={best_g}")

    # ---- FUSION + NOISE STUDY ----
    # pair each voice test clip with a random gesture test window of the same command
    print("[fusion] noise sweep")
    te_idx = np.flatnonzero(te_g)
    pair_g = np.empty(len(te), dtype=int)
    for k, i in enumerate(te):
        pair_g[k] = rng.choice(te_idx[yg[te_idx] == yv[i]])
    ypair = yv[te]
    snrs = [None, 20, 10, 5, 0, -5]
    modes = ["voice", "gesture", "average", "confidence"]
    sweep = {"snr": ["clean" if s is None else s for s in snrs],
             "voice_noisy": {m: [] for m in modes}, "gesture_noisy": {m: [] for m in modes}}
    mv, mg = vmodels[best_v], gmodels[best_g]
    p_g_clean = mg.predict_proba(Fg[pair_g])
    p_v_clean = mv.predict_proba(Fv_te)
    for s in snrs:
        # voice degraded, gesture clean
        Fn = np.array([voice_features(add_noise(Xr[i], s, rng)) for i in te])
        pv = mv.predict_proba(Fn)
        for m in modes:
            sweep["voice_noisy"][m].append(acc(fuse(pv, p_g_clean, m), ypair))
        # gesture degraded, voice clean
        Gn = np.array([emg_features(add_noise(Wr[j], s, rng)) for j in pair_g])
        pg = mg.predict_proba(Gn)
        for m in modes:
            sweep["gesture_noisy"][m].append(acc(fuse(p_v_clean, pg, m), ypair))
        print(f"  SNR={s}: voice-noisy conf-fusion={sweep['voice_noisy']['confidence'][-1]:.3f} "
              f"(voice only {sweep['voice_noisy']['voice'][-1]:.3f}) | gesture-noisy conf-fusion="
              f"{sweep['gesture_noisy']['confidence'][-1]:.3f} (gesture only {sweep['gesture_noisy']['gesture'][-1]:.3f})")
    results["fusion"] = sweep

    # ---- demo samples for the app ----
    sdir = os.path.join(ROOT, "assets", "samples")
    os.makedirs(sdir, exist_ok=True)
    for ci, cmd in enumerate(COMMANDS):
        for k, i in enumerate(te[yv[te] == ci][:3]):
            wavfile.write(os.path.join(sdir, f"{cmd}_{k}.wav"), SR, (Xr[i] * 32767).astype(np.int16))
    sel = []
    for ci in range(len(COMMANDS)):
        sel += list(te_idx[yg[te_idx] == ci][:4])
    np.savez_compressed(os.path.join(sdir, "emg_samples.npz"), windows=Wr[sel], labels=yg[sel])

    with open(os.path.join(MODEL_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("saved models/ + results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--train", action="store_true")
    a = ap.parse_args()
    if a.download:
        download_data()
    if a.train:
        train()
