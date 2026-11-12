# GestVox — control a robot with your voice and hand (MLSP PBL)

**Say how far. Swipe which way. The robot obeys.**
GestVox is a multimodal human-robot interaction prototype. Your **voice** gives the distance ("three" → 3 cells) and a **hand gesture** seen by the webcam gives the direction (swipe left / right / up / down). A simulated robot on a 9×9 grid executes the combined command. If either input is unsure, the command is **rejected** instead of moving the robot wrongly.

Live demo: *(add the Streamlit Cloud URL after deploying)*

---

## How it works

```
mic ──► band-pass + VAD ──► MFCC ──► SVM/RF/k-NN ──► units (1–5) ──┐
                                                                     ├─► confidence gate ─► robot moves N cells
webcam ► skin+motion mask ► contour/trajectory ► SVM/RF/k-NN ► direction ─┘
```

1. Press **start** in the *live session* tab and allow camera + microphone.
2. Say one number (**one–five**) while swiping your hand. Recording lasts **at most 2 s** and **stops early** once the spoken word ends (never before 1.2 s, so the swipe is captured).
3. The browser sends 16 kHz audio and ~10 fps 160×120 frames to Python. Everything on screen (levels, SNR, hand-detection %, trajectories, probabilities, latency) is **computed from your capture**.
4. Voice → units, gesture → direction. A **confidence gate** accepts the command only if both top probabilities are ≥ the threshold slider (default 0.45) and speech and a hand were actually found.
5. The right half of the screen animates the robot cell by cell (walls stop it; the log says so).

### Interaction model: complementary fusion + gating
The two modalities fill **different slots** of one command (units vs direction) rather than voting on the same label. The research contribution is the **confidence-gated complementary fusion**: it trades a little coverage for fewer wrong robot moves. The *joint study* tab shows joint accuracy under noise and a gate-threshold sweep.

---

## Datasets (public)

| Modality | Dataset | Source | Use |
|---|---|---|---|
| Voice (units) | **Google Speech Commands v0.02** (Warden, 2018) — words *one, two, three, four, five*, ~19k clips (4,000 used: 3,000 train / 1,000 test) | <http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz> | train + evaluate voice models, live demo |
| Gesture (direction, offline benchmark) | **UCI "EMG data for gestures"** (Lobov et al., 2018) — 8-channel Myo armband EMG @ 200 Hz, 36 subjects. Gestures mapped to directions: radial dev. = left, ulnar dev. = right, wrist ext. = up, wrist flex. = down | <https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures> | rigorous subject-independent benchmark + *dataset replay* tab |
| Gesture (live) | OpenCV camera clips. Starter models are trained on **synthetic hands** (rendered, then passed through the same OpenCV pipeline). Your own saved clips can be added | generated / user-recorded | live camera gesture |

---

## Signal-processing techniques

**Voice**
- Pre-emphasis (0.97)
- Butterworth band-pass 80–4000 Hz (zero-phase, `scipy.signal`)
- Energy-based voice-activity detection (silence trimming)
- STFT → mel filterbank → log → DCT = **MFCC** (13) + delta, with cepstral mean normalisation and pooling over 4 time segments (keeps the temporal order of the word)

**Gesture – EMG (offline)**
- Butterworth band-pass 10–90 Hz + 50 Hz mains notch
- Windowing (1 s) and RMS envelope
- Time-domain features (MAV, RMS, waveform length, zero crossings, slope-sign changes) and spectral features (mean/median frequency via Welch PSD)

**Gesture – camera (live, classical OpenCV, no deep learning)**
- Haar face detection → face region masked out
- YCrCb skin segmentation + morphology (open/close)
- **Frame-difference motion mask: hand = skin AND moving**, which removes the static face, neck and arm
- Largest contour, convex hull and convexity defects (finger count, solidity)
- Centroid trajectory → net displacement (dx, dy), path length, motion energy

## Machine-learning models (compared per modality)
SVM (RBF, probability-calibrated), Random Forest, k-NN, with `StandardScaler` pipelines. Metrics: accuracy, macro precision / recall / F1, confusion matrix, inference latency, and a DSP on/off ablation.

## Results (held-out test sets)

| Task | SVM | Random Forest | k-NN |
|---|---|---|---|
| Voice → units (5 classes) | **93.6 %** | 91.6 % | 90.9 % |
| Voice, without DSP chain | 87.2 % | 82.6 % | 81.0 % |
| EMG → direction (4 classes, subject-independent) | 78.9 % | **81.3 %** | 77.4 % |
| Camera → direction (synthetic test clips) | 96.7 % | 98.0 % | **98.0 %** |

- The voice DSP chain (band-pass + VAD + CMN) gives about +6 to +10 points. For EMG the filters make little difference on this data. This is reported as measured.
- **Joint command accuracy** (units **and** direction both correct, voice and EMG paired at random): **78.0 %** clean. With noise on voice at 20 / 10 / 5 / 0 dB SNR: 75.7 / 53.7 / 28.1 / 19.4 %. With noise on EMG: 76.7 / 59.3 / 37.6 / 26.9 %.
- **Gate at 0.45:** 97.6 % of commands executed at 79.5 % accuracy, avoiding 9 % of wrong moves. Raising the threshold increases that protection at the cost of coverage (full curve in the app).

## Honest limitations
- The robot is **simulated**, not hardware.
- Offline voice/EMG pairs are matched by random sampling, not recorded together (synthetic pairing).
- The camera classifier is trained on **synthetic** hands, so the 97–98 % figure does not transfer to real hands. For real use, record your own swipes (see below). The UCI-EMG result is the rigorous gesture benchmark.
- Skin segmentation is lighting-dependent. Use good light, a plain background, and swipe clearly and quickly in front of your face.
- A model can be confidently wrong, so the gate reduces errors but does not remove them.

---

## Project layout

| File | Purpose |
|---|---|
| `pipeline.py` | **Main deliverable.** Data download, DSP, features, model training/evaluation, joint noise study, gating, camera-model training. CLI: `--download`, `--train`, `--train-cam` |
| `vision.py` | OpenCV hand analysis (skin + motion mask, contour, trajectory features) and synthetic clip generator |
| `app.py` | Streamlit app (live session, dataset replay, signal chain, models, joint study, robot simulation) |
| `components/capture/index.html` | Browser camera + microphone recorder (2 s max, voice-activity auto-stop) |
| `models/` | Trained models (`voice`, `gesture`, `cam_gesture` `.joblib`) and `results.json` |
| `assets/samples/` | Demo voice clips and EMG windows for the replay tab |

## Run locally

Windows (PowerShell), from the project folder:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

Open <http://localhost:8501> in **Chrome/Edge/Brave** and allow camera + microphone. Trained models are committed, so no training is needed. The app reloads models automatically when their files change.

### Re-train from scratch (optional)
```powershell
.venv\Scripts\python pipeline.py --download --train   # ~2.3 GB one-time stream of Speech Commands (keeps only 5 words) + UCI EMG; ~3 min training
.venv\Scripts\python pipeline.py --train-cam          # camera-gesture classifier
```

### Improve the camera gesture with your own data
In the *live session* tab, after a capture, open **improve the gesture model with your own data**, choose the swipe you actually performed, and **save this clip as a training sample** (aim for ~5 per direction). Then press **retrain camera models now**. Saved clips go to `assets/cam_real.npz`. Commit that file to keep them (Streamlit Cloud does not persist saved files).

## Deploy (Streamlit Community Cloud)
Vercel cannot host a Streamlit server, so use Streamlit Community Cloud: push to GitHub → <https://share.streamlit.io> → select this repo and `app.py` → deploy. Camera/mic work over HTTPS. Python packages are in `requirements.txt` (OpenCV is pinned to 4.11 because 5.x removed the face-detector API used here).

## Submission
Upload `pipeline.py` (self-contained training/evaluation implementation) to the course Drive folder, together with `vision.py` if the evaluator needs the camera part.
