# GestVox — control a robot with your voice and hand (MLSP PBL)

Live demo: *(add Streamlit Cloud URL after deploy)*

## Datasets
- **Voice** — Google Speech Commands v0.02 (Warden 2018), 5 number words one-five (full v0.02, ~19k clips). <http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz>
- **Gesture** — UCI *EMG data for gestures* (Lobov et al. 2018), 8-ch Myo EMG @200 Hz, 36 subjects, 4 gestures mapped to directions (radial dev.=left, ulnar dev.=right, wrist ext.=up, wrist flex.=down). <https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures>

## Interaction model
Voice says **how many units** (one-five), gesture gives the **direction** (swipe left/right/up/down). Fusion is complementary (each modality fills one slot) with a confidence gate that rejects the command if either input is unsure. The right half of the screen is a 9x9 robot simulation that animates the move.

## Live session (camera + mic)
Press **start** in the *live session* tab -> allow camera + microphone -> 3 s auto-capture. The browser sends 16 kHz audio and ~10 fps 160x120 frames to Python, which measures and classifies both (all numbers shown are computed from your capture) and fuses them.
Gesture = classical OpenCV: Haar face masking, YCrCb skin segmentation, morphology, convex hull + convexity defects (finger count), centroid trajectory, frame-difference motion -> SVM / RF / k-NN on clip features. Gestures: swipe left/right/up/down.
Note: the starter camera models are trained on synthetic hands rendered and passed through the same OpenCV pipeline; use *improve the gesture model* in the app to add your own labelled clips and retrain. Offline benchmark of the public datasets is in the *models* tab.

## Signal processing
Voice: pre-emphasis, Butterworth band-pass 80–4000 Hz, energy VAD, STFT → mel → log → DCT (MFCC, Δ, ΔΔ).
Gesture: Butterworth band-pass 10–90 Hz, 50 Hz notch, windowing, RMS envelope, MAV/RMS/WL/ZC/SSC + mean/median frequency.

## Models compared (per modality)
SVM (RBF, calibrated), Random Forest, k-NN — accuracy, macro P/R/F1, latency, confusion matrix, DSP ablation. Gesture split is subject-independent (subjects 1–26 train / 27–36 test).

## Research contribution
(Updated) Complementary voice+gesture command (units + direction) with confidence gating; joint-accuracy noise study and gate-threshold curve in the *joint study* tab. Earlier idea, kept for reference:
Confidence-weighted late fusion (weights = 1 − normalised posterior entropy) vs voice-only, gesture-only and plain averaging under an SNR sweep. Voice/gesture pairs are synthetic (matched by command label).

## Run
```
pip install -r requirements.txt
python pipeline.py --download --train   # ~3 min; writes models/ + assets/samples
python pipeline.py --train-cam          # camera-gesture classifier
streamlit run app.py
```
Trained models are committed, so `streamlit run app.py` works directly. Deploy: push to GitHub → share.streamlit.io → select `app.py`.
Submit `pipeline.py` to the course Drive.
