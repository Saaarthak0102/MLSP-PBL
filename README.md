# HRI/CMD — Human-Robot Interaction via Voice + Gesture (MLSP PBL)

Live demo: *(add Streamlit Cloud URL after deploy)*

## Datasets
- **Voice** — Google Speech Commands v0.02 (Warden 2018), test split, 6 commands: go/stop/left/right/up/down. <http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz>
- **Gesture** — UCI *EMG data for gestures* (Lobov et al. 2018), 8-ch Myo EMG @200 Hz, 36 subjects, 6 gestures mapped to the same 6 commands. <https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures>

## Signal processing
Voice: pre-emphasis, Butterworth band-pass 80–4000 Hz, energy VAD, STFT → mel → log → DCT (MFCC, Δ, ΔΔ).
Gesture: Butterworth band-pass 10–90 Hz, 50 Hz notch, windowing, RMS envelope, MAV/RMS/WL/ZC/SSC + mean/median frequency.

## Models compared (per modality)
SVM (RBF, calibrated), Random Forest, k-NN — accuracy, macro P/R/F1, latency, confusion matrix, DSP ablation. Gesture split is subject-independent (subjects 1–26 train / 27–36 test).

## Research contribution
Confidence-weighted late fusion (weights = 1 − normalised posterior entropy) vs voice-only, gesture-only and plain averaging under an SNR sweep. Voice/gesture pairs are synthetic (matched by command label).

## Run
```
pip install -r requirements.txt
python pipeline.py --download --train   # ~3 min; writes models/ + assets/samples
streamlit run app.py
```
Trained models are committed, so `streamlit run app.py` works directly. Deploy: push to GitHub → share.streamlit.io → select `app.py`.
Submit `pipeline.py` to the course Drive.
