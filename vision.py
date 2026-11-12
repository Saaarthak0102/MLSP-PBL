"""
OpenCV hand-gesture analysis (classical, lightweight - no deep learning).

Pipeline per clip (list of BGR frames, 160x120, user-facing / mirrored):
  1. Haar face detection on a middle frame -> face box masked out (faces are skin too)
  2. YCrCb skin thresholding  -> binary mask
  3. morphology (open/close)  -> remove speckle, fill holes
  4. largest contour          -> hand blob
  5. convex hull + convexity defects -> finger count, solidity
  6. centroid trajectory + frame-difference motion energy
  7. clip-level feature vector  -> ML classifier (see pipeline.py)
"""
import cv2
import numpy as np

W, H = 160, 120
GESTURES = ["swipe_left", "swipe_right", "swipe_up", "swipe_down", "open_palm", "fist"]
GESTURE_TO_CMD = {"swipe_left": "left", "swipe_right": "right", "swipe_up": "up",
                  "swipe_down": "down", "open_palm": "go", "fist": "stop"}
FEATURE_NAMES = ["dx", "dy", "path", "fingers", "solidity", "area", "motion", "detect", "std_x", "std_y"]

_FACE = None


def _face_detector():
    global _FACE
    if _FACE is None:
        _FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _FACE


def decode_jpegs(blobs):
    out = []
    for b in blobs:
        im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            out.append(cv2.resize(im, (W, H)))
    return out


def skin_mask(frame):
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(ycc, (40, 135, 85), (255, 175, 125))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)


def find_faces(frames):
    g = cv2.cvtColor(frames[len(frames) // 2], cv2.COLOR_BGR2GRAY)
    f = _face_detector().detectMultiScale(g, 1.2, 4, minSize=(24, 24))
    return [tuple(int(v) for v in r) for r in f]


def hand_in_frame(frame, faces):
    """Return dict(contour, centroid, area, fingers, solidity, mask) or None."""
    m = skin_mask(frame)
    for (x, y, w, h) in faces:                 # blank out the face (slightly enlarged)
        px, py = int(w * .25), int(h * .35)
        m[max(y - py, 0):y + h + py, max(x - px, 0):x + w + px] = 0
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 0.012 * W * H:
        return None
    mo = cv2.moments(c)
    cx, cy = mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]
    hull_pts = cv2.convexHull(c)
    hull_area = max(cv2.contourArea(hull_pts), 1)
    fingers = 0
    hull_idx = cv2.convexHull(c, returnPoints=False)
    if hull_idx is not None and len(hull_idx) > 3 and len(c) > 3:
        try:
            defects = cv2.convexityDefects(c, hull_idx)
        except cv2.error:
            defects = None
        if defects is not None:
            for s, e, f, d in defects[:, 0]:
                a, b, p = c[s][0], c[e][0], c[f][0]
                ab, ap, bp = np.linalg.norm(a - b), np.linalg.norm(a - p), np.linalg.norm(b - p)
                if ap * bp == 0 or ab == 0:
                    continue
                ang = np.degrees(np.arccos(np.clip((ap ** 2 + bp ** 2 - ab ** 2) / (2 * ap * bp), -1, 1)))
                if ang < 80 and d / 256.0 > 0.12 * np.sqrt(area):    # deep, narrow valley between fingers
                    fingers += 1
            fingers = min(fingers + 1, 5) if fingers else 0
    return dict(contour=c, hull=hull_pts, centroid=(cx / W, cy / H), area=area / (W * H),
                fingers=fingers, solidity=area / hull_area, mask=m)


def analyze(frames):
    """Return (feature_vector, series dict, overlay_frames list). Needs >=3 frames."""
    faces = find_faces(frames) if len(frames) else []
    cents, areas, fings, sols, over, masks = [], [], [], [], [], []
    prev = None
    motion = []
    for fr in frames:
        h = hand_in_frame(fr, faces)
        g = cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        motion.append(float(np.abs(g.astype(np.float32) - prev).mean()) if prev is not None else 0.0)
        prev = g.astype(np.float32)
        ov = fr.copy()
        for (x, y, w, hh) in faces:
            cv2.rectangle(ov, (x, y), (x + w, y + hh), (90, 90, 90), 1)
        if h is None:
            cents.append(None); areas.append(0.0); fings.append(0); sols.append(0.0)
            masks.append(np.zeros((H, W), np.uint8))
        else:
            cents.append(h["centroid"]); areas.append(h["area"]); fings.append(h["fingers"]); sols.append(h["solidity"])
            masks.append(h["mask"])
            cv2.drawContours(ov, [h["hull"]], -1, (120, 255, 178), 1)
            cv2.drawContours(ov, [h["contour"]], -1, (80, 180, 255), 1)
            cv2.circle(ov, (int(h["centroid"][0] * W), int(h["centroid"][1] * H)), 3, (0, 0, 255), -1)
        over.append(ov)
    det = np.array([c is not None for c in cents])
    series = dict(detected=det, area=np.array(areas), fingers=np.array(fings), solidity=np.array(sols),
                  motion=np.array(motion), cx=np.array([c[0] if c else np.nan for c in cents]),
                  cy=np.array([c[1] if c else np.nan for c in cents]), masks=masks)
    return clip_features(series), series, over


def clip_features(s):
    det = s["detected"]
    n = len(det)
    if det.sum() < 3:
        f = np.zeros(len(FEATURE_NAMES), np.float32)
        f[7] = det.mean() if n else 0
        return f
    cx, cy = s["cx"][det], s["cy"][det]
    k = max(len(cx) // 4, 1)
    dx = cx[-k:].mean() - cx[:k].mean()
    dy = cy[-k:].mean() - cy[:k].mean()
    path = np.sum(np.hypot(np.diff(cx), np.diff(cy)))
    return np.array([dx, dy, path, s["fingers"][det].mean(), s["solidity"][det].mean(),
                     s["area"][det].mean(), s["motion"].mean(), det.mean(), cx.std(), cy.std()], np.float32)


# --------------------------------------------------------------------------
# Synthetic clips (render a skin-coloured hand moving) - used to bootstrap training
# --------------------------------------------------------------------------
SKIN_BGR = (120, 160, 205)


def _draw_hand(img, cx, cy, size, fingers_out, col=SKIN_BGR):
    cv2.ellipse(img, (int(cx), int(cy)), (int(size * .8), int(size)), 0, 0, 360, col, -1)
    if fingers_out:
        for i in range(5):
            ang = np.radians(-70 + i * 35)
            x2, y2 = cx + np.sin(ang) * size * 2.1, cy - np.cos(ang) * size * 2.1
            cv2.line(img, (int(cx + np.sin(ang) * size * .5), int(cy - np.cos(ang) * size * .6)),
                     (int(x2), int(y2)), col, max(int(size * .32), 3))


def synth_clip(gesture, rng, n=30):
    bg = rng.integers(40, 110)
    frames = []
    size = rng.uniform(11, 17)
    cx0, cy0 = rng.uniform(.4, .6) * W, rng.uniform(.45, .6) * H
    vec = {"swipe_left": (-1, 0), "swipe_right": (1, 0), "swipe_up": (0, -1), "swipe_down": (0, 1)}.get(gesture, (0, 0))
    span = rng.uniform(.28, .42)
    hand_delay = rng.integers(0, 8)
    col = tuple(int(np.clip(c + rng.normal(0, 14), 0, 255)) for c in SKIN_BGR)
    dist = (rng.uniform(.1, .9) * W, rng.uniform(.1, .9) * H, rng.uniform(4, 10)) if rng.random() < .4 else None
    for t in range(n):
        img = np.full((H, W, 3), bg, np.uint8)
        img = cv2.add(img, rng.integers(0, 14, (H, W, 3), dtype=np.uint8))
        u = (t / (n - 1)) - .5
        cx = cx0 + vec[0] * u * span * W + rng.normal(0, 1.2)
        cy = cy0 + vec[1] * u * span * H * 1.3 + rng.normal(0, 1.2)
        if gesture.startswith("swipe"):
            out = rng.random() < .55           # swipe with open or closed hand
        else:
            out = gesture == "open_palm"
        if dist is not None:
            cv2.circle(img, (int(dist[0]), int(dist[1])), int(dist[2]), col, -1)   # static skin-coloured distractor
        if t >= hand_delay and rng.random() > .08:                              # occasional lost frames
            _draw_hand(img, cx, cy, size * rng.uniform(.9, 1.1), out, col)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        frames.append(img)
    return frames
