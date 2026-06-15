# RGB-Only Eye & Gaze Monitoring Module
## Smart ICU Companion System — Complete Implementation Guide

---

## What This Module Does

Your module runs in **parallel** with the face emotion and pose estimation branches. It takes the same camera feed and extracts six clinical signals:

| Signal | Clinical Value |
|---|---|
| Gaze direction (yaw, pitch) | Consciousness, responsiveness |
| Blink rate & inter-blink interval | Neurological arousal, sedation level |
| PERCLOS | Drowsiness, microsleep detection |
| Pupil-to-Iris Ratio (PIR) | Drug response, intracranial pressure trend |
| Saccade / nystagmus classification | Neurological anomalies |
| HMM conscious state | Awake → Drowsy → Microsleep → Unresponsive |

All outputs feed the XGBoost fusion layer alongside face and pose outputs.

---

## Hardware Requirement (Prototype)

- **Camera:** 1080p @ 60 FPS RGB webcam (e.g. Logitech Brio 4K or Elgato Facecam)
- **Distance:** 50–70 cm from face
- **Why 1080p:** Gives ~90–110 px iris diameter — minimum for reliable EAR and gaze
- **Why 60 FPS:** Blinks are 100–400 ms; at 30 FPS you miss short blinks entirely
- **Aperture:** f/1.8 or wider for dim lighting without motion blur
- **Prototype note:** No oxygen masks, no supine patients — seated healthy subjects only

---

## Stage 1 — Face & Landmark Detection (No Training Needed)

### What runs here
Three models in parallel on every frame.

### Model 1: MediaPipe Face Mesh
- **Input:** RGB frame (any resolution)
- **Output:** 478 landmarks including iris points 468–477
- **Key flag:** `refine_landmarks=True` — activates iris tracking
- **Gives you:** Iris centre, iris radius, pupil centre (for PIR), head pose estimate
- **Install:** `pip install mediapipe`
- **Training:** None. Use as-is.

### Model 2: Dlib 68-point Predictor
- **Input:** RGB frame
- **Output:** 68 landmarks — points 36–41 (left eye), 42–47 (right eye)
- **Why use it:** Cross-validates MediaPipe; more accurate EAR because eye outline is better calibrated
- **Gives you:** 6 points per eye → EAR calculation
- **Install:** `pip install dlib` + download `shape_predictor_68_face_landmarks.dat`
- **Training:** None. Pretrained weights from dlib.net.

### Model 3: RetinaFace
- **Input:** RGB frame
- **Output:** Face bounding box + 5-point landmark + confidence score (0–1)
- **Why use it:** Confidence score tells you when face is occluded (turned head, hands over face)
- **Rule:** If confidence < 0.6 → suppress alerts from that frame, don't fire false positives
- **Install:** `pip install retinaface`
- **Training:** None.

### Output of Stage 1
- 468 landmarks per frame (MediaPipe)
- 6 eye-outline points per eye (Dlib)
- Face confidence score per frame (RetinaFace)

---

## Stage 2 — Gaze Estimation (Fine-Tune Required)

### What runs here
Three CNN/ViT models in parallel. Each outputs gaze angles + uncertainty. Final answer is uncertainty-weighted average.

### Model 1: L2CS-Net
- **Architecture:** ResNet-50 backbone
- **Input:** 224×224 RGB eye-region crop (from Stage 1 bounding box)
- **Output:** (yaw, pitch) in degrees + confidence score
- **Why use it:** Best accuracy on frontal head poses; label distribution learning gives uncertainty estimate
- **Repo:** `github.com/Ahmednull/l2cs-net`
- **Pretrained on:** Gaze360 + MPIIGaze
- **Training for prototype:** Download pretrained weights → test on your webcam → collect 500 frames of yourself looking at a 3×3 dot grid → fine-tune for 10–20 epochs
- **Target accuracy:** < 3° angular error

### Model 2: RT-GENE
- **Architecture:** VGG-16 backbone
- **Input:** Full face image + head pose matrix (from MediaPipe)
- **Output:** (yaw, pitch) in degrees
- **Why use it:** Explicitly normalises for head pose — when subject's head is tilted, L2CS-Net degrades but RT-GENE holds accuracy
- **Repo:** `github.com/Tobias-Fischer/rt_gene`
- **Pretrained on:** MPIIGaze + Gaze360
- **Training for prototype:** Same fine-tuning data as L2CS-Net

### Model 3: GazeTR (Gaze Transformer)
- **Architecture:** Vision Transformer (ViT)
- **Input:** Full face image 224×224
- **Output:** (yaw, pitch) in degrees
- **Why use it:** Attends to full face context — when part of face is blocked, ViT maintains accuracy where CNNs fail
- **Repo:** `github.com/yihuacheng/GazeTR`
- **Pretrained on:** ETH-XGaze
- **Training for prototype:** Use pretrained weights; fine-tune only if accuracy is poor

### Ensemble Formula
```
weight_i = 1 / uncertainty_i
final_yaw = (w1*yaw1 + w2*yaw2 + w3*yaw3) / (w1 + w2 + w3)
final_pitch = same formula for pitch
```
Mean angular error drops from ~4.5° (single model) to ~2.5° (ensemble).

### Dataset for Fine-Tuning
- **Start with:** MPIIGaze (213,659 images, easiest to work with)
- **Scale up:** ETH-XGaze (1M+ images, 110 subjects, extreme head poses)
- **Prototype shortcut:** 500–1000 of your own frames looking at a 3×3 grid is enough for fine-tuning

---

## Stage 3 — Blink & Drowsiness (No Training — Pure Math)

### EAR — Eye Aspect Ratio

**Formula:**
```
EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
```
Where p1–p6 are Dlib points 36–41 for left eye, 42–47 for right.

**Blink rule:** EAR < 0.20 for ≥ 2 consecutive frames = one blink

**Outputs:**
- Blink count per minute
- Inter-blink interval (milliseconds)
- Running EAR value per frame

### PERCLOS — Percentage Eye Closure

**Formula:**
```
PERCLOS = (frames where EAR < 0.20 in last 30s) / (total frames in 30s window)
```

**Alert threshold:** PERCLOS > 0.80 = microsleep / deep sedation → trigger alert

**Implementation:** Rolling window counter, no model needed.

### Datasets for Validation
- **Eyeblink8:** 8 subjects, frame-level blink annotation — github.com/wzs9706/Eyeblink8
- **NTHU-DDD:** 18 subjects, drowsiness detection

---

## Stage 4 — Pupil Ratio / PIR (No Training for Basic, U-Net for Upgrade)

### PIR — Pupil-to-Iris Ratio (Basic, Prototype-Ready)

**What it measures:** Pupil diameter relative to iris diameter in the same frame. Cancels lighting-induced constriction.

**How to compute:**
1. MediaPipe `refine_landmarks=True` gives 4 iris points per eye (landmarks 468–471 left, 473–477 right)
2. Fit a circle to the 4 iris landmarks → get `iris_radius`
3. Use inner iris landmarks for `pupil_radius`
4. `PIR = pupil_radius / iris_radius`

**Normal range:** 0.30–0.60
**Alert — PIR > 0.70:** Suspicious for mydriasis (drug response, intracranial pressure)
**Alert — PIR < 0.20:** Suspicious for miosis

**Important:** Use for trend monitoring only, not absolute clinical diagnosis.

### U-Net Pupil Segmentation (Optional Upgrade — Phase 2)

- **Architecture:** U-Net with EfficientNet-B2 encoder
- **Input:** 128×128 RGB eye crop
- **Output:** Binary pixel-level pupil mask
- **Why upgrade:** More accurate for asymmetric pupils; gives tighter radius measurement than circle-fitting
- **Dataset:** EyeDentify (Kaggle: vijuls/pupildiameterdatasets) — 212,073 webcam eye images with Tobii ground truth
- **Also use:** OpenEDS 2019/2020 (Meta) — 12,759 eye images with pupil/iris masks
- **Loss function:** Binary cross-entropy + Dice loss
- **Augmentation:** Brightness, contrast, blur to simulate lighting variation
- **Training time:** ~50 epochs on GPU

---

## Stage 5 — Saccade & Nystagmus Detection

### Step 1: Lucas-Kanade Optical Flow (No Training)

- **What it does:** Tracks iris centroid pixel position frame-to-frame
- **How:** `cv2.calcOpticalFlowPyrLK()` — track the MediaPipe iris centre point
- **Compute:** `velocity = pixel_displacement / time_between_frames` → convert to °/sec using camera calibration
- **Saccade rule:** velocity > 30°/sec AND duration 20–200 ms = saccade event

### Step 2: Bi-LSTM Saccade Classifier (Training Required)

- **Why not just use thresholds:** Nystagmus is a repetitive back-and-forth pattern visible only over multiple cycles. A per-frame threshold misses it.
- **Architecture:** 2-layer Bi-LSTM, hidden size 128
- **Input per frame:** (velocity_x, velocity_y, EAR) — sequence of 150 frames = 2.5 seconds at 60 FPS
- **Output:** 4-class softmax → [fixation, saccade, nystagmus, smooth_pursuit]
- **Dataset:** GazeCom (natural scene gaze with fixation/saccade labels) — michaeldorr.de/smoothpursuit
- **Training:** Label 150-frame windows, train on 80% subjects, validate on 20%
- **Loss:** Cross-entropy

---

## Stage 6 — Temporal State Modelling (HMM)

### What it does
Models transitions between 5 discrete conscious states over time.

### States
1. AWAKE_ALERT
2. DROWSY
3. MICROSLEEP
4. EYES_CLOSED
5. UNRESPONSIVE

### Input features fed into HMM each second
- EAR (mean over last 60 frames)
- PERCLOS (30-second rolling window)
- Blink rate (blinks per minute)
- Gaze stability score (variance of gaze direction over last 5 seconds)

### Implementation
```python
pip install hmmlearn
from hmmlearn.hmm import GaussianHMM
model = GaussianHMM(n_components=5, covariance_type="full", n_iter=100)
```

### Training
- Record 20–30 sessions of yourself (or volunteers) in front of webcam
- Label each minute: AWAKE, DROWSY, EYES_CLOSED etc.
- Extract (EAR, PERCLOS, blink_rate) per second
- Train on 80% sessions, validate on 20%
- Even 20–30 sessions is sufficient for HMM

### Output
- Current state label (e.g. DROWSY)
- State probability vector [p_awake, p_drowsy, p_microsleep, p_closed, p_unresponsive]
- This vector goes directly into XGBoost fusion layer

---

## Stage 7 — XGBoost Fusion Layer

### What it does
Takes ALL model outputs as a feature vector per frame. Learns which models to trust under which conditions.

### Full Feature Vector
```
[L2CS_yaw, L2CS_pitch, L2CS_confidence,
 RTGENE_yaw, RTGENE_pitch, RTGENE_confidence,
 GazeTR_yaw, GazeTR_pitch, GazeTR_confidence,
 EAR_left, EAR_right, PERCLOS,
 PIR_left, PIR_right,
 blink_rate, inter_blink_interval,
 saccade_velocity, saccade_class (one-hot 4),
 HMM_state (one-hot 5), HMM_p_vector (5 values),
 RetinaFace_confidence]
```

### Training
- Run all Stage 1–6 models on validation data where you know ground truth gaze angle
- Assemble feature vector per frame
- Train XGBoost on 80% frames, validate on 20%
- Label source: have volunteers look at known screen targets (3×3 or 5×5 dot grid)

---

## Full Training Roadmap

### Phase 0 — Week 1–3 (Zero Training, Working Prototype)
- [ ] Install MediaPipe, Dlib, RetinaFace, OpenCV
- [ ] Implement EAR formula (Dlib points 36–47)
- [ ] Implement PERCLOS rolling window
- [ ] Implement PIR using MediaPipe iris landmarks
- [ ] Implement Lucas-Kanade optical flow for iris tracking
- **Result:** Working pipeline, all signals extracting, no ML training done yet

### Phase 1 — Week 4–8 (Fine-Tune Gaze Models)
- [ ] Download L2CS-Net pretrained weights from GitHub
- [ ] Download MPIIGaze dataset
- [ ] Test L2CS-Net on your webcam → measure angular error
- [ ] Collect 500–1000 frames: sit in front of webcam, look at 3×3 dot grid
- [ ] Fine-tune L2CS-Net for 10–20 epochs on your data
- [ ] Repeat for RT-GENE
- [ ] Add GazeTR last (heavier compute)
- [ ] Implement uncertainty-weighted ensemble
- **Result:** < 3° gaze angular error on your camera

### Phase 2 — Week 9–12 (U-Net Pupil Segmentation)
- [ ] Download EyeDentify from Kaggle
- [ ] Download OpenEDS from GitHub (facebookresearch/OpenEDS)
- [ ] Set up U-Net with EfficientNet-B2 encoder (PyTorch)
- [ ] Augment: brightness ±40%, contrast ±30%, Gaussian blur
- [ ] Train for 50 epochs, BCE + Dice loss
- [ ] Replace PIR circle-fitting with U-Net mask output
- **Result:** More accurate pupil radius, especially for asymmetric pupils

### Phase 3 — Week 13–16 (Bi-LSTM Saccade Classifier)
- [ ] Download GazeCom dataset
- [ ] Extract (velocity_x, velocity_y, EAR) sequences
- [ ] Label 150-frame windows as fixation/saccade/nystagmus/pursuit
- [ ] Train 2-layer Bi-LSTM (hidden 128) in PyTorch
- [ ] Validate on held-out subjects
- **Result:** Nystagmus and smooth pursuit detected, not just raw velocity

### Phase 4 — Week 17–18 (HMM State Model)
- [ ] Record 20–30 webcam sessions
- [ ] Manually label each minute as AWAKE/DROWSY/EYES_CLOSED etc.
- [ ] Extract (EAR, PERCLOS, blink_rate) per second
- [ ] Train GaussianHMM (5 states, full covariance) with hmmlearn
- **Result:** Temporal conscious state tracking with clinical labels

### Phase 5 — Week 19–20 (XGBoost Fusion)
- [ ] Run all models on combined validation data
- [ ] Assemble full feature vector per frame (28+ features)
- [ ] Train XGBoost on 80% frames
- [ ] Validate gaze angular error and state classification accuracy
- [ ] Hook outputs into main system fusion engine
- **Result:** Complete module ready for integration

---

## Complete Install List

```bash
# Core vision
pip install mediapipe opencv-python dlib retinaface

# Deep learning
pip install torch torchvision

# Classical ML
pip install xgboost hmmlearn scikit-learn

# Utilities
pip install numpy scipy pandas mlflow

# Dlib extra (Ubuntu)
sudo apt install cmake build-essential
```

---

## Prototype Limitations (State in Report)

1. Dataset (EyeDentify) collected from seated healthy adults aged 21–44 — not ICU patients
2. No supine head pose — real ICU patients lie flat
3. No occlusion from oxygen masks, nasal cannulas, or medical tubes
4. Lighting controlled — real ICU has inconsistent ambient light
5. Absolute pupil diameter in mm not possible on RGB alone — requires IR hardware
6. HMM trained on healthy volunteers — clinical state labels approximate

These limitations define your **future work** section and are entirely acceptable for a prototype.

---

## Key Metrics to Report

| Module | Metric | Target |
|---|---|---|
| Gaze estimation | Mean angular error (°) | < 3° |
| Blink detection | F1 score vs ground truth | > 0.90 |
| PERCLOS | Correlation with EEG drowsiness labels | > 0.80 |
| PIR | MAE vs Tobii pupil diameter | < 0.05 ratio units |
| Saccade classifier | 4-class accuracy | > 85% |
| HMM state | State classification accuracy | > 80% |
