# ASL Translator

Real-time **American Sign Language → Text + Speech** translator built with Python, MediaPipe, and TensorFlow — 100% free and open-source.

> ✋ Show a hand sign to the camera. The model recognises the gesture, builds a sentence letter by letter, and reads it aloud.

---

## Demo

[ Camera window ]                     [ Terminal dashboard ]

┌──────────────────────┐              ═══════════════════════════════════
│  B   92%             │              ASL TRANSLATOR  ◆  00:42
│  ████████████░░░░░░  │              ───────────────────────────────────
│                      │              Current gesture    Stats
│  [hand skeleton]     │              B  92%             Frames   1240
│                      │              ████████░░░        Detections 87
│ HELLO WORLD▌         │              Typing: WORLD      Words       2
└──────────────────────┘              FPS      28
                                      ───────────────────────────────────
                                      Sentence
                                        HELLO WORLD▌
                                      [S=speak  C=clear  Q=quit]

---

## Features

- **Live hand tracking** — MediaPipe detects 21 3D landmarks per hand at 30 fps
- **Custom neural network** — dense network trained on *your* gesture data
- **Sentence builder** — spell words letter by letter; `SPACE` commits a word, `DELETE` removes the last letter
- **Text-to-speech** — Google TTS reads the sentence aloud (free, requires internet)
- **Rich terminal dashboard** — live stats, confidence bar, and full sentence display
- **100% free stack** — no paid APIs, no cloud subscriptions

---

## Setup

### 1 — Clone & enter the project

```bash
git clone https://github.com/YOUR_USERNAME/asl-translator
cd asl-translator
```

### 2 — Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ TensorFlow requires Python 3.8–3.11. MediaPipe requires Python ≤ 3.11 on Windows.

---

## Workflow

The project has **3 phases** that you run in order:

Phase 1 → Collect data     python main.py  → 1
Phase 2 → Train model      python main.py  → 2    (~5 min)
Phase 3 → Run translator   python main.py  → 3

### Phase 1 — Collect gesture data

Run the collector and follow on-screen instructions:

```bash
python main.py   # then press 1
```

- A webcam window opens with a live hand skeleton overlay
- Press **SPACE** to start a 3-second countdown, then hold each sign steady
- The script records **100 landmark vectors per batch**
- Aim for **at least 3–5 batches per gesture** (300–500 samples)

**Recommended order:** Start with `A B C` to verify everything works, then collect the full alphabet + `SPACE` + `DELETE` + `NOTHING`.

### Phase 2 — Train the model

```bash
python main.py   # then press 2
# or directly:
python train.py
```

- Typical training time: **3–7 minutes** on CPU
- Target validation accuracy: **95–99%** with 300+ samples per class
- Training curves are saved to `models/training_curves.png`

### Phase 3 — Run the translator

```bash
python main.py   # then press 3
```

- Two windows open: the **camera feed** and the **terminal dashboard**
- Sign letters one by one — the model reads them live
- Use `SPACE` gesture to finish a word, `DELETE` to backspace
- Press **S** to hear the sentence, **C** to clear, **Q** to quit

---

## Project structure

asl-translator/
├── core/
│   ├── hand_detector.py    # MediaPipe wrapper — extracts 63-feature landmark vector
│   ├── model.py            # Keras model — build / save / load / predict
│   ├── predictor.py        # Real-time inference + smoothing + sentence builder
│   └── tts.py              # Text-to-speech with gTTS + pygame
├── data/
│   ├── collector.py        # Interactive webcam gesture recorder
│   └── landmarks/          # Saved .npy batches (one folder per gesture)
├── dashboard/
│   └── display.py          # Rich terminal UI + OpenCV camera overlay
├── models/                 # Trained model saved here after training
├── train.py                # Full training pipeline
├── main.py                 # Central entry point
├── config.yaml             # All settings (model, data, inference, TTS)
└── requirements.txt

SignMind/
├── models/
│   └── hand_landmarker.task  ← détecteur de mains
│
└── data/
    └── landmarks/
        ├── A/
        │   └── batch_0000.npy  ← SUPPRIME ( si mal enregistré )
        ├── B/
        │   └── batch_0000.npy  ← pareil
        └── ...

## Model architecture

Input (63)
  → Dense(128) → BatchNorm → Dropout(0.3)
  → Dense(64)  → BatchNorm → Dropout(0.3)
  → Dense(32)              → Dropout(0.15)
  → Dense(N, softmax)

- **Input:** 63 features = 21 hand landmarks × (x, y, z), translated to wrist-origin and scale-normalised
- **Optimiser:** Adam (lr = 0.001, with ReduceLROnPlateau)
- **Early stopping** on validation accuracy (patience = 15 epochs)

---

## Tech stack

| Library | Role | Cost |

| [MediaPipe](https://mediapipe.dev) | Hand landmark detection | Free |
| [TensorFlow / Keras](https://tensorflow.org) | Model training & inference | Free |
| [OpenCV](https://opencv.org) | Webcam capture & frame annotation | Free |
| [Rich](https://github.com/Textualize/rich) | Terminal dashboard | Free |
| [gTTS](https://gtts.readthedocs.io) | Google Text-to-Speech | Free |
| [scikit-learn](https://scikit-learn.org) | Data splitting & label encoding | Free |
| [pygame](https://pygame.org) | Audio playback | Free |

---

## Configuration (`config.yaml`)

| Key | Default | Description |

| `model.epochs` | 100 | Max training epochs (early stopping applies) |
| `model.hidden_layers` | [128, 64, 32] | Neurons per layer |
| `inference.confidence_threshold` | 0.85 | Minimum confidence to accept a prediction |
| `inference.smoothing_buffer` | 10 | Frames used for majority-vote smoothing |
| `inference.sentence_pause_seconds` | 2.0 | Seconds without hand before auto-committing word |
| `tts.language` | en | gTTS language code |

# remarque :
 Ce qui est sauvegardé dans data/landmarks/, c'est uniquement 21 points mathématiques représentant ta main 
 
 # Contenu d'un fichier batch_0000.npy  (ce que tu "enregistres" vraiment)

[ 0.00,  0.00,  0.00,   ←  poignet (origine)
 -0.12,  0.34,  0.01,   ←  base du pouce
  0.08,  0.51,  0.02,   ←  articulation pouce
  ...                   ←  18 autres points
]