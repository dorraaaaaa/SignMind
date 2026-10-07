"""
Hand detector — MediaPipe Tasks API.
Compatible avec mediapipe 0.10.x et versions plus récentes.

L'ancienne API (mp.solutions.hands) a été supprimée dans les versions récentes.
Ce module utilise la nouvelle Tasks API et dessine le squelette via OpenCV pur,
sans dépendance sur les utilitaires de dessin de mediapipe.

Premier lancement : télécharge automatiquement hand_landmarker.task (~5 Mo).
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# ── Modèle ────────────────────────────────────────────────────────────────────
_MODEL_DIR  = Path("models")
_MODEL_PATH = _MODEL_DIR / "hand_landmarker.task"
_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# ── Connexions entre les 21 landmarks (squelette de la main) ──────────────────
_CONNECTIONS = [
    # Pouce
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Majeur
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Annulaire
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Auriculaire
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Paume
    (5, 9), (9, 13), (13, 17),
]


def _ensure_model() -> str:
    """Télécharge hand_landmarker.task si absent."""
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not _MODEL_PATH.exists():
        print("\n  Téléchargement du modèle de détection de mains (~5 Mo)…")
        try:
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
            print("  ✓ Modèle téléchargé\n")
        except Exception as exc:
            raise RuntimeError(
                f"\n[ERREUR] Impossible de télécharger le modèle.\n"
                f"  Vérifie ta connexion internet et réessaie.\n"
                f"  URL : {_MODEL_URL}\n"
                f"  Détail : {exc}"
            )
    return str(_MODEL_PATH)


class HandDetector:
    """
    Détecteur de main basé sur MediaPipe Tasks API.

    Usage
    -----
    detector = HandDetector()
    frame, landmarks = detector.detect(frame)   # landmarks: ndarray(63,) ou None
    detector.release()
    """

    def __init__(
        self,
        max_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
    ):
        import mediapipe as mp

        model_path = _ensure_model()

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._mp  = mp
        self._ts  = 0  # timestamp monotone en ms

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray):
        """
        Détecte une main dans `frame`, dessine le squelette, retourne les landmarks.

        Parameters
        ----------
        frame : np.ndarray  — image BGR d'OpenCV

        Returns
        -------
        (frame_annoté, landmarks | None)
        landmarks : np.ndarray(63,) — 21 points × (x,y,z), normalisés
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        # Timestamp strictement croissant (requis par le mode VIDEO)
        self._ts = max(int(time.time() * 1000), self._ts + 1)
        result = self._landmarker.detect_for_video(mp_img, self._ts)

        if not result.hand_landmarks:
            return frame, None

        hand_lm = result.hand_landmarks[0]   # première main détectée

        self._draw_skeleton(frame, hand_lm)
        return frame, self._normalise(hand_lm, frame.shape)

    def release(self) -> None:
        """Libère les ressources MediaPipe."""
        self._landmarker.close()

    # ------------------------------------------------------------------
    # Dessin — OpenCV pur, pas de dépendance sur mediapipe.solutions
    # ------------------------------------------------------------------

    def _draw_skeleton(self, frame: np.ndarray, hand_lm: list) -> None:
        h, w = frame.shape[:2]
        pts  = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]

        # Connexions (os)
        for a, b in _CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 210, 90), 2, cv2.LINE_AA)

        # Articulations (joints)
        for i, (cx, cy) in enumerate(pts):
            r = 6 if i == 0 else 4          # poignet légèrement plus grand
            cv2.circle(frame, (cx, cy), r, (255, 255, 255), -1)
            cv2.circle(frame, (cx, cy), r, (0, 170, 70),    1)

    # ------------------------------------------------------------------
    # Normalisation des landmarks
    # ------------------------------------------------------------------

    def _normalise(self, hand_lm: list, frame_shape: tuple) -> np.ndarray:
        """
        Convertit les landmarks en vecteur normalisé (63,).

        1. Coordonnées pixel (x*w, y*h, z*w)
        2. Soustraction du poignet (landmark 0) → invariant en position
        3. Division par max(|coords|) → invariant en taille / distance caméra
        4. Aplatissement → (63,)
        """
        h, w, _ = frame_shape
        pts = np.array(
            [[lm.x * w, lm.y * h, lm.z * w] for lm in hand_lm],
            dtype=np.float32,
        )  # (21, 3)

        pts -= pts[0]                # poignet à l'origine

        scale = np.max(np.abs(pts))
        if scale > 0:
            pts /= scale             # normalisation de taille

        return pts.flatten()         # (63,)