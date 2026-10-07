"""
Real-time gesture predictor — version 2.
Gère deux types de gestes :
  - Lettres   (A-Z, SPACE, DELETE, NOTHING) → épellation mot par mot
  - Mots      (HELLO, THANK_YOU, …)         → ajoutés directement à la phrase
"""
from __future__ import annotations

import time
from collections import Counter, deque
from typing import Optional, Tuple

import numpy as np

from core.hand_detector import HandDetector
from core.model import GestureModel

# Gestes de contrôle (pas des lettres, pas des mots)
_CONTROL = {"SPACE", "DELETE", "NOTHING"}

# Lettres simples
_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _is_word(gesture: str) -> bool:
    """True si le geste est un mot entier (ex: HELLO, THANK_YOU)."""
    return gesture not in _CONTROL and gesture not in _LETTERS


class GesturePredictor:
    def __init__(self, config: dict):
        cfg_inf = config["inference"]
        self._threshold: float    = cfg_inf["confidence_threshold"]
        self._pause_secs: float   = cfg_inf["sentence_pause_seconds"]
        buf_size: int             = cfg_inf.get("smoothing_buffer", 10)

        self._detector = HandDetector()
        self._model    = GestureModel(config)
        self._model.load("models/gesture_model.pkl", "models/classes.json")

        self._buffer: deque[str]       = deque(maxlen=buf_size)
        self._letters: list[str]       = []   # lettres en cours de frappe
        self._words:   list[str]       = []   # mots validés
        self._last_gesture: Optional[str] = None
        self._last_added_gesture: Optional[str] = None   # anti-répétition
        self._no_hand_since: float     = time.time()

    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> Tuple:
        frame, landmarks = self._detector.detect(frame)

        gesture:    Optional[str] = None
        confidence: float         = 0.0

        if landmarks is not None:
            self._no_hand_since = time.time()
            raw, confidence = self._model.predict(landmarks)

            if confidence >= self._threshold:
                gesture = self._smooth(raw)
                self._handle(gesture)
        else:
            # Auto-commit lettre courante après pause
            if (time.time() - self._no_hand_since > self._pause_secs
                    and self._letters):
                self._commit_word()
            self._last_gesture = None
            self._last_added_gesture = None

        return (
            frame,
            gesture,
            confidence,
            self._word_str(),
            self._sentence_str(),
        )

    def clear(self) -> None:
        self._letters.clear()
        self._words.clear()
        self._last_gesture = None
        self._last_added_gesture = None
        self._buffer.clear()

    def release(self) -> None:
        self._detector.release()

    # ------------------------------------------------------------------
    # Logique de phrase
    # ------------------------------------------------------------------

    def _handle(self, gesture: str) -> None:
        # Ignorer si même geste que la frame précédente
        if gesture == self._last_gesture:
            return
        self._last_gesture = gesture

        if gesture == "NOTHING":
            return
        elif gesture == "SPACE":
            self._commit_word()
        elif gesture == "DELETE":
            if self._letters:
                self._letters.pop()
            elif self._words:
                # DELETE sur un mot entier → retire le dernier mot
                last = self._words.pop()
                # Si c'était une lettre on remet les lettres dans le buffer
                if len(last) == 1:
                    self._letters = list(last)
        elif _is_word(gesture):
            # Mot entier → commit les lettres en cours + ajoute le mot
            self._commit_word()
            display = gesture.replace("_", " ").lower().capitalize()
            self._words.append(display)
        else:
            # Lettre simple
            self._letters.append(gesture)

    def _commit_word(self) -> None:
        if self._letters:
            self._words.append("".join(self._letters))
            self._letters.clear()

    # ------------------------------------------------------------------

    def _smooth(self, gesture: str) -> str:
        self._buffer.append(gesture)
        if len(self._buffer) == self._buffer.maxlen:
            return Counter(self._buffer).most_common(1)[0][0]
        return gesture

    def _word_str(self) -> str:
        return "".join(self._letters)

    def _sentence_str(self) -> str:
        parts = list(self._words)
        if self._letters:
            parts.append("".join(self._letters))
        return " ".join(parts)