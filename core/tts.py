"""
Text-to-speech using gTTS (Google, free) + pygame for audio playback.
All speech runs in a daemon thread so it never blocks the main loop.
"""
from __future__ import annotations

import os
import tempfile
import threading

# Both libraries are optional — if not installed we fall back to a print
try:
    import pygame
    _PYGAME_OK = True
except ImportError:
    _PYGAME_OK = False

try:
    from gtts import gTTS
    _GTTS_OK = True
except ImportError:
    _GTTS_OK = False


class TextToSpeech:
    """
    Non-blocking TTS.

    Usage
    -----
    tts = TextToSpeech()
    tts.speak("Hello world")        # returns immediately
    tts.speak("Done", blocking=True)  # waits until audio finishes
    tts.cleanup()                   # call before exit
    """

    def __init__(self, language: str = "en"):
        self.language = language
        self._lock = threading.Lock()
        self.is_speaking = False

        if _PYGAME_OK:
            pygame.mixer.init()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def speak(self, text: str, blocking: bool = False) -> None:
        """Convert `text` to speech and play it."""
        if not text.strip():
            return

        if not (_GTTS_OK and _PYGAME_OK):
            # Graceful fallback: just print
            print(f"[TTS] {text}")
            return

        if blocking:
            self._play(text)
        else:
            threading.Thread(target=self._play, args=(text,), daemon=True).start()

    def cleanup(self) -> None:
        """Release pygame mixer."""
        if _PYGAME_OK:
            pygame.mixer.quit()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _play(self, text: str) -> None:
        """Download MP3 from Google TTS and play through pygame."""
        with self._lock:
            self.is_speaking = True
            tmp_path = None
            try:
                tts = gTTS(text=text, lang=self.language, slow=False)

                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    tmp_path = f.name
                tts.save(tmp_path)

                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(50)

            except Exception as exc:
                print(f"[TTS Error] {exc}")
            finally:
                self.is_speaking = False
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass