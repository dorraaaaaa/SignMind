"""
Live dashboard combining:
  • A Rich terminal UI (gesture panel, stats, sentence bar)
  • An OpenCV window with the annotated camera feed

Controls (in the OpenCV window)
--------------------------------
  Q   Quit
  C   Clear sentence
  S   Speak current sentence with TTS
"""
from __future__ import annotations

import time
from typing import Optional

import cv2
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class Dashboard:
    """
    Orchestrates the main inference loop.

    Usage
    -----
    Dashboard().run(predictor, tts)   # tts can be None to disable speech
    """

    def __init__(self):
        self.console = Console()
        self._start = time.time()
        self._stats = {
            "frames": 0,
            "detections": 0,
            "words": 0,
            "fps": 0.0,
        }

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, predictor, tts=None) -> None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.console.print("[bold red]Cannot open camera.[/bold red]")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        prev_sentence = ""
        fps_timer = time.time()

        self.console.print()
        self.console.print(
            "[bold green]✓ ASL Translator running[/bold green]"
            "  [dim]— switch to the camera window to sign[/dim]"
        )
        self.console.print(
            "[dim]  Camera controls:  Q = quit  ·  C = clear  ·  S = speak[/dim]\n"
        )

        with Live(console=self.console, refresh_per_second=20, screen=False) as live:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                self._stats["frames"] += 1

                # FPS
                now = time.time()
                self._stats["fps"] = 1.0 / max(now - fps_timer, 1e-9)
                fps_timer = now

                # Inference
                frame, gesture, conf, word, sentence = predictor.process_frame(frame)

                # Update stats
                if gesture and gesture not in ("NOTHING", "SPACE", "DELETE"):
                    self._stats["detections"] += 1
                if len(sentence.split()) > len(prev_sentence.split()):
                    self._stats["words"] += 1

                # Camera overlay
                _draw_overlay(frame, gesture, conf, sentence)
                cv2.imshow("ASL Translator  [Q=quit  C=clear  S=speak]", frame)

                # Rich UI
                live.update(
                    self._build_layout(gesture, conf, word, sentence)
                )

                # Auto-speak newly added words (non-blocking)
                if tts and sentence != prev_sentence:
                    new_part = sentence[len(prev_sentence):].strip()
                    if new_part:
                        tts.speak(new_part)
                    prev_sentence = sentence

                # Keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    predictor.clear()
                    prev_sentence = ""
                    self._stats["words"] = 0
                elif key == ord("s") and tts:
                    if sentence:
                        tts.speak(sentence, blocking=False)

        cap.release()
        cv2.destroyAllWindows()
        predictor.release()
        if tts:
            tts.cleanup()

    # ------------------------------------------------------------------
    # Rich layout builder
    # ------------------------------------------------------------------

    def _build_layout(
        self,
        gesture: Optional[str],
        confidence: float,
        word: str,
        sentence: str,
    ) -> Layout:

        elapsed = int(time.time() - self._start)
        mm, ss = divmod(elapsed, 60)

        root = Layout()
        root.split_column(
            Layout(name="header",   size=3),
            Layout(name="body",     size=8),
            Layout(name="sentence", size=5),
        )
        root["body"].split_row(
            Layout(name="gesture", ratio=3),
            Layout(name="stats",   ratio=2),
        )

        # ── Header ──────────────────────────────────────────────────────
        root["header"].update(
            Panel(
                Text(f"  ASL TRANSLATOR  ◆  {mm:02d}:{ss:02d}", style="bold white"),
                style="bold green",
                padding=(0, 1),
            )
        )

        # ── Gesture panel ────────────────────────────────────────────────
        g = Text()
        if gesture and gesture not in ("NOTHING",):
            color = "bold green" if confidence >= 0.9 else "bold yellow"
            g.append(f"\n  {gesture}\n\n", style=color)
            filled = int(confidence * 22)
            bar = "█" * filled + "░" * (22 - filled)
            g.append(f"  {bar}  {confidence:.0%}\n", style="green")
        else:
            g.append("\n  Waiting for hand...\n", style="dim")
            if word:
                g.append(f"\n  Typing: {word}", style="dim green")

        root["gesture"].update(
            Panel(g, title="[bold]Current gesture", border_style="green")
        )

        # ── Stats panel ──────────────────────────────────────────────────
        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        t.add_column("k", style="dim",        no_wrap=True)
        t.add_column("v", style="bold green", no_wrap=True)
        t.add_row("Frames",     str(self._stats["frames"]))
        t.add_row("Detections", str(self._stats["detections"]))
        t.add_row("Words",      str(self._stats["words"]))
        t.add_row("FPS",        f"{self._stats['fps']:.0f}")

        root["stats"].update(
            Panel(t, title="[bold]Stats", border_style="dim")
        )

        # ── Sentence panel ───────────────────────────────────────────────
        s = Text()
        s.append("  ")
        if sentence:
            display = sentence[-70:] if len(sentence) > 70 else sentence
            s.append(display, style="bold white")
            if word:
                s.append(f"  [{word}▌]", style="dim green")
        else:
            s.append("Start signing…", style="dim")

        root["sentence"].update(
            Panel(
                s,
                title="[bold]Sentence",
                border_style="blue",
                subtitle="[dim]S=speak  C=clear  Q=quit",
            )
        )

        return root


# ── Camera overlay helpers ────────────────────────────────────────────────────

def _draw_overlay(
    frame,
    gesture: Optional[str],
    confidence: float,
    sentence: str,
) -> None:
    """Draw semi-transparent info bars on top and bottom of the camera frame."""
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 54), (0, 0, 0), -1)

    if gesture and gesture != "NOTHING":
        color = (0, 255, 100) if confidence >= 0.9 else (0, 200, 255)
        cv2.putText(
            frame, f"{gesture}   {confidence:.0%}",
            (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame, "No gesture detected",
            (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (110, 110, 110), 1, cv2.LINE_AA,
        )

    cv2.putText(
        frame, "Q=quit  S=speak  C=clear",
        (w - 232, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1, cv2.LINE_AA,
    )

    # Bottom bar
    cv2.rectangle(frame, (0, h - 52), (w, h), (0, 0, 0), -1)
    display = (sentence[-58:] if len(sentence) > 58 else sentence) or "Start signing…"
    cv2.putText(
        frame, display,
        (12, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA,
    )