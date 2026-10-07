"""
ASL Translator — main entry point.

Presents a simple numbered menu:
  1. Collect gesture data
  2. Train the model
  3. Run real-time translator
  4. Quit
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


# ── Status helpers ────────────────────────────────────────────────────────────

def _model_ready() -> bool:
    return (
        Path("models/gesture_model.pkl").exists()
        and Path("models/classes.json").exists()
    )


def _data_info() -> dict:
    """Return number of gesture classes and total samples collected."""
    root = Path("data/landmarks")
    if not root.exists():
        return {"classes": 0, "samples": 0}

    import numpy as np

    class_dirs = [d for d in root.iterdir() if d.is_dir()]
    total = 0
    for cd in class_dirs:
        for f in cd.glob("batch_*.npy"):
            try:
                total += len(np.load(f))
            except Exception:
                pass

    return {"classes": len(class_dirs), "samples": total}


# ── Banner ────────────────────────────────────────────────────────────────────

def _print_banner() -> None:
    console.print()
    console.print(
        Panel(
            Text(
                "  ASL TRANSLATOR\n"
                "  Real-Time Sign Language → Text + Speech",
                style="bold white",
            ),
            border_style="green",
            padding=(0, 2),
        )
    )
    console.print()

    data = _data_info()
    model_ok = _model_ready()

    # Data status
    if data["classes"] > 0:
        console.print(
            f"  Data   [green]✓[/green]  "
            f"[green]{data['classes']} gesture classes  ·  {data['samples']} samples[/green]"
        )
    else:
        console.print("  Data   [red]✗[/red]  [red]No data yet — start with option 1[/red]")

    # Model status
    if model_ok:
        import json
        classes = json.loads(Path("models/classes.json").read_text())
        console.print(
            f"  Model  [green]✓[/green]  "
            f"[green]Ready  ·  {len(classes)} classes[/green]"
        )
    else:
        console.print("  Model  [yellow]✗[/yellow]  [yellow]Not trained yet[/yellow]")

    console.print()


# ── Menu ──────────────────────────────────────────────────────────────────────

def main() -> None:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    _print_banner()

    console.print("  [bold]1[/bold]  Collect gesture data")
    console.print("  [bold]2[/bold]  Train the model")
    console.print("  [bold]3[/bold]  Run real-time translator")
    console.print("  [bold]4[/bold]  Quit")
    console.print()

    choice = input("  Choice → ").strip()

    # ── Option 1 : collect ────────────────────────────────────────────────
    if choice == "1":
        from data.collector import run_collector
        run_collector()

    # ── Option 2 : train ──────────────────────────────────────────────────
    elif choice == "2":
        data = _data_info()
        if data["classes"] < 2:
            console.print(
                "\n  [red]You need data for at least 2 gesture classes.[/red]\n"
                "  Run option 1 first."
            )
            return
        import train as tr
        tr.train()

    # ── Option 3 : run ────────────────────────────────────────────────────
    elif choice == "3":
        if not _model_ready():
            console.print(
                "\n  [yellow]Model not found.[/yellow] "
                "Train the model first (option 2)."
            )
            return

        from core.predictor import GesturePredictor
        from core.tts import TextToSpeech
        from dashboard.display import Dashboard

        console.print("\n  [green]Loading model…[/green]")
        predictor = GesturePredictor(config)

        use_tts = input("  Enable text-to-speech? [y/N]  ").strip().lower() == "y"
        tts = TextToSpeech(config["tts"]["language"]) if use_tts else None

        console.print("  [green]Starting dashboard…[/green]\n")
        Dashboard().run(predictor, tts)

    # ── Option 4 : quit ───────────────────────────────────────────────────
    elif choice == "4":
        console.print("\n  [dim]Goodbye.[/dim]\n")
        sys.exit(0)

    else:
        console.print("\n  [red]Invalid choice.[/red] Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()