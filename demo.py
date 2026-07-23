"""Point d'entrée démo — Groupe 2 OTA (CVPR 2021)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"


def run(cmd: list[str], cwd: Path) -> int:
    print(f"\n>>> {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=cwd)


def main() -> int:
    py = sys.executable
    steps = [
        ([py, "tests/test_geometry.py"], CODE),
        ([py, "tests/test_sinkhorn.py"], CODE),
        ([py, "experiment_ambiguous_anchors.py"], CODE),
    ]
    for step, cwd in steps:
        rc = run(step, cwd)
        if rc != 0:
            print(f"Échec: {step}")
            return rc
    print("\n=== Démo OTA terminée avec succès ===")
    print("Figures : figures/assignment_maps.png, figures/ambiguous_vs_radius.png")
    print("Tableau : tables/ambiguous_anchor_counts.csv")
    print("Rapport : report/rapport.pdf, report/carnet_lecture.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
