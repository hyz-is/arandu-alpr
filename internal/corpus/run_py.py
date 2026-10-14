#!/usr/bin/env python3
"""Produce the Python side of the accuracy corpus.

Run this inside the alpr-py environment, where fast_alpr and its weights are installed:

    python3 internal/corpus/run_py.py \
        --manifest internal/corpus/manifest.json \
        --out internal/corpus/results/py

The two sides must be given the SAME WEIGHTS. The comparator hashes them and refuses to draw a
conclusion when they differ, because a result set produced against different weights is not
stale -- it is meaningless, and it fails in a way that looks like an accuracy regression.

One asymmetry is structural and cannot be removed here. fast_alpr detects and reads in a single
call and filters afterwards, so this side always produces every reading the model produced. The
Go side splits detection from recognition and drops a candidate before the OCR ever sees it.
That is why the comparator has two modes: --gates=off compares model against model, --gates=on
measures what the Go gates discard. Without that distinction every gate looks like a bug.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

try:
    import numpy as np
    from PIL import Image
except ImportError as exc:  # pragma: no cover - environment problem, not a code path
    sys.exit(f"missing dependency: {exc}. Run this inside the alpr-py environment.")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_rgb(path: Path) -> "np.ndarray":
    """Decode a PNG to RGB.

    PNG, never JPEG: a lossy codec makes the two implementations disagree at the decoder, before
    either model has seen a pixel, and that shows up in the report as a preprocessing bug that
    does not exist.
    """
    if path.suffix.lower() != ".png":
        sys.exit(f"{path} is not a PNG; the corpus is lossless so the decode cannot itself "
                 f"become a source of disagreement")
    return np.array(Image.open(path).convert("RGB"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="internal/corpus/manifest.json")
    ap.add_argument("--out", default="internal/corpus/results/py")
    ap.add_argument("--detector", default=os.getenv("DETECTOR_MODEL", "yolo-v9-s-608-license-plate-end2end"))
    ap.add_argument("--ocr", default=os.getenv("OCR_MODEL", "cct-s-v2-global-model"))
    ap.add_argument("--device", default=os.getenv("ALPR_DEVICE", "cpu"))
    args = ap.parse_args()

    try:
        from fast_alpr import ALPR
    except ImportError:
        sys.exit("fast_alpr is not installed. Run this inside the alpr-py environment.")

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpr = ALPR(detector_model=args.detector, ocr_model=args.ocr, ocr_device=args.device)

    alphabet = manifest.get("alphabet", "")
    alphabet_sha = sha16(alphabet) if alphabet else ""

    for frame in manifest["frames"]:
        path = base / frame["file"]
        if frame.get("sha256"):
            actual = file_sha256(path)
            if actual != frame["sha256"]:
                sys.exit(f"{frame['id']}: the frame changed since the manifest was written "
                         f"({actual} vs {frame['sha256']}); a corpus that drifts silently proves nothing")

        results = alpr.predict(load_rgb(path))

        readings = []
        for r in results:
            box = r.detection.bounding_box
            char_probs = list(getattr(r.ocr, "char_probs", []) or [])
            # Mean, not minimum: one weak character in an otherwise certain plate is usually a
            # smudge on one digit, and the Go side takes the mean for the same reason.
            confidence = float(sum(char_probs) / len(char_probs)) if char_probs else 0.0
            readings.append({
                "plate": (r.ocr.text or "").upper().replace("-", "").replace(" ", "").replace("_", ""),
                "char_confidences": [float(p) for p in char_probs],
                "confidence": confidence,
                "region": getattr(r.ocr, "region", "") or "",
                "region_confidence": float(getattr(r.ocr, "region_conf", 0.0) or 0.0),
                "box": [float(box.x1), float(box.y1),
                        float(box.x2 - box.x1), float(box.y2 - box.y1)],
                "detector_score": float(r.detection.confidence),
            })

        # Same ordering rule as the Go side, so a diff is a difference and never an artefact of
        # which candidate came out of NMS first.
        readings.sort(key=lambda r: (-r["detector_score"], r["box"][0], r["box"][1]))

        (out_dir / f"{frame['id']}.json").write_text(json.dumps({
            "frame_id": frame["id"],
            "impl": "py",
            "detector_sha256": "",   # fast_alpr resolves weights from its own cache
            "ocr_sha256": "",
            "alphabet_sha256": alphabet_sha,
            # Always false: this side cannot gate before the OCR, because there is no "before".
            "gates_applied": False,
            "readings": readings,
        }, indent=2))
        print(f"{frame['id']:<24} {len(readings)} reading(s)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
