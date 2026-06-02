"""
simple_extractor.py — interval-based frame extraction.

Saves one frame every INTERVAL_SECONDS whenever the foreground activity
exceeds MIN_ACTIVITY pixels².  No vehicle tracking, no direction detection.
Good for generating high-volume, scene-realistic training data.

Run directly:  python simple_extractor.py
Or via main:   python main.py --mode interval
"""

import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import helpers

helpers.load_env_local()
# --- CONFIGURATION ---
INPUT_DIR = helpers.require("INPUT_DIR_VIDEOS")
OUTPUT_DIR = helpers.require("OUTPUT_DIR_DATASET")

INTERVAL_SECONDS = 2.0  # Minimum gap between saved frames
MIN_ACTIVITY = 3000  # Min foreground area (px²) to consider a frame "active"
VISUALIZE = True  # Show live preview while processing

BG_HISTORY = 500
BG_VAR_THRESHOLD = 50

VIDEO_EXTENSIONS = (
    "*.mp4",
    "*.MP4",
    "*.avi",
    "*.AVI",
    "*.mov",
    "*.MOV",
    "*.mkv",
    "*.MKV",
    "*.m4v",
    "*.M4V",
)

WINDOW_NAME = "AI Data Extractor — Interval Mode  (Q = skip video)"
WINDOW_SIZE = (1280, 720)

_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

_METADATA_FIELDS = (
    "filename",
    "source_video",
    "video_mtime",
    "frame_idx",
    "timestamp_s",
    "lighting",
)


def _append_metadata_csv(output_path: Path, rows: list[dict]) -> None:
    """Append rows to metadata.csv, creating with header if needed."""
    if not rows:
        return
    csv_path = output_path / "metadata.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_METADATA_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _foreground_area(fgmask: np.ndarray) -> float:
    """Return the number of foreground pixels after morphological cleanup."""
    clean = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, _MORPH_KERNEL)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    return float(np.count_nonzero(clean))


def process_videos() -> None:
    input_path = Path(INPUT_DIR)
    input_path.mkdir(exist_ok=True)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    if VISUALIZE:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, *WINDOW_SIZE)

    video_files: list[Path] = []
    for pattern in VIDEO_EXTENSIONS:
        video_files.extend(input_path.glob(pattern))
    video_files = sorted(set(video_files))
    print(f"Found {len(video_files)} video file(s).  Starting extraction...\n")

    for video_file in video_files:
        print(f"Processing: {video_file.name}")
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"  Could not open {video_file.name} — skipping.\n")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        interval_frames = max(1, int(fps * INTERVAL_SECONDS))
        mtime_iso = datetime.fromtimestamp(video_file.stat().st_mtime).isoformat(
            timespec="seconds"
        )

        fgbg = cv2.createBackgroundSubtractorMOG2(
            history=BG_HISTORY,
            varThreshold=BG_VAR_THRESHOLD,
            detectShadows=True,
        )

        frame_count = 0
        frames_since_save = (
            interval_frames  # allow saving on the very first active frame
        )
        skip_video = False
        saved = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx = frame_count
            frame_count += 1
            timestamp = frame_idx / fps
            fgmask = fgbg.apply(frame)
            activity = _foreground_area(fgmask)
            frames_since_save += 1

            active = activity >= MIN_ACTIVITY
            due = frames_since_save >= interval_frames

            if active and due:
                file_name = f"{video_file.stem}_T{int(timestamp * 1000):08d}ms.jpg"
                cv2.imwrite(str(output_path / file_name), frame)
                _append_metadata_csv(
                    output_path,
                    [
                        {
                            "filename": file_name,
                            "source_video": video_file.name,
                            "video_mtime": mtime_iso,
                            "frame_idx": frame_idx,
                            "timestamp_s": round(timestamp, 3),
                            "lighting": "unknown",
                        }
                    ],
                )
                frames_since_save = 0
                saved += 1
                print(
                    f"  Saved T={timestamp:.1f}s  →  {file_name}"
                    f"  (activity={int(activity)})"
                )

            if VISUALIZE:
                display = frame.copy()
                bar = f"{'ACTIVE' if active else 'idle':6s} | activity={int(activity):6d} | saved={saved}"
                cv2.putText(
                    display,
                    bar,
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 220, 255),
                    2,
                )
                cv2.imshow(WINDOW_NAME, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    skip_video = True
                    break

        cap.release()
        print(f"  Finished {video_file.name} — {saved} frame(s) saved.\n")

    if VISUALIZE:
        cv2.destroyAllWindows()

    print("All videos processed.")


if __name__ == "__main__":
    process_videos()
