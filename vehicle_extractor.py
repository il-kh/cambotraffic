"""
vehicle_extractor.py — vehicle-tracking keyframe extractor.

Detects individual vehicle passages via background subtraction, finds the
peak (largest bounding-box area) frame per vehicle, then seeks back and saves
five semantically-labelled JPEG keyframes (entry / best / center / farther /
furthest for receding; furthest / farther / center / best / exit for approaching).

Run directly:  python vehicle_extractor.py
Or via main:   python main.py --mode vehicle  (default)
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

SENSITIVITY = 3000  # Min contour area (px²) to register a vehicle
COOLDOWN_SECONDS = 2.0  # Min seconds between keyframe batches
VISUALIZE = True  # Show live preview while processing

# Camera direction: which way the camera faces along the road.
#   'auto'  — detect from optical flow of the first DIRECTION_SAMPLE_FRAMES frames
#   'right' — camera faces right  (close-lane L→R enters LEFT;  exit edge = LEFT)
#   'left'  — camera faces left   (close-lane R→L enters RIGHT; exit edge = RIGHT)
CAMERA_DIRECTION = "auto"

EDGE_MARGIN = 80  # Pixels from the exit edge considered "hard cut-off"
DIRECTION_SAMPLE_FRAMES = 120
VEHICLE_GONE_SECONDS = 0.5  # Seconds of no detection → vehicle has left

# Per-slot step sizes (frames between consecutive slots).
# Use smaller gaps near the camera (fast apparent motion) and larger gaps far away.
# Four gaps for five slots — tune for the road geometry and typical vehicle speed.
#
# Receding sequence:   entry → best → center → farther → furthest
# Approaching sequence: furthest → farther → center → best → exit
SLOT_STEPS_RECEDING = (
    5,
    10,
    25,
    40,
)  # entry→best, best→center, center→farther, farther→furthest
SLOT_STEPS_APPROACHING = (
    40,
    25,
    10,
    5,
)  # furthest→farther, farther→center, center→best, best→exit

BG_HISTORY = 500
BG_VAR_THRESHOLD = 50

# Supported video file extensions (both cases for case-sensitive filesystems)
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

WINDOW_NAME = "AI Data Extractor — Vehicle Mode  (Q = skip video)"
WINDOW_SIZE = (1280, 720)  # Initial window size; drag edges to resize

_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# Slot labels (chronological order within each sequence)
SLOT_NAMES_RECEDING = ("entry", "best", "center", "farther", "furthest")
SLOT_NAMES_APPROACHING = ("furthest", "farther", "center", "best", "exit")

# Frame offsets relative to peak ("best" = 0), derived from SLOT_STEPS_*.
# Edit SLOT_STEPS_* above; these recompute automatically.
_sr = SLOT_STEPS_RECEDING
_sa = SLOT_STEPS_APPROACHING
SLOT_OFFSETS_RECEDING = (-_sr[0], 0, _sr[1], _sr[1] + _sr[2], _sr[1] + _sr[2] + _sr[3])
SLOT_OFFSETS_APPROACHING = (
    -(_sa[0] + _sa[1] + _sa[2]),
    -(_sa[1] + _sa[2]),
    -_sa[2],
    0,
    _sa[3],
)
del _sr, _sa

# Valid lighting labels for the metadata CSV (edit 'unknown' → one of these before training)
LIGHTING_CHOICES = (
    "day_sun",
    "day_overcast",
    "day_rain",
    "dusk_dawn",
    "night_dry",
    "night_rain",
)


# ---------------------------------------------------------------------------
# Direction detection
# ---------------------------------------------------------------------------


def detect_camera_direction(cap: cv2.VideoCapture) -> str:
    """
    Analyse the first DIRECTION_SAMPLE_FRAMES frames with dense optical flow to
    determine the dominant horizontal direction of traffic movement.

    Returns 'right' if the camera faces right (vehicles travel leftward in the
    frame and exit at the left edge), or 'left' if it faces left (vehicles travel
    rightward and exit at the right edge).
    """
    print("  Analysing optical flow to detect camera direction...")
    x_flows: list[float] = []
    prev_gray = None
    start_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)

    for _ in range(DIRECTION_SAMPLE_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )  # type: ignore[call-overload]
            magnitude = np.hypot(flow[..., 0], flow[..., 1])
            moving = magnitude > 2.0
            if moving.any():
                x_flows.append(float(flow[..., 0][moving].mean()))
        prev_gray = gray

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)  # rewind

    if not x_flows:
        print("  No significant motion found; defaulting to 'right'.")
        return "right"

    avg_x = float(np.mean(x_flows))
    # Negative mean x-flow  →  traffic moves leftward  →  camera faces right
    direction = "right" if avg_x < 0 else "left"
    arrow = "←" if direction == "right" else "→"
    print(
        f"  Camera faces {direction.upper()} (traffic moves {arrow}, "
        f"avg Δx = {avg_x:+.2f} px/frame)."
    )
    return direction


# ---------------------------------------------------------------------------
# Vehicle detection helpers
# ---------------------------------------------------------------------------


def largest_vehicle_contour(fgmask: np.ndarray):
    """
    Clean the foreground mask and return the largest contour above SENSITIVITY
    as (contour, area, (x, y, w, h)), or None if no vehicle is detected.
    """
    clean = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, _MORPH_KERNEL)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < SENSITIVITY:
        return None
    return best, area, tuple(cv2.boundingRect(best))


def vehicle_is_cut_off(
    bbox: tuple[int, int, int, int], frame_w: int, exit_edge: str
) -> bool:
    """Return True if the vehicle's bounding box overlaps the exit-edge margin."""
    x, _, w, _ = bbox
    if exit_edge == "left":
        return x < EDGE_MARGIN
    else:
        return (x + w) > (frame_w - EDGE_MARGIN)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def show_frame(
    frame: np.ndarray,
    contour: np.ndarray | None,
    bbox: tuple[int, int, int, int] | None,
    frame_w: int,
    frame_h: int,
    exit_edge: str,
    camera_dir: str,
    status: str,
    timestamp: float,
) -> None:
    display = frame.copy()
    overlay = display.copy()

    # Shade the exit/cut-off zone in red
    if exit_edge == "left":
        cv2.rectangle(overlay, (0, 0), (EDGE_MARGIN, frame_h), (0, 0, 180), -1)
    else:
        cv2.rectangle(
            overlay, (frame_w - EDGE_MARGIN, 0), (frame_w, frame_h), (0, 0, 180), -1
        )
    cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)

    if contour is not None:
        cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(display, (x, y), (x + w, y + h), (255, 180, 0), 2)

    label = f"Cam:{camera_dir.upper()} | {status} | T={timestamp:.1f}s"
    cv2.putText(
        display, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2
    )
    cv2.imshow(WINDOW_NAME, display)


# ---------------------------------------------------------------------------
# Keyframe saving helpers
# ---------------------------------------------------------------------------


def _travel_direction(first_x: float | None, last_x: float | None) -> str:
    """Infer travel direction from first and last tracked x-centre of the bounding box."""
    if first_x is None or last_x is None:
        return "unknown"
    return "ltr" if last_x > first_x else "rtl"


def _is_approaching(travel_dir: str, exit_edge: str) -> bool:
    """
    Return True if the vehicle is heading *toward* the exit edge (far lane).
    For these vehicles the peak (largest bbox) occurs near the end of their traversal,
    so the useful diversity frames are before the peak, not after.
    """
    return (exit_edge == "left" and travel_dir == "rtl") or (
        exit_edge == "right" and travel_dir == "ltr"
    )


_METADATA_FIELDS = (
    "filename",
    "source_video",
    "video_mtime",
    "slot",
    "slot_offset",
    "frame_idx",
    "timestamp_s",
    "camera_dir",
    "travel_dir",
    "lighting",
)


def _append_metadata_csv(output_path: Path, rows: list[dict]) -> None:
    """Append image metadata rows to metadata.csv, creating the file with header if needed."""
    if not rows:
        return
    csv_path = output_path / "metadata.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_METADATA_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def save_vehicle_keyframes(
    cap: cv2.VideoCapture,
    peak_frame_idx: int,
    fps: float,
    camera_dir: str,
    travel_dir: str,  # 'ltr' | 'rtl' | 'unknown'
    video_path: Path,
    output_path: Path,
    total_frames: int,
    approaching: bool = False,
) -> None:
    """
    Seek to each of the five keyframe positions, save as JPEG, and append a row
    to metadata.csv in the output directory.

    Approaching vehicles use SLOT_OFFSETS_APPROACHING (frames before the peak);
    receding vehicles use SLOT_OFFSETS_RECEDING (frames after the peak).
    The video position is restored after all seeks.

    Edit the 'lighting' column in metadata.csv before YOLO/HAILO training.
    Valid values: day_sun | day_overcast | day_rain | dusk_dawn | night_dry | night_rain
    """
    slot_names = SLOT_NAMES_APPROACHING if approaching else SLOT_NAMES_RECEDING
    slot_offsets = SLOT_OFFSETS_APPROACHING if approaching else SLOT_OFFSETS_RECEDING

    saved_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    mtime_iso = datetime.fromtimestamp(video_path.stat().st_mtime).isoformat(
        timespec="seconds"
    )
    metadata_rows: list[dict] = []
    saved_indices: set[int] = set()

    for slot, offset in zip(slot_names, slot_offsets):
        raw_idx = peak_frame_idx + offset
        clamped_idx = max(0, min(raw_idx, total_frames - 1))
        if clamped_idx in saved_indices:
            print(
                f"    [{slot}] Frame {clamped_idx} already saved — skipped (clamped duplicate)."
            )
            continue
        saved_indices.add(clamped_idx)

        cap.set(cv2.CAP_PROP_POS_FRAMES, clamped_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"    [{slot}] Cannot read frame {clamped_idx} — skipped.")
            continue

        ts = clamped_idx / fps
        file_name = f"{video_path.stem}_T{int(ts * 1000):08d}ms.jpg"

        cv2.imwrite(str(output_path / file_name), frame)
        metadata_rows.append(
            {
                "filename": file_name,
                "source_video": video_path.name,
                "video_mtime": mtime_iso,
                "slot": slot,
                "slot_offset": offset,
                "frame_idx": clamped_idx,
                "timestamp_s": round(ts, 3),
                "camera_dir": camera_dir,
                "travel_dir": travel_dir,
                "lighting": "unknown",
            }
        )
        print(f"    [{slot:8s}] T={ts:.1f}s  →  {file_name}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, saved_pos)
    _append_metadata_csv(output_path, metadata_rows)


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------


def process_videos() -> None:
    input_path = Path(INPUT_DIR)
    input_path.mkdir(exist_ok=True)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    if VISUALIZE:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, *WINDOW_SIZE)

    # Collect video files across all supported extensions (deduplicated, sorted)
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
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = int(1e9)  # some containers don't report a valid total

        cooldown_frames = int(fps * COOLDOWN_SECONDS)
        vehicle_gone_frames = int(fps * VEHICLE_GONE_SECONDS)

        if CAMERA_DIRECTION == "auto":
            camera_dir = detect_camera_direction(cap)
        else:
            camera_dir = CAMERA_DIRECTION
            print(f"  Camera direction (manual): {camera_dir.upper()}")

        exit_edge = "left" if camera_dir == "right" else "right"
        print(f"  Exit edge: {exit_edge.upper()}.  Cut-off margin: {EDGE_MARGIN}px.\n")

        fgbg = cv2.createBackgroundSubtractorMOG2(
            history=BG_HISTORY,
            varThreshold=BG_VAR_THRESHOLD,
            detectShadows=True,
        )

        # --- Per-vehicle tracking state ---
        best_frame_idx: int = 0
        best_area: float = 0.0
        tracking: bool = False
        first_x: float | None = None  # x-centre at first detection of current vehicle
        last_x: float | None = None  # x-centre at most recent detection
        cooldown_counter: int = 0
        no_vehicle_counter: int = 0
        frame_count: int = 0
        skip_video: bool = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_idx = frame_count - 1  # 0-based index for cap.set / seeking
            timestamp = frame_idx / fps
            fgmask = fgbg.apply(frame)

            if cooldown_counter > 0:
                cooldown_counter -= 1
                if VISUALIZE:
                    show_frame(
                        frame,
                        None,
                        None,
                        frame_w,
                        frame_h,
                        exit_edge,
                        camera_dir,
                        "COOLDOWN",
                        timestamp,
                    )
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        skip_video = True
                        break
                continue

            result = largest_vehicle_contour(fgmask)

            if result is not None:
                contour, area, bbox = result
                no_vehicle_counter = 0
                x, y, w, h = bbox
                x_center = x + w / 2.0

                # Record entry x-position; always update latest x-position
                if first_x is None:
                    first_x = x_center
                last_x = x_center

                # Update peak only when vehicle is not yet cut off at the exit edge
                if (
                    not vehicle_is_cut_off(bbox, frame_w, exit_edge)
                    and area > best_area
                ):
                    best_frame_idx = frame_idx
                    best_area = area
                    tracking = True

                if VISUALIZE:
                    cut = vehicle_is_cut_off(bbox, frame_w, exit_edge)
                    status = f"{'CUT-OFF' if cut else 'TRACKING'} | area={int(area)}"
                    show_frame(
                        frame,
                        contour,
                        bbox,
                        frame_w,
                        frame_h,
                        exit_edge,
                        camera_dir,
                        status,
                        timestamp,
                    )
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        skip_video = True
                        break
            else:
                no_vehicle_counter += 1

                if no_vehicle_counter == vehicle_gone_frames and tracking:
                    travel_dir = _travel_direction(first_x, last_x)
                    approaching = _is_approaching(travel_dir, exit_edge)
                    slot_count = len(
                        SLOT_NAMES_APPROACHING if approaching else SLOT_NAMES_RECEDING
                    )
                    lane_label = (
                        "far-lane (approaching)"
                        if approaching
                        else "close-lane (receding)"
                    )
                    print(
                        f"  Vehicle gone — peak T={best_frame_idx / fps:.1f}s "
                        f"(area={int(best_area)}, travel={travel_dir}, {lane_label}).  "
                        f"Saving {slot_count} keyframes..."
                    )
                    save_vehicle_keyframes(
                        cap,
                        best_frame_idx,
                        fps,
                        camera_dir,
                        travel_dir,
                        video_file,
                        output_path,
                        total_frames,
                        approaching=approaching,
                    )
                    best_frame_idx = 0
                    best_area = 0.0
                    tracking = False
                    first_x = None
                    last_x = None
                    cooldown_counter = cooldown_frames

                if VISUALIZE:
                    show_frame(
                        frame,
                        None,
                        None,
                        frame_w,
                        frame_h,
                        exit_edge,
                        camera_dir,
                        "IDLE",
                        timestamp,
                    )
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        skip_video = True
                        break

        # Flush any pending vehicle at end of video
        if not skip_video and tracking and best_area > 0:
            travel_dir = _travel_direction(first_x, last_x)
            approaching = _is_approaching(travel_dir, exit_edge)
            print(
                f"  End of video — flushing keyframes "
                f"(peak T={best_frame_idx / fps:.1f}s, travel={travel_dir})..."
            )
            save_vehicle_keyframes(
                cap,
                best_frame_idx,
                fps,
                camera_dir,
                travel_dir,
                video_file,
                output_path,
                total_frames,
                approaching=approaching,
            )

        cap.release()
        print(f"  Finished {video_file.name}.\n")

    if VISUALIZE:
        cv2.destroyAllWindows()

    print("All videos processed.")


if __name__ == "__main__":
    process_videos()
