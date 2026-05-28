import cv2
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
INPUT_DIR = "./videos"      # Folder containing the video files
OUTPUT_DIR = Path("./dataset")    # Where to save the photos

SENSITIVITY = 3000          # Min contour area (px²) to register a vehicle
COOLDOWN_SECONDS = 2.0      # Min seconds between saves for the same video
VISUALIZE = True            # Show live preview while processing

# Camera direction: which way the camera faces along the road.
#   'auto'  — detect automatically from the first few seconds of optical flow
#   'right' — camera faces right; vehicles exit at the LEFT edge of the frame
#   'left'  — camera faces left;  vehicles exit at the RIGHT edge of the frame
CAMERA_DIRECTION = 'auto'

# Pixels from the exit edge; vehicle is "cut off" if its bounding box enters this zone
EDGE_MARGIN = 80
DIRECTION_SAMPLE_FRAMES = 120   # Frames analysed for automatic direction detection
VEHICLE_GONE_SECONDS = 0.5  # Seconds with no detected motion → vehicle has left frame

BG_HISTORY = 500
BG_VAR_THRESHOLD = 50

WINDOW_NAME = 'AI Data Extractor  (Q = skip video)'
WINDOW_SIZE = (1280, 720)   # Initial window size; drag edges to resize

_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


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
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )  # type: ignore[call-overload]
            magnitude = np.hypot(flow[..., 0], flow[..., 1])
            moving = magnitude > 2.0
            if moving.any():
                x_flows.append(float(flow[..., 0][moving].mean()))
        prev_gray = gray

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)  # rewind

    if not x_flows:
        print("  No significant motion found; defaulting to 'right'.")
        return 'right'

    avg_x = float(np.mean(x_flows))
    # Negative mean x-flow  →  traffic moves leftward  →  camera faces right
    direction = 'right' if avg_x < 0 else 'left'
    arrow = '←' if direction == 'right' else '→'
    print(f"  Camera faces {direction.upper()} (traffic moves {arrow}, "
          f"avg Δx = {avg_x:+.2f} px/frame).")
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


def vehicle_is_cut_off(bbox: tuple[int, int, int, int], frame_w: int, exit_edge: str) -> bool:
    """Return True if the vehicle's bounding box overlaps the exit-edge margin."""
    x, _, w, _ = bbox
    if exit_edge == 'left':
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
    exit_edge: str,
    direction: str,
    status: str,
    timestamp: float,
) -> None:
    display = frame.copy()

    # Shade the cut-off zone in red so the operator can see it
    overlay = display.copy()
    if exit_edge == 'left':
        cv2.rectangle(overlay, (0, 0), (EDGE_MARGIN, display.shape[0]), (0, 0, 180), -1)
    else:
        cv2.rectangle(
            overlay, (frame_w - EDGE_MARGIN, 0), (frame_w, display.shape[0]), (0, 0, 180), -1
        )
    cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)

    if contour is not None:
        cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(display, (x, y), (x + w, y + h), (255, 180, 0), 2)

    label = f"Dir:{direction.upper()} | {status} | T={timestamp:.1f}s"
    cv2.putText(display, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
    cv2.imshow(WINDOW_NAME, display)


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_videos() -> None:
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    if VISUALIZE:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, *WINDOW_SIZE)

    avi_files = list(input_path.glob("*.avi"))
    print(f"Found {len(avi_files)} AVI file(s).  Starting extraction...\n")

    for video_file in avi_files:
        print(f"Processing: {video_file.name}")
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"  Could not open {video_file.name} — skipping.\n")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cooldown_frames = int(fps * COOLDOWN_SECONDS)
        vehicle_gone_frames = int(fps * VEHICLE_GONE_SECONDS)

        # --- Determine which edge vehicles exit from ---
        if CAMERA_DIRECTION == 'auto':
            direction = detect_camera_direction(cap)
        else:
            direction = CAMERA_DIRECTION
            print(f"  Camera direction (manual): {direction.upper()}")

        exit_edge = 'left' if direction == 'right' else 'right'
        print(f"  Vehicles exit at the {exit_edge.upper()} edge.  "
              f"Cut-off margin: {EDGE_MARGIN}px.\n")

        fgbg = cv2.createBackgroundSubtractorMOG2(
            history=BG_HISTORY, varThreshold=BG_VAR_THRESHOLD, detectShadows=True,
        )

        # Peak-tracking state
        best_frame: np.ndarray | None = None
        best_area: float = 0.0
        best_timestamp: float = 0.0
        cooldown_counter: int = 0
        no_vehicle_counter: int = 0
        frame_count: int = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            timestamp = frame_count / fps
            fgmask = fgbg.apply(frame)

            if cooldown_counter > 0:
                cooldown_counter -= 1
                if VISUALIZE:
                    show_frame(
                        frame, None, None, frame_w, exit_edge, direction, 'COOLDOWN', timestamp
                    )
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            result = largest_vehicle_contour(fgmask)

            if result is not None:
                contour, area, bbox = result
                no_vehicle_counter = 0

                # Only update the best frame when the vehicle is NOT cut off at the exit edge
                if not vehicle_is_cut_off(bbox, frame_w, exit_edge) and area > best_area:
                    best_frame = frame.copy()
                    best_area = area
                    best_timestamp = timestamp

                if VISUALIZE:
                    cut = vehicle_is_cut_off(bbox, frame_w, exit_edge)
                    status = f"{'CUT-OFF — skipped' if cut else 'TRACKING'} | area={int(area)}"
                    show_frame(
                        frame, contour, bbox, frame_w, exit_edge, direction, status, timestamp
                    )
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                no_vehicle_counter += 1

                # Vehicle has left the frame — save the peak (closest, unclipped) frame
                if no_vehicle_counter == vehicle_gone_frames and best_frame is not None:
                    print(f"  {best_timestamp:.1f}s — saving peak frame "
                          f"(area = {int(best_area)} px²).")
                    save_image(best_frame, video_file.stem, best_timestamp, output_path)
                    best_frame = None
                    best_area = 0.0
                    cooldown_counter = cooldown_frames

                if VISUALIZE:
                    show_frame(frame, None, None, frame_w, exit_edge, direction, 'IDLE', timestamp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        # Flush any pending best frame at end of video
        if best_frame is not None:
            print(f"  End of video — saving pending peak frame at {best_timestamp:.1f}s.")
            save_image(best_frame, video_file.stem, best_timestamp, output_path)

        cap.release()
        print(f"  Finished {video_file.name}.\n")

    if VISUALIZE:
        cv2.destroyAllWindows()

    print("All videos processed.")


def save_image(frame: np.ndarray, stem: str, timestamp: float, output_path: Path) -> None:
    file_name = f"{stem}_T{int(timestamp)}s.jpg"
    cv2.imwrite(str(output_path / file_name), frame)


if __name__ == "__main__":
    process_videos()
