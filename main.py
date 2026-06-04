"""
main.py — entry point for the traffic video keyframe extractor.

Usage:
    python main.py                  # vehicle tracker (default)
    python main.py --mode vehicle   # same
    python main.py --mode interval  # simple interval-based extractor    python main.py --lighting day_sun
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Traffic video keyframe extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "modes:\n"
            "  vehicle   Track individual vehicles; save 5 semantically-labelled\n"
            "            keyframes per passage (entry/best/center/farther/furthest).\n"
            "            Metadata: filename, source_video, slot, travel_dir, lighting, …\n\n"
            "  interval  Save one frame every INTERVAL_SECONDS whenever foreground\n"
            "            activity exceeds MIN_ACTIVITY.  No tracking, higher volume.\n"
            "            Metadata: filename, source_video, frame_idx, timestamp_s, lighting\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["vehicle", "interval"],
        default="vehicle",
        help="Extraction strategy (default: vehicle)",
    )
    parser.add_argument(
        "--lighting",
        choices=[
            "day_sun",
            "day_overcast",
            "day_rain",
            "dusk_dawn",
            "night_dry",
            "night_rain",
        ],
        default=None,
        help="Lighting condition written into every metadata row (default: unknown)",
    )
    args = parser.parse_args()

    if args.mode == "interval":
        import simple_extractor

        simple_extractor.process_videos()
    else:
        import vehicle_extractor

        vehicle_extractor.process_videos(lighting=args.lighting or "unknown")


if __name__ == "__main__":
    main()
