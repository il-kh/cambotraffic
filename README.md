# Cambodia Traffic Count Apps

This repository is the starting point for a collection of applications focused on traffic counting and analysis in Cambodia. The tools here are designed to help automate, visualize, and process traffic data from video and image sources, supporting research and infrastructure planning.

## Features
- Automated vehicle detection and tracking from video files
- Frame extraction and dataset creation
- Visualization of detection and tracking results

## Getting Started
1. Install `pixi` on your system (https://pixi.prefix.dev/latest/installation/)
2. Install the environment:
    ```sh
    pixi install
    ```
1. Place your video files in the `videos/` directory.
2. Run the main script:
   ```sh
   pixi run python main.py
   ```
3. Extracted frames and results will be saved in the `dataset/` directory by default.

## Requirements
- Python 3.12+ (recommended)
- OpenCV, NumPy (see `pixi.toml` for environment setup)
