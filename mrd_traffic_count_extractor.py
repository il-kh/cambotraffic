import helpers

helpers.load_env_local()

# --- CONFIGURATION ---
TRAFFIC_FILES_DIR = helpers.require("INPUT_DIR_TRAFFIC_FILES")
ROAD_REF_LIST = helpers.require("INPUT_DIR_ROAD_REF_LIST")
OUTPUT_DIR = helpers.require("OUTPUT_DIR_TRAFFIC_COUNTS")
