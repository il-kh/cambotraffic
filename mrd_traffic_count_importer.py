import re
from pathlib import Path

import polars as pl

import helpers

helpers.load_env_local()

# --- CONFIGURATION ---
TRAFFIC_FILES_DIR = helpers.require("INPUT_DIR_TRAFFIC_FILES")
ROAD_REF_LIST = helpers.require("INPUT_DIR_ROAD_REF_LIST")
OUTPUT_DIR = helpers.require("OUTPUT_DIR_TRAFFIC_COUNTS")

vehicle_cat_tuples = [
    ("Category 1: Light & Non-Motor", "Pedestrian / Bicycle", "PEDESTRIAN"),
    ("Category 1: Light & Non-Motor", "Pedestrian / Bicycle", "BICYCLE"),
    ("Category 1: Light & Non-Motor", "Animal Cart", "HORSE"),
    ("Category 2: Two & Three Wheels", "Motorcycle", "MOTORCYCLE"),
    ("Category 2: Two & Three Wheels", "Tuk-Tuk / Remorque / Jumbo", "LITTLE_CAR"),
    ("Category 2: Two & Three Wheels", "Hand tractor / Power tiller", "TUKTUK"),
    ("Category 3: Private & Small Commercial", "Car / Taxi", "VL"),
    ("Category 3: Private & Small Commercial", "SUV / 4WD / Pick-up", "PU"),
    ("Category 3: Private & Small Commercial", "Minibus", "MINIBUS"),
    ("Category 4: Medium Commercial", "Medium Bus", "BUS"),
    ("Category 4: Medium Commercial", "Mini porter / Light truck", "LITTLE_TRUCK"),
    ("Category 4: Medium Commercial", "Medium truck (2 axles)", "TRUCK"),
    ("Category 4: Medium Commercial", "Tractor", "TRACTOR"),
    ("Category 5: Heavy & Long", "Heavy Bus", "LARGE_BUS"),
    ("Category 5: Heavy & Long", "Heavy truck (3+ axles)", "LARGE_TRUCK"),
]

vehicle_cat_df = pl.DataFrame(
    vehicle_cat_tuples, schema=["Category", "Subcategory", "ID"], orient="row"
)

# Ordered list of categories (for final column ordering)
CATEGORY_ORDER = list(dict.fromkeys(t[0] for t in vehicle_cat_tuples))

# Regex to extract rrims_id (5 or 6 digits before the first "_") from a filename
_RRIMS_RE = re.compile(r"^(\d{5,6})_")


def rrims_id_from_path(p: Path) -> str:
    m = _RRIMS_RE.match(p.name)
    return m.group(1) if m else p.stem


def row_count(p: Path) -> int:
    """Count data rows in a pipe-delimited file (total lines minus the header)."""
    return p.read_bytes().count(b"\n") - 1


# Road reference: authoritative mapping of rrims_id → road_class_code.
# Column "name" contains the rrims_id, "road_class_code" the class.
road_ref_df = pl.read_csv(ROAD_REF_LIST, ignore_errors=True).select(
    [pl.col("name").cast(pl.String).alias("rrims_id"), "road_class_code"]
)

# ---------------------------------------------------------------------------
# Step 1: Collect all FAITS.txt files
# ---------------------------------------------------------------------------
traffic_dir = TRAFFIC_FILES_DIR
traffic_files = list(traffic_dir.glob("**/*FAITS.txt"))
print(f"Found {len(traffic_files)} traffic file(s).")

# ---------------------------------------------------------------------------
# Step 2: Deduplicate — for roads counted at multiple locations keep the file
#          with the most data rows and discard the smaller one(s).
# ---------------------------------------------------------------------------
files_by_road: dict[str, list[Path]] = {}
for f in traffic_files:
    rid = rrims_id_from_path(f)
    files_by_road.setdefault(rid, []).append(f)

selected_files: list[Path] = []
for rid, files in sorted(files_by_road.items()):
    if len(files) == 1:
        selected_files.append(files[0])
    else:
        best = max(files, key=row_count)
        discarded = [f.name for f in files if f != best]
        print(f"  Road {rid}: keeping '{best.name}', discarding {discarded}")
        selected_files.append(best)

print(f"Processing {len(selected_files)} file(s) after deduplication.\n")

# ---------------------------------------------------------------------------
# Step 3: Read and concatenate all selected files, tagging each row with the
#          rrims_id derived from the filename (authoritative).
#          Cross-check against the rrims_id embedded in the SECTION column and
#          emit a warning when they disagree.
# ---------------------------------------------------------------------------
frames: list[pl.DataFrame] = []
for traffic_file in selected_files:
    df = pl.read_csv(traffic_file, separator="|")
    file_rid = rrims_id_from_path(traffic_file)

    # rrims_id found inside the SECTION column values of this file
    section_rids = (
        df["SECTION"].str.extract(r"^(\d{5,6})-").drop_nulls().unique().to_list()
    )
    mismatches = [rid for rid in section_rids if rid != file_rid]
    if mismatches:
        print(
            f"WARNING: '{traffic_file.name}' — filename rrims_id is '{file_rid}' but "
            f"SECTION column contains rrims_id(s) {mismatches}. "
            f"Using filename value '{file_rid}'."
        )

    df = df.with_columns(pl.lit(file_rid).alias("rrims_id"))
    frames.append(df)

if not frames:
    print("No traffic files found.")
    raise SystemExit(1)

traffic_df = pl.concat(frames)

# Join the authoritative road_class_code from the reference CSV.
# Rows for roads not present in the reference list are dropped with a warning.
traffic_df = traffic_df.join(road_ref_df, on="rrims_id", how="left")
unmatched = traffic_df.filter(pl.col("road_class_code").is_null())
if unmatched.height > 0:
    bad_ids = unmatched["rrims_id"].unique().sort().to_list()
    print(
        f"WARNING: {unmatched.height} row(s) belong to roads not found in the road "
        f"reference list and will be excluded from the summary.\n"
        f"  Unknown rrims_ids: {bad_ids}"
    )
    traffic_df = traffic_df.filter(pl.col("road_class_code").is_not_null())

# Map each vehicle type to its Category
traffic_df = traffic_df.join(
    vehicle_cat_df.select(["ID", "Category"]),
    left_on="TYPE VEHICULE",
    right_on="ID",
    how="left",
).with_columns(pl.col("Category").fill_null("Uncategorized"))

# ---------------------------------------------------------------------------
# Step 4: Compute average daily traffic per (rrims_id, road_class_code, Category)
#
#   a) Count vehicles per (rrims_id, road_class_code, DATE, Category)
#   b) Average those daily counts across all survey dates
# ---------------------------------------------------------------------------
daily_counts = traffic_df.group_by(
    ["rrims_id", "road_class_code", "DATE", "Category"]
).agg(pl.len().alias("count"))

avg_by_road = daily_counts.group_by(["rrims_id", "road_class_code", "Category"]).agg(
    pl.col("count").mean().alias("avg_daily")
)

# ---------------------------------------------------------------------------
# Step 5: Identify the road with the highest total average daily volume
#          within each road_class_code
# ---------------------------------------------------------------------------
road_totals = avg_by_road.group_by(["rrims_id", "road_class_code"]).agg(
    pl.col("avg_daily").sum().alias("total_avg_daily")
)

road_max_per_class = (
    road_totals.sort("total_avg_daily", descending=True)
    .unique(subset=["road_class_code"], keep="first", maintain_order=True)
    .select(
        pl.col("road_class_code"),
        pl.col("rrims_id").alias("Road w. Max Volume (RRIMS_id)"),
    )
)

# ---------------------------------------------------------------------------
# Step 6: Average across all roads in the same class → summary per
#          (road_class_code, Category), then pivot to wide format
# ---------------------------------------------------------------------------
class_summary = avg_by_road.group_by(["road_class_code", "Category"]).agg(
    pl.col("avg_daily").mean().alias("avg_daily")
)

summary = class_summary.pivot(
    values="avg_daily",
    index="road_class_code",
    on="Category",
    aggregate_function="mean",
)

# Attach the max-volume road identifier and road count per class
road_count_per_class = road_totals.group_by("road_class_code").agg(
    pl.col("rrims_id").n_unique().alias("Road Count per Class")
)
summary = summary.join(road_max_per_class, on="road_class_code", how="left").join(
    road_count_per_class, on="road_class_code", how="left"
)

# ---------------------------------------------------------------------------
# Step 7: Order columns and sort rows
# ---------------------------------------------------------------------------
fixed_cols = [
    "road_class_code",
    "Road Count per Class",
    "Road w. Max Volume (RRIMS_id)",
]
cat_cols = [c for c in CATEGORY_ORDER if c in summary.columns]
extra_cols = [c for c in summary.columns if c not in fixed_cols and c not in cat_cols]
summary = summary.select(fixed_cols + cat_cols + extra_cols).sort("road_class_code")

# Round average values to 1 decimal place
num_cols = cat_cols + extra_cols
summary = summary.with_columns([pl.col(c).round(0) for c in num_cols])

# ---------------------------------------------------------------------------
# Step 8: Print and save
# ---------------------------------------------------------------------------
print("=== Traffic Summary by Road Class (Average Daily Counts) ===")
print(summary)

output_path = Path(str(OUTPUT_DIR))
output_path.mkdir(parents=True, exist_ok=True)
out_file = output_path / "traffic_summary_by_road_class.csv"
summary.write_csv(out_file)
print(f"\nSaved to {out_file}")
