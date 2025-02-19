import os
import pandas as pd
import numpy as np
import math
from tqdm import tqdm  # for the progress bar
from sklearn.neighbors import BallTree
import geopandas as gpd
from shapely.geometry import Point

def match_signs_with_balltree():
    """
    Matches road signs from two different datasets using spatial proximity.

    **Functionality:**
    - Reads two CSV files containing road sign locations (latitude/longitude).
    - Uses BallTree (Haversine metric) to efficiently find closest matches.
    - Applies a distance threshold (e.g., 12 meters) to determine matching signs.
    - Outputs matched and unmatched signs as CSV and Shapefiles.

    **Input:**
    - Two CSV files containing latitude and longitude columns.
    - A predefined distance threshold in meters for matching.

    **Output:**
    - `comparison_data/matched.csv`: Signs from file 1 with closest sign from file 2.
    - `comparison_data/unmatched.csv`: Signs that didn't find a match within the threshold.
    - `comparison_data/matched.shp`: Shapefile version of matched signs.
    - `comparison_data/unmatched.shp`: Shapefile version of unmatched signs.
    - Summary statistics printed in the console.
    """

    # ---------------------------------------------------------------------
    # 1. File paths for your two datasets (update these paths as needed)
    # ---------------------------------------------------------------------
    mach9_file = r"WDOT/WDOT_Before_Sign_Face_projected_lat_lon.csv"
    client_file = r"WDOT/WDOT_Noise_Reduction_Test_Sign_Face_projected_lat_lon.csv"

    # ---------------------------------------------------------------------
    # 2. Read in CSV files (using low_memory=False to reduce dtype warnings)
    # ---------------------------------------------------------------------
    print(f"Reading file: {mach9_file} ...")
    df_mach9 = pd.read_csv(mach9_file, low_memory=False)
    print(f"Reading file: {client_file} ...")
    df_client = pd.read_csv(client_file, low_memory=False)

    # ---------------------------------------------------------------------
    # 3. Set the column names for lat/lon in each dataset
    # ---------------------------------------------------------------------
    mach9_lat_col, mach9_lon_col = "Latitude", "Longitude"
    client_lat_col, client_lon_col = "Latitude", "Longitude"

    # ---------------------------------------------------------------------
    # 4. Set a distance threshold (in meters) for matching
    # ---------------------------------------------------------------------
    DISTANCE_THRESHOLD = 12  # e.g., 12 meters

    # ---------------------------------------------------------------------
    # 5. Create helper columns to track matches in both dataframes
    # ---------------------------------------------------------------------
    df_mach9["matched"] = False
    df_mach9["match_idx"] = -1
    df_client["matched"] = False
    df_client["match_idx"] = -1

    # ---------------------------------------------------------------------
    # 6. Convert coordinates to radians (BallTree requires radians)
    # ---------------------------------------------------------------------
    mach9_coords = np.deg2rad(df_mach9[[mach9_lat_col, mach9_lon_col]].values)
    client_coords = np.deg2rad(df_client[[client_lat_col, client_lon_col]].values)

    # ---------------------------------------------------------------------
    # 7. Build a BallTree on the client data using the haversine metric
    # ---------------------------------------------------------------------
    tree = BallTree(client_coords, metric='haversine')
    radius_radians = DISTANCE_THRESHOLD / 6371000.0  # convert threshold to radians

    print("Starting matching process using BallTree...")
    # ---------------------------------------------------------------------
    # 8. For each Mach9 sign, find candidate client signs within the specified radius
    # ---------------------------------------------------------------------
    for i in tqdm(range(len(df_mach9)), desc="Matching Mach9 signs"):
        if df_mach9.loc[i, "matched"]:
            continue

        point = mach9_coords[i].reshape(1, -1)
        candidate_indices = tree.query_radius(point, r=radius_radians, return_distance=False)[0]

        if candidate_indices.size == 0:
            continue

        best_candidate = -1
        best_distance = float("inf")
        for idx in candidate_indices:
            if not df_client.loc[idx, "matched"]:
                d = haversine_distance(
                    df_mach9.loc[i, mach9_lat_col], df_mach9.loc[i, mach9_lon_col],
                    df_client.loc[idx, client_lat_col], df_client.loc[idx, client_lon_col]
                )
                if d < best_distance:
                    best_distance = d
                    best_candidate = idx

        if best_candidate != -1:
            df_mach9.at[i, "matched"] = True
            df_mach9.at[i, "match_idx"] = best_candidate
            df_client.at[best_candidate, "matched"] = True
            df_client.at[best_candidate, "match_idx"] = i

    # ---------------------------------------------------------------------
    # 9. Compute summary metrics
    # ---------------------------------------------------------------------
    total_mach9, total_client = len(df_mach9), len(df_client)
    matched_signs = df_mach9["matched"].sum()
    unmatched_mach9 = total_mach9 - matched_signs
    unmatched_client = total_client - df_client["matched"].sum()

    print("\n--- Matching Summary ---")
    print(f"Total signs in file 1: {total_mach9}")
    print(f"Total signs in file 2: {total_client}")
    print(f"Number of matched signs: {matched_signs}")
    print(f"Unmatched in file 1: {unmatched_mach9}")
    print(f"Unmatched in file 2: {unmatched_client}")

    # ---------------------------------------------------------------------
    # 10. Prepare output folder and filenames
    # ---------------------------------------------------------------------
    output_folder = os.path.join("WDOT", "comparison_data")
    os.makedirs(output_folder, exist_ok=True)

    matched_csv = os.path.join(output_folder, "matched.csv")
    unmatched_csv = os.path.join(output_folder, "unmatched.csv")
    matched_shp = os.path.join(output_folder, "matched.shp")
    unmatched_shp = os.path.join(output_folder, "unmatched.shp")

    # ---------------------------------------------------------------------
    # 11. Create output DataFrames
    # ---------------------------------------------------------------------
    df_matched = df_mach9[df_mach9["matched"]].copy()
    df_matched["File2_Lat"] = df_matched["match_idx"].apply(
        lambda idx: df_client.loc[idx, client_lat_col] if idx != -1 else None
    )
    df_matched["File2_Lon"] = df_matched["match_idx"].apply(
        lambda idx: df_client.loc[idx, client_lon_col] if idx != -1 else None
    )

    df_unmatched = pd.concat([
        df_mach9[~df_mach9["matched"]].assign(Source="File 1"),
        df_client[~df_client["matched"]].assign(Source="File 2")
    ], ignore_index=True)

    # ---------------------------------------------------------------------
    # 12. Write CSV files
    # ---------------------------------------------------------------------
    df_matched.to_csv(matched_csv, index=False)
    df_unmatched.to_csv(unmatched_csv, index=False)

    print("\nCSV Output files written:")
    print(f"  Matched:   {matched_csv}")
    print(f"  Unmatched: {unmatched_csv}")

    # ---------------------------------------------------------------------
    # 13. Write to Shapefiles
    # ---------------------------------------------------------------------
    df_matched["geometry"] = df_matched.apply(lambda row: Point(row[mach9_lon_col], row[mach9_lat_col]), axis=1)
    gpd.GeoDataFrame(df_matched, geometry="geometry", crs="EPSG:4326").to_file(matched_shp, driver="ESRI Shapefile")

    df_unmatched["geometry"] = df_unmatched.apply(lambda row: Point(row["Longitude"], row["Latitude"]), axis=1)
    gpd.GeoDataFrame(df_unmatched, geometry="geometry", crs="EPSG:4326").to_file(unmatched_shp, driver="ESRI Shapefile")

    print("\nShapefile Output files written:")
    print(f"  Matched Shapefile:   {matched_shp}")
    print(f"  Unmatched Shapefile: {unmatched_shp}")

if __name__ == "__main__":
    match_signs_with_balltree()
