import os
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from shapely.geometry import Point
import geopandas as gpd
from tqdm import tqdm
from datetime import datetime

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000
    return c * r

def compute_bbox_metrics(df, lat_col, lon_col):
    min_lat = df[lat_col].min()
    max_lat = df[lat_col].max()
    min_lon = df[lon_col].min()
    max_lon = df[lon_col].max()
    diag_length = haversine_distance(min_lat, min_lon, max_lat, max_lon)
    hor_dist = haversine_distance(min_lat, min_lon, min_lat, max_lon)
    ver_dist = haversine_distance(min_lat, min_lon, max_lat, min_lon)
    bbox_area = hor_dist * ver_dist
    return {"min_lat": min_lat, "max_lat": max_lat, "min_lon": min_lon, "max_lon": max_lon,
            "diag_length": diag_length, "hor_dist": hor_dist, "ver_dist": ver_dist, "bbox_area": bbox_area}

def run_conflation_for_threshold(df_mach9, df_client, threshold, mach9_lat_col, mach9_lon_col, client_lat_col, client_lon_col, base_output_folder):
    output_folder = os.path.join(base_output_folder, f"buffer_{threshold}")
    os.makedirs(output_folder, exist_ok=True)
    df_mach9_run = df_mach9.copy().reset_index(drop=True)
    df_client_run = df_client.copy().reset_index(drop=True)
    df_mach9_run["matched"] = False
    df_mach9_run["match_idx"] = -1
    df_mach9_run["match_distance"] = np.nan
    df_client_run["matched"] = False
    df_client_run["match_idx"] = -1
    mach9_coords = np.deg2rad(df_mach9_run[[mach9_lat_col, mach9_lon_col]].values)
    client_coords = np.deg2rad(df_client_run[[client_lat_col, client_lon_col]].values)
    tree = BallTree(client_coords, metric='haversine')
    radius_radians = threshold / 6371000.0
    for i in tqdm(range(len(df_mach9_run)), desc=f"Matching for {threshold} m"):
        if df_mach9_run.loc[i, "matched"]:
            continue
        point = mach9_coords[i].reshape(1, -1)
        candidate_indices = tree.query_radius(point, r=radius_radians, return_distance=False)[0]
        if candidate_indices.size == 0:
            continue
        best_candidate = -1
        best_distance = float("inf")
        for idx in candidate_indices:
            if not df_client_run.loc[idx, "matched"]:
                d = haversine_distance(df_mach9_run.loc[i, mach9_lat_col], df_mach9_run.loc[i, mach9_lon_col],
                                       df_client_run.loc[idx, client_lat_col], df_client_run.loc[idx, client_lon_col])
                if d < best_distance:
                    best_distance = d
                    best_candidate = idx
        if best_candidate != -1:
            df_mach9_run.at[i, "matched"] = True
            df_mach9_run.at[i, "match_idx"] = best_candidate
            df_mach9_run.at[i, "match_distance"] = best_distance
            df_client_run.at[best_candidate, "matched"] = True
            df_client_run.at[best_candidate, "match_idx"] = i
    total_mach9 = len(df_mach9_run)
    total_client = len(df_client_run)
    matched_signs = df_mach9_run["matched"].sum()
    unmatched_mach9 = total_mach9 - matched_signs
    unmatched_client = total_client - df_client_run["matched"].sum()
    perc_file1 = (matched_signs / total_mach9 * 100) if total_mach9 else 0
    perc_file2 = (matched_signs / total_client * 100) if total_client else 0
    match_distances = df_mach9_run.loc[df_mach9_run["matched"], "match_distance"].dropna()
    if len(match_distances) > 0:
        avg_distance = match_distances.mean()
        min_distance = match_distances.min()
        max_distance = match_distances.max()
    else:
        avg_distance = min_distance = max_distance = 0
    report_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bbox_mach9 = compute_bbox_metrics(df_mach9_run, mach9_lat_col, mach9_lon_col)
    bbox_client = compute_bbox_metrics(df_client_run, client_lat_col, client_lon_col)
    metrics = {
        "buffer": threshold,
        "total_mach9": total_mach9,
        "total_client": total_client,
        "matched_signs": matched_signs,
        "unmatched_mach9": unmatched_mach9,
        "unmatched_client": unmatched_client,
        "percentage_file1_matched": perc_file1,
        "percentage_file2_matched": perc_file2,
        "avg_match_distance": avg_distance,
        "min_match_distance": min_distance,
        "max_match_distance": max_distance,
        "report_datetime": report_datetime,
        "mach9_bbox_diag_length": bbox_mach9["diag_length"],
        "mach9_bbox_area": bbox_mach9["bbox_area"],
        "client_bbox_diag_length": bbox_client["diag_length"],
        "client_bbox_area": bbox_client["bbox_area"]
    }
    df_matched = df_mach9_run[df_mach9_run["matched"]].copy()
    df_matched["Source"] = "File 1"
    df_matched["File2_Lat"] = df_matched["match_idx"].apply(lambda idx: df_client_run.loc[idx, client_lat_col] if idx != -1 else None)
    df_matched["File2_Lon"] = df_matched["match_idx"].apply(lambda idx: df_client_run.loc[idx, client_lon_col] if idx != -1 else None)
    df_unmatched = pd.concat([df_mach9_run[~df_mach9_run["matched"]].assign(Source="File 1"),
                              df_client_run[~df_client_run["matched"]].assign(Source="File 2")], ignore_index=True)
    matched_csv = os.path.join(output_folder, "matched.csv")
    unmatched_csv = os.path.join(output_folder, "unmatched.csv")
    df_matched.to_csv(matched_csv, index=False)
    df_unmatched.to_csv(unmatched_csv, index=False)
    matched_shp = os.path.join(output_folder, "matched.shp")
    unmatched_shp = os.path.join(output_folder, "unmatched.shp")
    df_matched["geometry"] = df_matched.apply(lambda row: Point(row[mach9_lon_col], row[mach9_lat_col]), axis=1)
    gpd.GeoDataFrame(df_matched, geometry="geometry", crs="EPSG:4326").to_file(matched_shp, driver="ESRI Shapefile")
    df_unmatched["geometry"] = df_unmatched.apply(lambda row: Point(row["Longitude"], row["Latitude"]), axis=1)
    gpd.GeoDataFrame(df_unmatched, geometry="geometry", crs="EPSG:4326").to_file(unmatched_shp, driver="ESRI Shapefile")
    return {"metrics": metrics, "df_matched": df_matched, "df_unmatched": df_unmatched}
