import os
import pandas as pd
import numpy as np
from tqdm import tqdm  # for the progress bar
from sklearn.neighbors import BallTree
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the Earth specified in decimal degrees.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Radius of Earth in meters
    return c * r


def compute_bbox_metrics(df, lat_col, lon_col):
    """
    Compute bounding box metrics for a set of points.
    Returns min/max lat/lon, the diagonal (estimated corridor length),
    horizontal/vertical extents, and approximate bounding box area.
    """
    min_lat = df[lat_col].min()
    max_lat = df[lat_col].max()
    min_lon = df[lon_col].min()
    max_lon = df[lon_col].max()

    diag_length = haversine_distance(min_lat, min_lon, max_lat, max_lon)
    hor_dist = haversine_distance(min_lat, min_lon, min_lat, max_lon)
    ver_dist = haversine_distance(min_lat, min_lon, max_lat, min_lon)

    bbox_area = hor_dist * ver_dist
    return {
        "min_lat": min_lat,
        "max_lat": max_lat,
        "min_lon": min_lon,
        "max_lon": max_lon,
        "diag_length": diag_length,
        "hor_dist": hor_dist,
        "ver_dist": ver_dist,
        "bbox_area": bbox_area
    }


def run_conflation_for_threshold(df_mach9, df_client, threshold, mach9_lat_col, mach9_lon_col, client_lat_col,
                                 client_lon_col, base_output_folder):
    """
    Runs the conflation process for a given distance threshold.
    Saves CSV and Shapefile outputs into a subfolder and returns a dictionary with results.
    """
    # Create a subfolder for this buffer (e.g., "buffer_5")
    output_folder = os.path.join(base_output_folder, f"buffer_{threshold}")
    os.makedirs(output_folder, exist_ok=True)

    # Fresh copies for this run
    df_mach9_run = df_mach9.copy().reset_index(drop=True)
    df_client_run = df_client.copy().reset_index(drop=True)

    # Initialize helper columns
    df_mach9_run["matched"] = False
    df_mach9_run["match_idx"] = -1
    df_mach9_run["match_distance"] = np.nan
    df_client_run["matched"] = False
    df_client_run["match_idx"] = -1

    # Convert coordinates to radians
    mach9_coords = np.deg2rad(df_mach9_run[[mach9_lat_col, mach9_lon_col]].values)
    client_coords = np.deg2rad(df_client_run[[client_lat_col, client_lon_col]].values)

    # Build BallTree using client coordinates
    tree = BallTree(client_coords, metric='haversine')
    radius_radians = threshold / 6371000.0  # convert meters to radians

    print(f"Starting conflation for buffer: {threshold} meters ...")
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
                d = haversine_distance(
                    df_mach9_run.loc[i, mach9_lat_col], df_mach9_run.loc[i, mach9_lon_col],
                    df_client_run.loc[idx, client_lat_col], df_client_run.loc[idx, client_lon_col]
                )
                if d < best_distance:
                    best_distance = d
                    best_candidate = idx
        if best_candidate != -1:
            df_mach9_run.at[i, "matched"] = True
            df_mach9_run.at[i, "match_idx"] = best_candidate
            df_mach9_run.at[i, "match_distance"] = best_distance
            df_client_run.at[best_candidate, "matched"] = True
            df_client_run.at[best_candidate, "match_idx"] = i

    # Compute matching metrics
    total_mach9 = len(df_mach9_run)
    total_client = len(df_client_run)
    matched_signs = df_mach9_run["matched"].sum()
    unmatched_mach9 = total_mach9 - matched_signs
    unmatched_client = total_client - df_client_run["matched"].sum()
    percentage_file1_matched = (matched_signs / total_mach9 * 100) if total_mach9 > 0 else 0
    percentage_file2_matched = (matched_signs / total_client * 100) if total_client > 0 else 0

    match_distances = df_mach9_run.loc[df_mach9_run["matched"], "match_distance"].dropna()
    if len(match_distances) > 0:
        avg_distance = match_distances.mean()
        min_distance = match_distances.min()
        max_distance = match_distances.max()
    else:
        avg_distance = min_distance = max_distance = 0

    report_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Compute bounding box metrics for each dataset
    bbox_metrics_mach9 = compute_bbox_metrics(df_mach9_run, mach9_lat_col, mach9_lon_col)
    bbox_metrics_client = compute_bbox_metrics(df_client_run, client_lat_col, client_lon_col)

    metrics = {
        "buffer": threshold,
        "total_mach9": total_mach9,
        "total_client": total_client,
        "matched_signs": matched_signs,
        "unmatched_mach9": unmatched_mach9,
        "unmatched_client": unmatched_client,
        "percentage_file1_matched": percentage_file1_matched,
        "percentage_file2_matched": percentage_file2_matched,
        "avg_match_distance": avg_distance,
        "min_match_distance": min_distance,
        "max_match_distance": max_distance,
        "report_datetime": report_datetime,
        "mach9_bbox_diag_length": bbox_metrics_mach9["diag_length"],
        "mach9_bbox_area": bbox_metrics_mach9["bbox_area"],
        "client_bbox_diag_length": bbox_metrics_client["diag_length"],
        "client_bbox_area": bbox_metrics_client["bbox_area"]
    }

    # Prepare output DataFrames for saving
    df_matched = df_mach9_run[df_mach9_run["matched"]].copy()
    # Mark these as coming from File 1
    df_matched["Source"] = "File 1"
    df_matched["File2_Lat"] = df_matched["match_idx"].apply(
        lambda idx: df_client_run.loc[idx, client_lat_col] if idx != -1 else None
    )
    df_matched["File2_Lon"] = df_matched["match_idx"].apply(
        lambda idx: df_client_run.loc[idx, client_lon_col] if idx != -1 else None
    )
    df_unmatched = pd.concat([
        df_mach9_run[~df_mach9_run["matched"]].assign(Source="File 1"),
        df_client_run[~df_client_run["matched"]].assign(Source="File 2")
    ], ignore_index=True)

    # Save CSV outputs
    matched_csv = os.path.join(output_folder, "matched.csv")
    unmatched_csv = os.path.join(output_folder, "unmatched.csv")
    df_matched.to_csv(matched_csv, index=False)
    df_unmatched.to_csv(unmatched_csv, index=False)
    print(f"CSV files saved in {output_folder}")

    # Save Shapefile outputs
    matched_shp = os.path.join(output_folder, "matched.shp")
    unmatched_shp = os.path.join(output_folder, "unmatched.shp")
    df_matched["geometry"] = df_matched.apply(lambda row: Point(row[mach9_lon_col], row[mach9_lat_col]), axis=1)
    gpd.GeoDataFrame(df_matched, geometry="geometry", crs="EPSG:4326").to_file(matched_shp, driver="ESRI Shapefile")
    df_unmatched["geometry"] = df_unmatched.apply(lambda row: Point(row["Longitude"], row["Latitude"]), axis=1)
    gpd.GeoDataFrame(df_unmatched, geometry="geometry", crs="EPSG:4326").to_file(unmatched_shp, driver="ESRI Shapefile")
    print(f"Shapefiles saved in {output_folder}")

    return {
        "metrics": metrics,
        "df_matched": df_matched,
        "df_unmatched": df_unmatched
    }


def generate_conflation_report_all(results, logo_path=None, output_pdf="conflation_report.pdf"):
    """
    Generates a multipage PDF report that includes:
      - A cover page.
      - For each conflation run (each buffer distance):
          • One page with key metrics and a map.
          • One page with two full-width bar charts (with side margins) for counts:
                - Top 10 Matched MUTCD counts and Top 10 Unmatched MUTCD counts.
          • One page with two full-width bar charts for accuracy percentages arranged in two columns.
                Below each chart a table is printed showing the MUTCD code, its accuracy (%),
                and the count breakdown from File 1 vs File 2.
    """
    with PdfPages(output_pdf) as pdf:
        # --- Cover Page ---
        fig_cover = plt.figure(figsize=(8.27, 11.69))
        ax_cover = fig_cover.add_axes([0, 0, 1, 1])
        ax_cover.axis('off')
        title_text = "Conflation Report"
        subtitle_text = "Buffer Comparisons: 5m, 10m, and 12m"
        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cover_text = f"{title_text}\n\n{subtitle_text}\n\nReport Generated: {gen_time}"
        ax_cover.text(0.5, 0.5, cover_text, fontsize=16, ha='center', va='center')
        if logo_path is not None and os.path.exists(logo_path):
            ax_logo = fig_cover.add_axes([0.35, 0.7, 0.3, 0.2])
            logo = plt.imread(logo_path)
            ax_logo.imshow(logo)
            ax_logo.axis("off")
        pdf.savefig(fig_cover)
        plt.close()

        # --- Pages per Buffer Run ---
        for result in results:
            metrics = result["metrics"]
            df_matched = result["df_matched"]
            df_unmatched = result["df_unmatched"]

            # Page 1: Metrics & Map
            fig1 = plt.figure(figsize=(8.27, 11.69))
            ax_text = fig1.add_axes([0.1, 0.55, 0.8, 0.35])
            ax_text.axis('off')
            metrics_text = f"""
Buffer Distance: {metrics['buffer']} meters
Report Generated: {metrics['report_datetime']}

-- Matching Metrics --
Total signs in File 1: {metrics['total_mach9']}
Total signs in File 2: {metrics['total_client']}
Matched signs: {metrics['matched_signs']}
Unmatched in File 1: {metrics['unmatched_mach9']}
Unmatched in File 2: {metrics['unmatched_client']}

Percentage matched in File 1: {metrics['percentage_file1_matched']:.2f}%
Percentage matched in File 2: {metrics['percentage_file2_matched']:.2f}%

Average match distance: {metrics['avg_match_distance']:.2f} m
Minimum match distance: {metrics['min_match_distance']:.2f} m
Maximum match distance: {metrics['max_match_distance']:.2f} m

-- Bounding Box Metrics --
File 1 (Mach9): Diagonal ≈ {metrics['mach9_bbox_diag_length']:.2f} m, Area ≈ {metrics['mach9_bbox_area']:.2f} m²
File 2 (Client): Diagonal ≈ {metrics['client_bbox_diag_length']:.2f} m, Area ≈ {metrics['client_bbox_area']:.2f} m²
            """
            ax_text.text(0.5, 0.5, metrics_text, transform=ax_text.transAxes,
                         fontsize=10, ha='center', va='center')
            ax_map = fig1.add_axes([0.1, 0.1, 0.8, 0.35])
            ax_map.scatter(df_matched["Longitude"], df_matched["Latitude"],
                           color='green', label='Matched Signs', alpha=0.7, edgecolors='black', s=50)
            ax_map.scatter(df_unmatched["Longitude"], df_unmatched["Latitude"],
                           color='red', label='Unmatched Signs', alpha=0.7, edgecolors='black', s=50)
            ax_map.set_title(f"Map of Road Signs (Buffer: {metrics['buffer']} m)")
            ax_map.set_xlabel("Longitude")
            ax_map.set_ylabel("Latitude")
            ax_map.legend()
            ax_map.grid(True)
            pdf.savefig(fig1)
            plt.close()

            # Page 2: Bar Charts for Counts (with side margins)
            if "MUTCD" in df_matched.columns and "MUTCD" in df_unmatched.columns:
                top_matched_counts = df_matched["MUTCD"].value_counts().head(10)
                top_unmatched_counts = df_unmatched["MUTCD"].value_counts().head(10)

                fig_counts = plt.figure(figsize=(8.27, 11.69))
                ax_counts_top = fig_counts.add_axes([0.15, 0.55, 0.7, 0.35])
                ax_counts_bottom = fig_counts.add_axes([0.15, 0.1, 0.7, 0.35])
                # Using more muted colors:
                ax_counts_top.bar(top_matched_counts.index.astype(str), top_matched_counts.values, color='#8da0cb')
                ax_counts_top.set_title("Top 10 Matched MUTCD Counts")
                ax_counts_top.set_xlabel("MUTCD Code")
                ax_counts_top.set_ylabel("Count")
                ax_counts_bottom.bar(top_unmatched_counts.index.astype(str), top_unmatched_counts.values,
                                     color='#fc8d62')
                ax_counts_bottom.set_title("Top 10 Unmatched MUTCD Counts")
                ax_counts_bottom.set_xlabel("MUTCD Code")
                ax_counts_bottom.set_ylabel("Count")
                fig_counts.tight_layout()
                pdf.savefig(fig_counts)
                plt.close()

                # Page 3: Bar Charts for Percentages with Tables
                df_all = pd.concat([df_matched, df_unmatched])
                accuracy_series = df_all.groupby("MUTCD")["matched"].mean() * 100
                top10_accuracy = accuracy_series.sort_values(ascending=False).head(10)
                bottom10_accuracy = accuracy_series.sort_values(ascending=True).head(10)

                # Create a figure with a 2x2 GridSpec:
                # Top row: bar charts (each column one chart)
                # Bottom row: tables for the corresponding chart.
                fig_perc = plt.figure(figsize=(8.27, 11.69))
                gs = fig_perc.add_gridspec(nrows=2, ncols=2, height_ratios=[0.6, 0.4])

                # Top 10 accuracy bar chart (left)
                ax_perc_top = fig_perc.add_subplot(gs[0, 0])
                ax_perc_top.bar(top10_accuracy.index.astype(str), top10_accuracy.values, color='#66c2a5')
                ax_perc_top.set_title("Top 10 Accuracy (%)")
                ax_perc_top.set_xlabel("MUTCD Code")
                ax_perc_top.set_ylabel("Accuracy (%)")

                # Bottom 10 accuracy bar chart (right)
                ax_perc_bottom = fig_perc.add_subplot(gs[0, 1])
                ax_perc_bottom.bar(bottom10_accuracy.index.astype(str), bottom10_accuracy.values, color='#fc8d62')
                ax_perc_bottom.set_title("Lowest 10 Accuracy (%)")
                ax_perc_bottom.set_xlabel("MUTCD Code")
                ax_perc_bottom.set_ylabel("Accuracy (%)")

                # Prepare data for tables:
                # For top 10:
                top_table_data = []
                for code in top10_accuracy.index:
                    count_file1 = df_all[(df_all["Source"] == "File 1") & (df_all["MUTCD"] == code)].shape[0]
                    count_file2 = df_all[(df_all["Source"] == "File 2") & (df_all["MUTCD"] == code)].shape[0]
                    top_table_data.append([code, f'{top10_accuracy.loc[code]:.1f}%', count_file1, count_file2])
                # For bottom 10:
                bottom_table_data = []
                for code in bottom10_accuracy.index:
                    count_file1 = df_all[(df_all["Source"] == "File 1") & (df_all["MUTCD"] == code)].shape[0]
                    count_file2 = df_all[(df_all["Source"] == "File 2") & (df_all["MUTCD"] == code)].shape[0]
                    bottom_table_data.append([code, f'{bottom10_accuracy.loc[code]:.1f}%', count_file1, count_file2])
                col_labels = ["MUTCD", "Accuracy", "File 1", "File 2"]

                # Add table for top 10 (left-bottom)
                ax_table_top = fig_perc.add_subplot(gs[1, 0])
                ax_table_top.axis('tight')
                ax_table_top.axis('off')
                table_top = ax_table_top.table(cellText=top_table_data, colLabels=col_labels, loc='center')
                table_top.auto_set_font_size(False)
                table_top.set_fontsize(8)

                # Add table for bottom 10 (right-bottom)
                ax_table_bottom = fig_perc.add_subplot(gs[1, 1])
                ax_table_bottom.axis('tight')
                ax_table_bottom.axis('off')
                table_bottom = ax_table_bottom.table(cellText=bottom_table_data, colLabels=col_labels, loc='center')
                table_bottom.auto_set_font_size(False)
                table_bottom.set_fontsize(8)

                fig_perc.tight_layout()
                pdf.savefig(fig_perc)
                plt.close()
            else:
                fig_msg = plt.figure(figsize=(8.27, 11.69))
                ax_msg = fig_msg.add_axes([0, 0, 1, 1])
                ax_msg.axis('off')
                ax_msg.text(0.5, 0.5, "No MUTCD column found in the data.", ha='center', va='center', fontsize=14)
                pdf.savefig(fig_msg)
                plt.close()
    print(f"Conflation report saved to {output_pdf}")


def main():
    # File paths for the two datasets
    mach9_file = r"WDOT/WDOT_Before_Sign_Face_projected_lat_lon.csv"
    client_file = r"WDOT/WDOT_Noise_Reduction_Test_Sign_Face_projected_lat_lon.csv"

    print(f"Reading file: {mach9_file} ...")
    df_mach9 = pd.read_csv(mach9_file, low_memory=False)
    print(f"Reading file: {client_file} ...")
    df_client = pd.read_csv(client_file, low_memory=False)

    # Column names for latitude, longitude, and MUTCD code in both datasets.
    mach9_lat_col, mach9_lon_col = "Latitude", "Longitude"
    client_lat_col, client_lon_col = "Latitude", "Longitude"
    # We assume both datasets include a "MUTCD" column.

    # Base folder for all outputs (CSV, Shapefiles, and PDF)
    base_output_folder = os.path.join("WDOT", "conflation_results")
    os.makedirs(base_output_folder, exist_ok=True)

    # Buffer distances to run
    buffer_list = [5, 10, 12]
    results = []
    for buf in buffer_list:
        result = run_conflation_for_threshold(
            df_mach9, df_client, buf,
            mach9_lat_col, mach9_lon_col,
            client_lat_col, client_lon_col,
            base_output_folder
        )
        results.append(result)

    # Path to the logo image (update as needed)
    logo_path = r"Logos/Mach9_Logo_Black 1.png"
    # Final PDF report path (inside the base output folder)
    pdf_path = os.path.join(base_output_folder, "conflation_report.pdf")
    generate_conflation_report_all(results, logo_path=logo_path, output_pdf=pdf_path)


if __name__ == "__main__":
    main()
