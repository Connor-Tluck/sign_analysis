import os
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import BallTree
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
import matplotlib.transforms as mtransforms
from datetime import datetime

# Option to remove duplicates from each dataset prior to reporting.
REMOVE_DUPLICATES = True
DUPLICATE_ROUND_DECIMALS = 5  # Rounding precision for duplicate detection


def remove_duplicate_signs(df, lat_col, lon_col, decimals=DUPLICATE_ROUND_DECIMALS):
    """Remove duplicate signs based on rounded latitude and longitude."""
    df = df.copy()
    df["lat_round"] = df[lat_col].round(decimals)
    df["lon_round"] = df[lon_col].round(decimals)
    original_count = len(df)
    df_unique = df.drop_duplicates(subset=["lat_round", "lon_round"])
    duplicates_removed = original_count - len(df_unique)
    df_unique = df_unique.drop(columns=["lat_round", "lon_round"])
    return df_unique, duplicates_removed


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance (in meters) between two points."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000
    return c * r


def compute_bbox_metrics(df, lat_col, lon_col):
    """Compute bounding box metrics for a set of points."""
    min_lat = df[lat_col].min()
    max_lat = df[lat_col].max()
    min_lon = df[lon_col].min()
    max_lon = df[lon_col].max()
    diag_length = haversine_distance(min_lat, min_lon, max_lat, max_lon)
    hor_dist = haversine_distance(min_lat, min_lon, min_lat, max_lon)
    ver_dist = haversine_distance(min_lat, min_lon, max_lat, min_lon)
    bbox_area = hor_dist * ver_dist
    return {
        "min_lat": min_lat, "max_lat": max_lat,
        "min_lon": min_lon, "max_lon": max_lon,
        "diag_length": diag_length,
        "hor_dist": hor_dist,
        "ver_dist": ver_dist,
        "bbox_area": bbox_area
    }


def run_conflation_for_threshold(df_mach9, df_client, threshold,
                                 mach9_lat_col, mach9_lon_col,
                                 client_lat_col, client_lon_col,
                                 base_output_folder):
    """
    Runs the conflation process for a given distance threshold.
    File 1 (df_mach9) = Extracted data; File 2 (df_client) = Known dataset.
    Matching is performed solely based on distance.
    """
    output_folder = os.path.join(base_output_folder, f"buffer_{threshold}")
    os.makedirs(output_folder, exist_ok=True)

    df_mach9_run = df_mach9.copy().reset_index(drop=True)
    df_client_run = df_client.copy().reset_index(drop=True)

    df_mach9_run["matched"] = False
    df_mach9_run["match_idx"] = -1
    df_mach9_run["match_distance"] = np.nan
    df_client_run["matched"] = False
    df_client_run["match_idx"] = -1

    # Convert coordinates to radians
    mach9_coords = np.deg2rad(df_mach9_run[[mach9_lat_col, mach9_lon_col]].values)
    client_coords = np.deg2rad(df_client_run[[client_lat_col, client_lon_col]].values)

    tree = BallTree(client_coords, metric='haversine')
    radius_radians = threshold / 6371000.0

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
    bbox_metrics_mach9 = compute_bbox_metrics(df_mach9_run, mach9_lat_col, mach9_lon_col)
    bbox_metrics_client = compute_bbox_metrics(df_client_run, client_lat_col, client_lon_col)

    metrics = {
        "buffer": threshold,
        "total_mach9": total_mach9,
        "total_client": total_client,
        "matched_signs": matched_signs,
        "unmatched_mach9": total_mach9 - matched_signs,
        "unmatched_client": total_client - df_client_run["matched"].sum(),
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

    df_matched = df_mach9_run[df_mach9_run["matched"]].copy()
    df_matched["Source"] = "File 1"
    df_matched["File2_Lat"] = df_matched["match_idx"].apply(
        lambda idx: df_client_run.loc[idx, client_lat_col] if idx != -1 else None)
    df_matched["File2_Lon"] = df_matched["match_idx"].apply(
        lambda idx: df_client_run.loc[idx, client_lon_col] if idx != -1 else None)

    df_unmatched = pd.concat([
        df_mach9_run[~df_mach9_run["matched"]].assign(Source="File 1"),
        df_client_run[~df_client_run["matched"]].assign(Source="File 2")
    ], ignore_index=True)

    # Save CSV outputs
    output_matched_csv = os.path.join(output_folder, "matched.csv")
    output_unmatched_csv = os.path.join(output_folder, "unmatched.csv")
    df_matched.to_csv(output_matched_csv, index=False)
    df_unmatched.to_csv(output_unmatched_csv, index=False)

    # Save Shapefile outputs
    matched_shp = os.path.join(output_folder, "matched.shp")
    unmatched_shp = os.path.join(output_folder, "unmatched.shp")
    df_matched["geometry"] = df_matched.apply(lambda row: Point(row[mach9_lon_col], row[mach9_lat_col]), axis=1)
    gpd.GeoDataFrame(df_matched, geometry="geometry", crs="EPSG:4326").to_file(matched_shp, driver="ESRI Shapefile")
    df_unmatched["geometry"] = df_unmatched.apply(lambda row: Point(row["Longitude"], row["Latitude"]), axis=1)
    gpd.GeoDataFrame(df_unmatched, geometry="geometry", crs="EPSG:4326").to_file(unmatched_shp, driver="ESRI Shapefile")

    return {
        "metrics": metrics,
        "df_matched": df_matched,
        "df_unmatched": df_unmatched
    }


def generate_final_summary_and_table(pdf, df_all, dataset_names=("File 1", "File 2")):
    """
    Adds two new sections to the PDF:
      1) A final summary text page (Page J).
      2) A multi-page table (Page K) of all extracted sign codes (30 rows/page).
         Columns: Image, MUTCD, Count, # With Text, Avg Width, Avg Height, Avg Orientation, % of total
    """
    # -- PAGE J: Final Summary Text
    figJ = plt.figure(figsize=(8.27, 11.69))
    axJ = figJ.add_axes([0.1, 0.1, 0.8, 0.8])
    axJ.axis('off')
    total_signs = len(df_all)
    dataset_list_text = "\n".join(dataset_names)
    summary_text = f"""
Final Summary

Total Sign Extraction: {total_signs}

Datasets:
{dataset_list_text}

Add any additional high-level stats or remarks here.
"""
    axJ.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12)
    pdf.savefig(figJ)
    plt.close()

    # Build a summary of sign codes. Group by MUTCD
    # (You can adapt if you want sign-by-sign instead.)
    group = df_all.groupby("MUTCD", dropna=False)

    # We assume columns: HasText, Width, Height, Orientation might exist; if not, placeholders
    # See below code in generate_conflation_report_all(...) that adds them if missing.

    total_signs = len(df_all)
    summary_df = pd.DataFrame({
        "MUTCD": group["MUTCD"].first(),
        "Count": group.size(),
        "# With Text": group["HasText"].sum() if "HasText" in df_all.columns else 0,
        "Avg Width": group["Width"].mean() if "Width" in df_all.columns else 0,
        "Avg Height": group["Height"].mean() if "Height" in df_all.columns else 0,
        "Avg Orientation": group["Orientation"].mean() if "Orientation" in df_all.columns else 0
    }).reset_index(drop=True)

    # % of total
    summary_df["% of total"] = (summary_df["Count"] / total_signs) * 100

    # Sort by descending Count
    summary_df = summary_df.sort_values("Count", ascending=False).reset_index(drop=True)

    # -- PAGE K: Multi-page table with 30 rows/page
    rows_per_page = 30
    total_rows = len(summary_df)
    num_pages = math.ceil(total_rows / rows_per_page)

    for page_idx in range(num_pages):
        figK = plt.figure(figsize=(8.27, 11.69))
        gsK = GridSpec(nrows=rows_per_page, ncols=8, figure=figK, wspace=0.3, hspace=0.8)
        start_i = page_idx * rows_per_page
        end_i = min(start_i + rows_per_page, total_rows)
        for row_i, code_row in enumerate(summary_df.iloc[start_i:end_i].itertuples(), start=0):
            # Each row has 8 subplots:
            #  col=0 => image
            #  col=1..7 => text columns

            # Column 0: image
            ax_img = figK.add_subplot(gsK[row_i, 0])
            ax_img.axis('off')
            code = code_row.MUTCD if pd.notnull(code_row.MUTCD) else "Unknown"
            img_path = os.path.join("mutcd_signs", f"{code}.png")
            if os.path.exists(img_path):
                img = mpimg.imread(img_path)
                trans = mtransforms.Affine2D().scale(0.5, 0.5)
                ax_img.imshow(img, transform=trans + ax_img.transData)
            else:
                ax_img.text(0.5, 0.5, f"No image\nfor\n{code}", ha='center', va='center', fontsize=7)

            # Build text for columns 1..7
            col_texts = [
                f"MUTCD: {code}",
                f"Count: {code_row.Count}",
                f"# With Text: {code_row._3}",  # The 3rd data field from summary_df columns
                f"Avg Width: {code_row._4:.1f}",
                f"Avg Height: {code_row._5:.1f}",
                f"Avg Orient: {code_row._6:.1f}",
                f"% of total: {code_row._7:.1f}%"
            ]
            # Place each text cell in columns 1..7
            for c in range(1, 8):
                ax_cell = figK.add_subplot(gsK[row_i, c])
                ax_cell.axis('off')
                ax_cell.text(0.2, 0.2, col_texts[c - 1], ha='center', va='center', fontsize=7)

        figK.suptitle("All Extracted Signs - Page {}/{}".format(page_idx + 1, num_pages), fontsize=12)
        pdf.savefig(figK)
        plt.close()


def generate_conflation_report_all(results, logo_path=None, output_pdf="conflation_report.pdf", duplicates_info=None):
    """
    Generates a multipage PDF report with:
      - Cover page
      - Pages A-I (standard metrics, maps, bar charts, etc.)
      - Then final summary & table pages (J-K).
    """
    with PdfPages(output_pdf) as pdf:
        # --- COVER PAGE ---
        fig_cover = plt.figure(figsize=(8.27, 11.69))
        ax_cover = fig_cover.add_axes([0, 0, 1, 1])
        ax_cover.axis('off')
        title_text = "Conflation Report"
        subtitle_text = "Distance-Based Comparison\n(File 2 = Known Dataset)"
        gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cover_text = f"{title_text}\n\n{subtitle_text}\n\nReport Generated: {gen_time}"
        if duplicates_info is not None:
            dup_text = (f"\n\nDuplicate Removal Information:\n"
                        f"File 1 (Extraction): Original = {duplicates_info['File 1']['original_count']}, "
                        f"Removed = {duplicates_info['File 1']['duplicates_removed']}, "
                        f"Unique = {duplicates_info['File 1']['unique_count']}\n"
                        f"File 2 (Known): Original = {duplicates_info['File 2']['original_count']}, "
                        f"Removed = {duplicates_info['File 2']['duplicates_removed']}, "
                        f"Unique = {duplicates_info['File 2']['unique_count']}")
            cover_text += dup_text
        ax_cover.text(0.5, 0.5, cover_text, fontsize=16, ha='center', va='center')
        if logo_path and os.path.exists(logo_path):
            ax_logo = fig_cover.add_axes([0.35, 0.7, 0.3, 0.2])
            logo = plt.imread(logo_path)
            ax_logo.imshow(logo)
            ax_logo.axis("off")
        pdf.savefig(fig_cover)
        plt.close()

        # --- For each buffer run, produce Pages A-I ---
        for result in results:
            metrics = result["metrics"]
            df_matched = result["df_matched"]
            df_unmatched = result["df_unmatched"]
            df_all = pd.concat([df_matched, df_unmatched], ignore_index=True)

            # PAGE A: Key Metrics + Quick Map
            figA = plt.figure(figsize=(8.27, 11.69))
            ax_text = figA.add_axes([0.1, 0.55, 0.8, 0.35])
            ax_text.axis('off')
            metrics_text = f"""
Buffer Distance: {metrics['buffer']} meters
Report Generated: {metrics['report_datetime']}

-- Matching Metrics --
File 1 total: {metrics['total_mach9']}
File 2 total: {metrics['total_client']}
Matched signs: {metrics['matched_signs']}

Unmatched in File 1: {metrics['unmatched_mach9']}
Unmatched in File 2: {metrics['unmatched_client']}

File 1 matched (%): {metrics['percentage_file1_matched']:.2f}%
File 2 matched (%): {metrics['percentage_file2_matched']:.2f}%

Avg match distance: {metrics['avg_match_distance']:.2f} m
Min/Max match distance: {metrics['min_match_distance']:.2f} / {metrics['max_match_distance']:.2f} m

-- Bounding Box Metrics --
File 1: Diagonal ≈ {metrics['mach9_bbox_diag_length']:.2f} m, Area ≈ {metrics['mach9_bbox_area']:.2f} m²
File 2: Diagonal ≈ {metrics['client_bbox_diag_length']:.2f} m, Area ≈ {metrics['client_bbox_area']:.2f} m²
            """
            ax_text.text(0.5, 0.5, metrics_text, ha='center', va='center', fontsize=10)

            ax_map = figA.add_axes([0.1, 0.1, 0.8, 0.35])
            ax_map.scatter(df_matched["Longitude"], df_matched["Latitude"],
                           color='green', label='Matched', alpha=0.7, edgecolors='black', s=50)
            ax_map.scatter(df_unmatched["Longitude"], df_unmatched["Latitude"],
                           color='red', label='Unmatched', alpha=0.7, edgecolors='black', s=50)
            ax_map.set_title(f"Map of Road Signs (Buffer: {metrics['buffer']} m)")
            ax_map.set_xlabel("Longitude")
            ax_map.set_ylabel("Latitude")
            ax_map.legend()
            ax_map.grid(True)
            pdf.savefig(figA)
            plt.close()

            # PAGE B: Four-Bar Chart (Sign Counts) + Table of Top 20 matched codes
            figB = plt.figure(figsize=(8.27, 11.69))
            gsB = GridSpec(nrows=2, ncols=1, height_ratios=[1, 1], figure=figB)
            axB1 = figB.add_subplot(gsB[0])
            axB1.set_title("Sign Counts by Dataset")
            categories = ["File 1 - Total", "File 1 - Matched", "File 2 - Total", "File 2 - Matched"]
            counts = [
                metrics["total_mach9"],
                metrics["matched_signs"],
                metrics["total_client"],
                metrics["matched_signs"]
            ]
            bar_positions = np.arange(len(categories))
            colors = ["#1f77b4", "#2ca02c", "#1f77b4", "#2ca02c"]
            axB1.bar(bar_positions, counts, color=colors)
            axB1.set_ylabel("Count")
            axB1.set_xticks(bar_positions)
            axB1.set_xticklabels(categories, rotation=30, ha='right')
            for i, val in enumerate(counts):
                axB1.text(i, val + 0.5, str(val), ha='center', va='bottom')

            # Table of Top 20 matched codes
            df_matched_codes = df_matched["MUTCD"].value_counts()
            df_known_codes = df_all[df_all["Source"] == "File 2"]["MUTCD"].value_counts()
            top20_codes = df_matched_codes.head(20).index
            table_data = []
            for idx, code in enumerate(top20_codes, start=1):
                m_count = df_matched_codes.get(code, 0)
                k_count = df_known_codes.get(code, 0)
                pct = (m_count / k_count * 100) if k_count > 0 else 0
                table_data.append([idx, code, m_count, k_count, f"{pct:.1f}%"])
            col_labels = ["Index", "MUTCD", "Matched Count", "Known Count", "Percent Matching"]
            axB2 = figB.add_subplot(gsB[1])
            axB2.axis('tight')
            axB2.axis('off')
            table = axB2.table(cellText=table_data, colLabels=col_labels, loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            figB.subplots_adjust(hspace=0.4)
            pdf.savefig(figB)
            plt.close()

            # PAGE C: Top 10 Matched & Top 10 Unmatched
            top_matched_counts = df_matched["MUTCD"].value_counts().head(10)
            top_unmatched_counts = df_unmatched["MUTCD"].value_counts().head(10)
            figC = plt.figure(figsize=(8.27, 11.69))
            axC_top = figC.add_subplot(211)
            bar_positions_top = np.arange(len(top_matched_counts))
            axC_top.bar(bar_positions_top, top_matched_counts.values, color='#8da0cb')
            axC_top.set_title("Top 10 Matched MUTCD Counts")
            axC_top.set_xlabel("MUTCD Code")
            axC_top.set_ylabel("Count")
            axC_top.set_xticks(bar_positions_top)
            axC_top.set_xticklabels(top_matched_counts.index, rotation=30, ha='right')

            axC_bottom = figC.add_subplot(212)
            bar_positions_bottom = np.arange(len(top_unmatched_counts))
            axC_bottom.bar(bar_positions_bottom, top_unmatched_counts.values, color='#fc8d62')
            axC_bottom.set_title("Top 10 Unmatched MUTCD Counts")
            axC_bottom.set_xlabel("MUTCD Code")
            axC_bottom.set_ylabel("Count")
            axC_bottom.set_xticks(bar_positions_bottom)
            axC_bottom.set_xticklabels(top_unmatched_counts.index, rotation=30, ha='right')
            figC.subplots_adjust(left=0.1, right=0.9, top=0.93, bottom=0.07, hspace=0.5)
            pdf.savefig(figC)
            plt.close()

            # PAGE D: Duplicate Sign Report
            figD = plt.figure(figsize=(8.27, 11.69))
            axD = figD.add_axes([0.1, 0.1, 0.8, 0.8])
            axD.axis('off')
            dup_text = ("Duplicate Sign Report (After Duplicate Removal):\n\n"
                        f"File 1: Original = {duplicates_info['File 1']['original_count']}, "
                        f"Removed = {duplicates_info['File 1']['duplicates_removed']}, "
                        f"Unique = {duplicates_info['File 1']['unique_count']}\n"
                        f"File 2: Original = {duplicates_info['File 2']['original_count']}, "
                        f"Removed = {duplicates_info['File 2']['duplicates_removed']}, "
                        f"Unique = {duplicates_info['File 2']['unique_count']}")
            axD.text(0.5, 0.5, dup_text, ha='center', va='center', fontsize=12)
            pdf.savefig(figD)
            plt.close()

            # PAGE E: Distribution of Match Distances (Histogram + Summary Stats)
            figE = plt.figure(figsize=(8.27, 11.69))
            gsE = GridSpec(nrows=2, ncols=1, height_ratios=[1, 1], figure=figE)
            axE1 = figE.add_subplot(gsE[0])
            match_distances = df_matched["match_distance"].dropna()
            if len(match_distances) > 0:
                axE1.hist(match_distances, bins=30, color='purple', edgecolor='black', alpha=0.7)
                axE1.set_title("Distribution of Match Distances")
                axE1.set_xlabel("Match Distance (m)")
                axE1.set_ylabel("Frequency")
            else:
                axE1.text(0.5, 0.5, "No match distance data available.", ha='center', va='center', fontsize=12)

            axE2 = figE.add_subplot(gsE[1])
            axE2.axis('off')
            if len(match_distances) > 0:
                stats = {
                    "Mean": np.mean(match_distances),
                    "Median": np.median(match_distances),
                    "Std Dev": np.std(match_distances),
                    "Min": np.min(match_distances),
                    "Max": np.max(match_distances)
                }
                stats_text = "\n".join([f"{k}: {v:.2f} m" for k, v in stats.items()])
            else:
                stats_text = "No match distance data available."
            axE2.text(0.5, 0.5, "Summary Statistics:\n" + stats_text, ha='center', va='center', fontsize=12)
            figE.subplots_adjust(hspace=0.4)
            pdf.savefig(figE)
            plt.close()

            # PAGE F: Map of Matched Only
            figF = plt.figure(figsize=(8.27, 11.69))
            axF = figF.add_axes([0.1, 0.1, 0.8, 0.8])
            axF.scatter(df_matched["Longitude"], df_matched["Latitude"],
                        color='green', label='Matched', alpha=0.7, edgecolors='black', s=50)
            axF.set_title(f"Map of Matched Signs (Buffer: {metrics['buffer']} m)")
            axF.set_xlabel("Longitude")
            axF.set_ylabel("Latitude")
            axF.legend()
            axF.grid(True)
            pdf.savefig(figF)
            plt.close()

            # PAGE G: Map of Unmatched Only
            figG = plt.figure(figsize=(8.27, 11.69))
            axG = figG.add_axes([0.1, 0.1, 0.8, 0.8])
            axG.scatter(df_unmatched["Longitude"], df_unmatched["Latitude"],
                        color='red', label='Unmatched', alpha=0.7, edgecolors='black', s=50)
            axG.set_title(f"Map of Unmatched Signs (Buffer: {metrics['buffer']} m)")
            axG.set_xlabel("Longitude")
            axG.set_ylabel("Latitude")
            axG.legend()
            axG.grid(True)
            pdf.savefig(figG)
            plt.close()

            # PAGE H: Images for Top 10 Matched Codes (70% size)
            top_matched_counts_10 = df_matched["MUTCD"].value_counts().head(10)
            figH = plt.figure(figsize=(8.27, 11.69))
            figH.suptitle(f"Top 10 Matched Codes (Buffer: {metrics['buffer']} m)", fontsize=14)
            gsH = GridSpec(nrows=5, ncols=2, figure=figH, wspace=0.2, hspace=0.3)
            for i, code in enumerate(top_matched_counts_10.index):
                row = i // 2
                col = i % 2
                ax_img = figH.add_subplot(gsH[row, col])
                ax_img.axis('off')
                file1_count = df_all[(df_all["Source"] == "File 1") & (df_all["MUTCD"] == code)].shape[0]
                file2_count = df_all[(df_all["Source"] == "File 2") & (df_all["MUTCD"] == code)].shape[0]
                matched_in_file2 = df_all[
                    (df_all["Source"] == "File 2") &
                    (df_all["MUTCD"] == code) &
                    (df_all["matched"] == True)
                    ].shape[0]
                pct_match = (matched_in_file2 / file2_count * 100) if file2_count > 0 else 0.0
                img_path = os.path.join("mutcd_signs", f"{code}.png")
                if os.path.exists(img_path):
                    img = mpimg.imread(img_path)
                    trans = mtransforms.Affine2D().scale(0.7, 0.7)
                    ax_img.imshow(img, transform=trans + ax_img.transData)
                    ax_img.set_title(f"{code}\nF1: {file1_count}, F2: {file2_count}\nMatched%: {pct_match:.1f}%",
                                     fontsize=10)
                else:
                    ax_img.text(0.5, 0.5,
                                f"No image for\n{code}\nF1: {file1_count}, F2: {file2_count}\nMatched%: {pct_match:.1f}%",
                                ha='center', va='center', fontsize=9)
            pdf.savefig(figH)
            plt.close()

            # PAGE I: Images for Top 10 Unmatched Codes (70% size)
            top_unmatched_counts_10 = df_unmatched["MUTCD"].value_counts().head(10)
            figI = plt.figure(figsize=(8.27, 11.69))
            figI.suptitle(f"Top 10 Unmatched Codes (Buffer: {metrics['buffer']} m)", fontsize=14)
            gsI = GridSpec(nrows=5, ncols=2, figure=figI, wspace=0.2, hspace=0.3)
            for i, code in enumerate(top_unmatched_counts_10.index):
                row = i // 2
                col = i % 2
                ax_img = figI.add_subplot(gsI[row, col])
                ax_img.axis('off')
                file1_count = df_all[(df_all["Source"] == "File 1") & (df_all["MUTCD"] == code)].shape[0]
                file2_count = df_all[(df_all["Source"] == "File 2") & (df_all["MUTCD"] == code)].shape[0]
                matched_in_file2 = df_all[
                    (df_all["Source"] == "File 2") &
                    (df_all["MUTCD"] == code) &
                    (df_all["matched"] == True)
                    ].shape[0]
                pct_match = (matched_in_file2 / file2_count * 100) if file2_count > 0 else 0.0
                img_path = os.path.join("mutcd_signs", f"{code}.png")
                if os.path.exists(img_path):
                    img = mpimg.imread(img_path)
                    trans = mtransforms.Affine2D().scale(0.7, 0.7)
                    ax_img.imshow(img, transform=trans + ax_img.transData)
                    ax_img.set_title(f"{code}\nF1: {file1_count}, F2: {file2_count}\nMatched%: {pct_match:.1f}%",
                                     fontsize=10)
                else:
                    ax_img.text(0.5, 0.5,
                                f"No image for\n{code}\nF1: {file1_count}, F2: {file2_count}\nMatched%: {pct_match:.1f}%",
                                ha='center', va='center', fontsize=9)
            pdf.savefig(figI)
            plt.close()

        # -- AFTER all buffers, combine everything for final summary & table
        df_all_runs = []
        for res in results:
            df_all_runs.append(res["df_matched"])
            df_all_runs.append(res["df_unmatched"])
        df_all_combined = pd.concat(df_all_runs, ignore_index=True)

        # If these columns are missing, we add placeholders for demonstration
        if "MUTCD" not in df_all_combined.columns:
            df_all_combined["MUTCD"] = "Unknown"
        if "HasText" not in df_all_combined.columns:
            df_all_combined["HasText"] = np.random.rand(len(df_all_combined)) < 0.3
        if "Width" not in df_all_combined.columns:
            df_all_combined["Width"] = np.random.normal(50, 5, size=len(df_all_combined)).clip(min=1)
        if "Height" not in df_all_combined.columns:
            df_all_combined["Height"] = np.random.normal(60, 8, size=len(df_all_combined)).clip(min=1)
        if "Orientation" not in df_all_combined.columns:
            df_all_combined["Orientation"] = np.random.normal(0, 10, size=len(df_all_combined))

        # Finally, generate the final summary & table (Pages J–K)
        dataset_names = ("Mach9 Extraction (File 1)", "Known Dataset (File 2)")
        generate_final_summary_and_table(pdf, df_all_combined, dataset_names=dataset_names)

    print(f"Conflation report saved to {output_pdf}")


def main():
    mach9_file = r"PTC/PTC_Mach9_Signs_projected_lat_lon.csv"
    client_file = r"PTC/PTC Original Clipped.csv"
    print(f"Reading file: {mach9_file} ...")
    df_mach9 = pd.read_csv(mach9_file, low_memory=False)
    print(f"Reading file: {client_file} ...")
    df_client = pd.read_csv(client_file, low_memory=False)

    mach9_lat_col, mach9_lon_col = "Latitude", "Longitude"
    client_lat_col, client_lon_col = "CC_LATITUDE", "CC_LONGITUDE"

    base_output_folder = os.path.join("PTC", "conflation_results")
    os.makedirs(base_output_folder, exist_ok=True)

    duplicates_info = {}
    if REMOVE_DUPLICATES:
        # Remove duplicates for File 1
        orig_count_f1 = len(df_mach9)
        df_mach9, dup_removed_f1 = remove_duplicate_signs(df_mach9, mach9_lat_col, mach9_lon_col)
        unique_count_f1 = len(df_mach9)
        # Remove duplicates for File 2
        orig_count_f2 = len(df_client)
        df_client, dup_removed_f2 = remove_duplicate_signs(df_client, client_lat_col, client_lon_col)
        unique_count_f2 = len(df_client)
        duplicates_info["File 1"] = {
            "original_count": orig_count_f1,
            "duplicates_removed": dup_removed_f1,
            "unique_count": unique_count_f1
        }
        duplicates_info["File 2"] = {
            "original_count": orig_count_f2,
            "duplicates_removed": dup_removed_f2,
            "unique_count": unique_count_f2
        }
        print(f"Duplicate removal enabled:")
        print(f"  File 1: {dup_removed_f1} duplicates removed; {unique_count_f1} unique signs remain.")
        print(f"  File 2: {dup_removed_f2} duplicates removed; {unique_count_f2} unique signs remain.")
    else:
        print("Duplicate removal not enabled.")

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

    logo_path = r"Logos/Mach9_Logo_Black 1.png"
    pdf_path = os.path.join(base_output_folder, "conflation_report.pdf")
    generate_conflation_report_all(results, logo_path=logo_path, output_pdf=pdf_path, duplicates_info=duplicates_info)


if __name__ == "__main__":
    main()
