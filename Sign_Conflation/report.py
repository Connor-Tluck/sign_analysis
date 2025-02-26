import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image as PILImage  # For aspect ratio calculations
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from duplicates import remove_duplicate_signs
from distance_matching import run_conflation_for_threshold  # 5m distance matching
from sklearn.cluster import KMeans

# ---------------------
# Global settings and schema mappings
# ---------------------
GOOGLE_API_KEY = 'x'  # Replace with your actual API key

mach9_schema = {"lat": "Latitude", "lon": "Longitude", "mutcd": "MUTCD"}
client_schema = {"lat": "CC_LATITUDE", "lon": "CC_LONGITUDE", "mutcd": "MUTCD"}

REMOVE_DUPLICATES = True
DISTANCE_THRESHOLD = 5  # 5 meters

# Sign category prefix mappings
SIGN_CATEGORY_PREFIXES = {
    "R-": "R-Series (Regulatory Signs)",
    "W-": "W-Series (Warning Signs)",
    "M-": "M-Series (Mileposts & Reference Markers)",
    "I-": "I-Series (Interstate & Freeway Signs)",
    "D-": "D-Series (Guide Signs for Cities & Destinations)",
    "G-": "G-Series (General Guide Signs)",
    "E-": "E-Series (Expressway & Freeway Guide Signs)",
    "S-": "S-Series (School Zone Signs)",
    "OM": "OM-Series (Object Markers)",
    "PR": "PR-Series (Parking & Rest Area Signs)",
    "RR": "RR-Series (Railroad Crossing Signs)"
}


def preserve_aspect_image(path, desired_width, hAlign='CENTER'):
    """Maintains aspect ratio for images placed into the PDF."""
    from reportlab.platypus import Paragraph
    if not os.path.exists(path):
        return Paragraph("Image not found", getSampleStyleSheet()['BodyText'])
    pil_img = PILImage.open(path)
    w, h = pil_img.size
    aspect_ratio = h / float(w)
    new_height = desired_width * aspect_ratio
    return Image(path, width=desired_width, height=new_height, hAlign=hAlign)


def get_category_from_mutcd(mutcd_code):
    if not isinstance(mutcd_code, str):
        return "Unknown"
    code_upper = mutcd_code.upper().replace(" ", "")
    for prefix, category_name in SIGN_CATEGORY_PREFIXES.items():
        if code_upper.startswith(prefix):
            return category_name
        if code_upper and code_upper[0] == prefix[0] and len(code_upper) > 1 and code_upper[1].isdigit():
            return category_name
    return "Unknown"


def categorize_signs(df, mutcd_col="MUTCD"):
    df["SignCategory"] = df[mutcd_col].apply(get_category_from_mutcd)
    return df["SignCategory"].value_counts().to_dict()


# ---------------------
# Google API functions
# ---------------------
def fetch_street_view_image(lat, lon, output_path, heading=180, pitch=-10, fov=90, size="400x400"):
    """Fetches a Google Street View image and saves it locally."""
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": size,
        "location": f"{lat},{lon}",
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
        "key": GOOGLE_API_KEY
    }
    try:
        resp = requests.get(base_url, params=params, timeout=10)
        if resp.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(resp.content)
            return True
        else:
            print(f"Street View API error: {resp.status_code}, {resp.text}")
            return False
    except Exception as e:
        print(f"Street View request failed: {e}")
        return False


def fetch_street_view_metadata(lat, lon, heading=180, pitch=-10, fov=90, size="400x400"):
    """Fetches Google Street View metadata (including capture date)."""
    base_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {
        "size": size,
        "location": f"{lat},{lon}",
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
        "key": GOOGLE_API_KEY
    }
    try:
        resp = requests.get(base_url, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"Street View Metadata API error: {resp.status_code}, {resp.text}")
            return None
    except Exception as e:
        print(f"Street View Metadata request failed: {e}")
        return None


def fetch_static_map_image(lat, lon, output_path, zoom=16, size="400x400", maptype="roadmap"):
    """Fetches a Google Static Map image and saves it locally."""
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "size": size,
        "maptype": maptype,
        "markers": f"color:red|{lat},{lon}",
        "key": GOOGLE_API_KEY
    }
    try:
        resp = requests.get(base_url, params=params, timeout=10)
        if resp.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(resp.content)
            return True
        else:
            print(f"Static Map API error: {resp.status_code}, {resp.text}")
            return False
    except Exception as e:
        print(f"Static Map request failed: {e}")
        return False


# ---------------------
# Image helper functions
# ---------------------
def create_fixed_image(image_path, fixed_width):
    """Loads an image from `image_path` and fixes width while maintaining aspect ratio."""
    if not os.path.exists(image_path):
        return None
    pil_img = PILImage.open(image_path)
    w, h = pil_img.size
    aspect = h / float(w)
    fixed_height = fixed_width * aspect
    return Image(image_path, width=fixed_width, height=fixed_height, hAlign='CENTER')


def create_logo_image(logo_path, max_width=200, max_height=80):
    """Loads a logo image from `logo_path` with max width/height constraints."""
    from reportlab.platypus import Paragraph
    if not os.path.exists(logo_path):
        return Paragraph("Logo not found", getSampleStyleSheet()['BodyText'])
    pil_img = PILImage.open(logo_path)
    w, h = pil_img.size
    aspect = h / float(w)
    new_width = max_width
    new_height = aspect * new_width
    if new_height > max_height:
        new_height = max_height
        new_width = new_height / aspect
    return Image(logo_path, width=new_width, height=new_height)


# ---------------------
# Chart Generators
# ---------------------
def generate_dummy_chart(chart_type, filename):
    """Generates placeholder charts (venn, box, heatmap, line) for demonstration."""
    plt.figure(figsize=(6, 4))
    if chart_type == 'venn':
        plt.text(0.5, 0.5, "Venn Diagram Placeholder", fontsize=14, ha='center')
        plt.axis('off')
    elif chart_type == 'box':
        data = [np.random.normal(0, 1, 100) for _ in range(3)]
        plt.boxplot(data, tick_labels=['Group 1', 'Group 2', 'Group 3'])
        plt.title("Confidence Score Distribution")
    elif chart_type == 'heatmap':
        data = np.random.rand(10, 10)
        plt.imshow(data, cmap='hot', interpolation='nearest')
        plt.title("Sign Density Heatmap")
        plt.colorbar()
    elif chart_type == 'line':
        x = np.arange(10)
        y = np.random.randint(50, 150, 10)
        plt.plot(x, y, marker='o')
        plt.title("Sign Changes Over Time")
        plt.xlabel("Time Period")
        plt.ylabel("Number of Signs")
    else:
        plt.text(0.5, 0.5, f"{chart_type} Chart", fontsize=14, ha='center')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def generate_sign_type_distribution_chart(mach9_cat_counts):
    """Bar chart of sign categories for Mach9 dataset."""
    categories = list(mach9_cat_counts.keys())
    counts = [mach9_cat_counts[c] for c in categories]
    plt.figure(figsize=(8, 4))
    plt.bar(categories, counts, color='#1f77b4')
    plt.title("Sign Type Distribution (Mach9)", fontsize=14)
    plt.xlabel("Sign Category")
    plt.ylabel("Count")
    for i, count in enumerate(counts):
        plt.text(i, count, str(count), ha='center', va='bottom')
    plt.tight_layout()
    sign_type_chart_path = 'sign_type_distribution.png'
    plt.savefig(sign_type_chart_path, bbox_inches='tight')
    plt.close()
    return sign_type_chart_path


def generate_key_metrics_bar_chart(mach9_total, client_total, matched_signs, unmatched_mach9, unmatched_client):
    """
    A bar chart to visualize main comparison metrics: total, matched, unmatched, etc.
    Color-coded for clarity.
    """
    bar_chart_path = 'bar_chart_metrics.png'
    labels = ["Mach9 Total", "Client Total", "Matched (5m)", "M9 Unmatched", "Client Unmatched"]
    values = [mach9_total, client_total, matched_signs, unmatched_mach9, unmatched_client]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]  # Distinctive colors
    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, values, color=colors)
    plt.title("Key Metrics Comparison (Distance Matching)", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, str(yval), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(bar_chart_path, bbox_inches='tight')
    plt.close()
    return bar_chart_path


def generate_percent_diff_chart(duplicates_info):
    """
    Creates a stacked bar chart to show raw duplicates vs. unique for Mach9 and Client,
    as a percentage of each dataset's raw total.
    """
    chart_path = 'percent_diff_chart.png'

    mach9_raw = duplicates_info["Mach9"]["original_count"]
    mach9_dups = duplicates_info["Mach9"]["duplicates_removed"]
    mach9_uniq = duplicates_info["Mach9"]["unique_count"]

    client_raw = duplicates_info["Client"]["original_count"]
    client_dups = duplicates_info["Client"]["duplicates_removed"]
    client_uniq = duplicates_info["Client"]["unique_count"]

    # Convert to percentages of each dataset's raw total
    mach9_dups_perc = (mach9_dups / mach9_raw * 100) if mach9_raw else 0
    mach9_uniq_perc = (mach9_uniq / mach9_raw * 100) if mach9_raw else 0
    client_dups_perc = (client_dups / client_raw * 100) if client_raw else 0
    client_uniq_perc = (client_uniq / client_raw * 100) if client_raw else 0

    # We'll do a stacked bar chart with 2 bars: one for Mach9, one for Client
    labels = ["Mach9", "Client"]
    dups_perc = [mach9_dups_perc, client_dups_perc]
    uniq_perc = [mach9_uniq_perc, client_uniq_perc]

    plt.figure(figsize=(6, 4))
    bottom_bar = plt.bar(labels, dups_perc, color="#ff9999", label="Duplicates (%)")
    top_bar = plt.bar(labels, uniq_perc, bottom=dups_perc, color="#99ff99", label="Unique (%)")

    for i, bar in enumerate(bottom_bar):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval/2, f"{yval:.1f}%", ha="center", va="center", color="black")
    for i, bar in enumerate(top_bar):
        bottom_val = dups_perc[i]
        yval = bottom_val + bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, bottom_val + bar.get_height()/2, f"{bar.get_height():.1f}%",
                 ha="center", va="center", color="black")

    plt.title("Percent Differences: Duplicates vs. Unique", fontsize=14)
    plt.ylabel("Percentage of Raw Total")
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_path, bbox_inches='tight')
    plt.close()
    return chart_path


# ---------------------
# Main Report Generation
# ---------------------
def generate_report(mach9_csv, client_csv, output_pdf, logo_path):
    # 1. Load data
    df_mach9 = pd.read_csv(mach9_csv, low_memory=False)
    df_client = pd.read_csv(client_csv, low_memory=False)
    styles = getSampleStyleSheet()

    # 2. Standardize client columns
    if client_schema["lat"] in df_client.columns:
        df_client.rename(columns={client_schema["lat"]: "Latitude"}, inplace=True)
    else:
        df_client["Latitude"] = None
    if client_schema["lon"] in df_client.columns:
        df_client.rename(columns={client_schema["lon"]: "Longitude"}, inplace=True)
    else:
        df_client["Longitude"] = None
    if client_schema["mutcd"] in df_client.columns:
        df_client.rename(columns={client_schema["mutcd"]: "MUTCD"}, inplace=True)
    else:
        df_client["MUTCD"] = "Unknown"
    if "MUTCD" not in df_mach9.columns:
        df_mach9["MUTCD"] = "Unknown"

    # 3. Capture raw counts (before duplicates)
    raw_mach9_count = len(df_mach9)
    raw_client_count = len(df_client)

    # 4. Remove duplicates if enabled
    duplicates_info = {
        "Mach9": {
            "original_count": raw_mach9_count,
            "duplicates_removed": 0,
            "unique_count": raw_mach9_count
        },
        "Client": {
            "original_count": raw_client_count,
            "duplicates_removed": 0,
            "unique_count": raw_client_count
        }
    }
    if REMOVE_DUPLICATES:
        df_mach9_pre = len(df_mach9)
        df_mach9, dup_removed_mach9 = remove_duplicate_signs(df_mach9, "Latitude", "Longitude", decimals=5)
        df_client_pre = len(df_client)
        df_client, dup_removed_client = remove_duplicate_signs(df_client, "Latitude", "Longitude", decimals=5)

        duplicates_info["Mach9"]["duplicates_removed"] = dup_removed_mach9
        duplicates_info["Mach9"]["unique_count"] = len(df_mach9)
        duplicates_info["Client"]["duplicates_removed"] = dup_removed_client
        duplicates_info["Client"]["unique_count"] = len(df_client)

    # 5. Distance-based matching for 5m
    distance_result = run_conflation_for_threshold(
        df_mach9, df_client,
        threshold=DISTANCE_THRESHOLD,
        mach9_lat_col="Latitude", mach9_lon_col="Longitude",
        client_lat_col="Latitude", client_lon_col="Longitude",
        base_output_folder="distance_output"
    )
    dist_metrics = distance_result["metrics"]
    total_mach9 = dist_metrics["total_mach9"]       # deduplicated Mach9
    total_client = dist_metrics["total_client"]     # deduplicated Client
    matched_signs = dist_metrics["matched_signs"]
    unmatched_mach9 = dist_metrics["unmatched_mach9"]
    unmatched_client = dist_metrics["unmatched_client"]

    # Also retrieve matched/unmatched DataFrames for examples
    df_matched = distance_result["df_matched"]     # Mach9 matched records
    df_unmatched = distance_result["df_unmatched"] # Mach9 or Client unmatched

    # 6. Categorize signs (deduplicated DataFrames)
    mach9_cat_counts = categorize_signs(df_mach9, mutcd_col="MUTCD")
    client_cat_counts = categorize_signs(df_client, mutcd_col="MUTCD")

    # 7. Executive Summary + Approach
    summary_text = f"""
<b>Executive Summary</b><br/><br/>
This report compares Mach9's LiDAR-extracted sign features with a legacy client dataset using a 5‑meter distance matching approach.
Key findings include:<br/>
&nbsp;&nbsp;&bull;&nbsp; Total Mach9 Signs (Raw): {duplicates_info["Mach9"]["original_count"]}<br/>
&nbsp;&nbsp;&bull;&nbsp; Total Client Signs (Raw): {duplicates_info["Client"]["original_count"]}<br/>
&nbsp;&nbsp;&bull;&nbsp; Duplicates Removed - Mach9: {duplicates_info["Mach9"]["duplicates_removed"]}, Client: {duplicates_info["Client"]["duplicates_removed"]}<br/>
&nbsp;&nbsp;&bull;&nbsp; Total Mach9 (No Duplicates): {total_mach9}<br/>
&nbsp;&nbsp;&bull;&nbsp; Total Client (No Duplicates): {total_client}<br/>
&nbsp;&nbsp;&bull;&nbsp; Matched Signs (within 5 m): {matched_signs}<br/>
&nbsp;&nbsp;&bull;&nbsp; Unique to Mach9: {unmatched_mach9}<br/>
&nbsp;&nbsp;&bull;&nbsp; Unique to Client: {unmatched_client}<br/><br/>
<b>Approach and Methodology</b><br/>
1. Standardize column names for latitude, longitude, and MUTCD.<br/>
2. Remove duplicates based on coordinate precision.<br/>
3. Run a distance matching algorithm (5‑meter buffer) via a BallTree (haversine metric).<br/>
4. Compute key metrics based on the matching results.<br/>
5. Generate charts to illustrate distribution, clustering, and discrepancies.
"""

    # 8. Build the two new tables:

    # ---- Table 1: Overview Table ----
    total_raw = duplicates_info["Mach9"]["original_count"] + duplicates_info["Client"]["original_count"]
    mach9_raw_perc = (duplicates_info["Mach9"]["original_count"] / total_raw * 100) if total_raw else 0
    client_raw_perc = (duplicates_info["Client"]["original_count"] / total_raw * 100) if total_raw else 0
    mach9_dup_perc = 0
    if duplicates_info["Mach9"]["original_count"]:
        mach9_dup_perc = duplicates_info["Mach9"]["duplicates_removed"] / duplicates_info["Mach9"]["original_count"] * 100
    client_dup_perc = 0
    if duplicates_info["Client"]["original_count"]:
        client_dup_perc = duplicates_info["Client"]["duplicates_removed"] / duplicates_info["Client"]["original_count"] * 100
    combined_dedup = duplicates_info["Mach9"]["unique_count"] + duplicates_info["Client"]["unique_count"]
    mach9_no_dup_perc = (duplicates_info["Mach9"]["unique_count"] / combined_dedup * 100) if combined_dedup else 0
    client_no_dup_perc = (duplicates_info["Client"]["unique_count"] / combined_dedup * 100) if combined_dedup else 0

    data_overview = [
        ["Metric",                          "Mach9",                                            "Client",                                            "Mach9 Percentage",                           "Client Percentage"],
        ["Total Signs (Raw)",              duplicates_info["Mach9"]["original_count"],         duplicates_info["Client"]["original_count"],         f"{mach9_raw_perc:.2f}%",                     f"{client_raw_perc:.2f}%"],
        ["Duplicates",                     duplicates_info["Mach9"]["duplicates_removed"],     duplicates_info["Client"]["duplicates_removed"],     f"{mach9_dup_perc:.2f}%",                     f"{client_dup_perc:.2f}%"],
        ["Total Signs (without duplicates)",duplicates_info["Mach9"]["unique_count"],          duplicates_info["Client"]["unique_count"],          f"{mach9_no_dup_perc:.2f}%",                  f"{client_no_dup_perc:.2f}%"]
    ]

    # ---- Table 2: Comparison Table ----
    matched_mach9_perc = (matched_signs / total_mach9 * 100) if total_mach9 else 0
    unique_mach9_perc = (unmatched_mach9 / total_mach9 * 100) if total_mach9 else 0
    unique_client_perc = (unmatched_client / total_client * 100) if total_client else 0

    data_comparison = [
        ["Metric", "Count", "Total in Dataset", "Percentage", "Notes"],
        ["Matched Signs (5m)",
         matched_signs,
         total_mach9,
         f"{matched_mach9_perc:.2f}%",
         "Signs matched across both datasets within 5 m buffer."],
        ["Unique Signs in Mach9",
         unmatched_mach9,
         total_mach9,
         f"{unique_mach9_perc:.2f}%",
         "Signs in the Mach9 dataset not present in the Client dataset."],
        ["Unique Signs in Client",
         unmatched_client,
         total_client,
         f"{unique_client_perc:.2f}%",
         "Signs in the Client dataset not present in the Mach9 dataset."]
    ]

    # 9. Generate Charts
    bar_chart_path = generate_key_metrics_bar_chart(
        mach9_total=total_mach9,
        client_total=total_client,
        matched_signs=matched_signs,
        unmatched_mach9=unmatched_mach9,
        unmatched_client=unmatched_client
    )
    percent_diff_chart_path = generate_percent_diff_chart(duplicates_info)
    venn_chart_path = 'venn_diagram.png'
    generate_dummy_chart('venn', venn_chart_path)
    box_plot_path = 'box_plot_confidence.png'
    generate_dummy_chart('box', box_plot_path)
    sign_type_chart_path = generate_sign_type_distribution_chart(mach9_cat_counts)
    heatmap_path = 'heatmap_sign_density.png'
    generate_dummy_chart('heatmap', heatmap_path)
    line_chart_path = 'line_chart_temporal.png'
    generate_dummy_chart('line', line_chart_path)

    # 10. Spatial Distribution and Cluster Analysis
    cluster_chart_mach9 = "cluster_analysis_mach9.png"
    if "Longitude" in df_mach9.columns and "Latitude" in df_mach9.columns and not df_mach9.empty:
        coords = df_mach9[["Longitude", "Latitude"]].dropna().values
        k = 5 if len(coords) >= 5 else 1
        kmeans = KMeans(n_clusters=k, random_state=0).fit(coords)
        cluster_labels = kmeans.labels_
        plt.figure(figsize=(6, 4))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=cluster_labels, cmap="viridis", s=10)
        plt.title("Cluster Analysis of Mach9 Sign Locations", fontsize=14)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.colorbar(scatter, label="Cluster")
        plt.tight_layout()
        plt.savefig(cluster_chart_mach9, bbox_inches='tight')
        plt.close()
    else:
        cluster_chart_mach9 = None

    cluster_chart_client = "cluster_analysis_client.png"
    if "Longitude" in df_client.columns and "Latitude" in df_client.columns and not df_client.empty:
        coords = df_client[["Longitude", "Latitude"]].dropna().values
        k = 5 if len(coords) >= 5 else 1
        kmeans = KMeans(n_clusters=k, random_state=0).fit(coords)
        cluster_labels = kmeans.labels_
        plt.figure(figsize=(6, 4))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=cluster_labels, cmap="viridis", s=10)
        plt.title("Cluster Analysis of Client Sign Locations", fontsize=14)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.colorbar(scatter, label="Cluster")
        plt.tight_layout()
        plt.savefig(cluster_chart_client, bbox_inches='tight')
        plt.close()
    else:
        cluster_chart_client = None

    # 11. Sign Dimensions and Condition Analysis
    dimension_chart_path = "dimension_analysis.png"
    if "Width" in df_mach9.columns and "Height" in df_mach9.columns:
        plt.figure(figsize=(6, 4))
        data_width = df_mach9["Width"].dropna()
        data_height = df_mach9["Height"].dropna()
        plt.boxplot([data_width, data_height], tick_labels=["Width", "Height"])
        plt.title("Sign Dimensions (Mach9)", fontsize=14)
        plt.tight_layout()
        plt.savefig(dimension_chart_path, bbox_inches='tight')
        plt.close()
    else:
        dimension_chart_path = None

    # 12. Quality Metrics and Performance Evaluation
    quality_chart_path = "quality_metrics.png"
    quality_metrics_text = ""
    if "Confidence" in df_mach9.columns and not df_mach9["Confidence"].isna().all():
        plt.figure(figsize=(6, 4))
        data_conf = df_mach9["Confidence"].dropna()
        plt.hist(data_conf, bins=20, color="green", alpha=0.7)
        plt.title("Confidence Score Distribution (Mach9)", fontsize=14)
        plt.xlabel("Confidence Score")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(quality_chart_path, bbox_inches='tight')
        plt.close()
        avg_conf = data_conf.mean()
        median_conf = data_conf.median()
        std_conf = data_conf.std()
        quality_metrics_text = f"Average Confidence: {avg_conf:.2f}, Median: {median_conf:.2f}, Std Dev: {std_conf:.2f}"
    else:
        quality_chart_path = None
        quality_metrics_text = "No confidence data found or insufficient data to display."

    # 13. Sign Category Breakdown Table
    cat_table_data = [['Category', 'Mach9 Count', 'Client Count']]
    for c in sorted(set(mach9_cat_counts.keys()).union(client_cat_counts.keys())):
        mach9_count = mach9_cat_counts.get(c, 0)
        client_count = client_cat_counts.get(c, 0)
        cat_table_data.append([c, str(mach9_count), str(client_count)])

    # ---------------------
    # 14. Case Studies and Examples
    # ---------------------
    # We'll show 3 sections:
    #  A) Unique Signs in Mach9 Dataset [unmatched_mach9]
    #  B) Unique Signs in Client Dataset [unmatched_client]
    #  C) Matching Signs [matched_signs]
    #
    # For each, we sample up to 10 examples from the relevant subset.

    # Make sure folders for images exist
    street_folder = "street_view"
    map_folder = "map_view"
    os.makedirs(street_folder, exist_ok=True)
    os.makedirs(map_folder, exist_ok=True)

    # Create a helper to build 10 examples from a given DataFrame
    def build_examples_for_subset(df_subset, max_examples=10, dataset_label="Mach9"):
        """
        Returns a list of (image_row, description) for up to `max_examples` rows in df_subset.
        We'll fetch street view, map images, and build a paragraph describing each sign.
        """
        from reportlab.platypus import Paragraph

        if df_subset.empty:
            return []

        # Random sample up to `max_examples`
        df_sample = df_subset.sample(n=min(max_examples, len(df_subset)), random_state=42).copy()
        examples_list = []

        for idx, row in df_sample.iterrows():
            # If the row is from Mach9, lat/lon are in (Latitude, Longitude)
            # If from Client, we still use (Latitude, Longitude) because we standardized columns
            lat = row.get("Latitude", None)
            lon = row.get("Longitude", None)
            code = row.get("MUTCD", "Unknown")
            mach9_url = row.get("URL", "Unknown")
            text_info = row.get("Text", "No additimonal text")

            # Build file paths for images
            street_img_path = os.path.join(street_folder, f"{dataset_label}_{idx}_street.jpg")
            map_img_path = os.path.join(map_folder, f"{dataset_label}_{idx}_map.jpg")

            # Attempt to fetch images if lat/lon is valid
            if lat and lon and not pd.isna(lat) and not pd.isna(lon):
                fetch_street_view_image(lat, lon, street_img_path, heading=180, pitch=-10, fov=90, size="400x400")
                fetch_static_map_image(lat, lon, map_img_path, zoom=16, size="400x400")
                metadata = fetch_street_view_metadata(lat, lon, heading=180, pitch=-10, fov=90, size="400x400")
                capture_date = metadata.get("date", "Unknown") if metadata else "Unknown"

            else:
                capture_date = "Unknown"

            # Attempt to load MUTCD sign image
            mutcd_sign_path = os.path.join("../mutcd_signs", f"{code}.png")
            street_img = create_fixed_image(street_img_path, 120) if os.path.exists(street_img_path) else Paragraph(f"No Street View for <b>{code}</b>", styles['BodyText'])
            map_img = create_fixed_image(map_img_path, 120) if os.path.exists(map_img_path) else Paragraph(f"No Map View for <b>{code}</b>", styles['BodyText'])
            mutcd_img = create_fixed_image(mutcd_sign_path, 25) if os.path.exists(mutcd_sign_path) else Paragraph(f"No MUTCD Sign Image for <b>{code}</b>", styles['BodyText'])

            # Build a Street View URL for direct link
            street_view_url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
            description = (f"<b>MUTCD:</b> {code}<br/>"
                           f"<b>Text:</b> {text_info}<br/>"
                           f"<b>Dataset:</b> {dataset_label}<br/>"
                           f"<b>Lat/Lon:</b> {lat}, {lon}<br/>"
                           f"<b>Street View URL:</b> <a href='{street_view_url}'>{street_view_url}</a><br/>"
                            f"<b>Mach9 URL:</b> <a href='{mach9_url}'>{mach9_url}</a><br/>"
                           f"<b>Capture Date:</b> {capture_date}<br/>")
            examples_list.append(([street_img, map_img, mutcd_img],
                                  Paragraph(description, styles['BodyText'])))
        return examples_list

    # Split the unmatched DataFrame by source
    # (In distance_matching.py, "Source" is "File 1" or "File 2". We'll assume "File 1" = Mach9, "File 2" = Client.)
    df_unmatched_mach9 = df_unmatched[df_unmatched["Source"] == "File 1"]
    df_unmatched_client = df_unmatched[df_unmatched["Source"] == "File 2"]

    # Build examples
    examples_unique_mach9 = build_examples_for_subset(df_unmatched_mach9, max_examples=10, dataset_label="Mach9")
    examples_unique_client = build_examples_for_subset(df_unmatched_client, max_examples=10, dataset_label="Client")
    # For matched, we just sample from df_matched
    # (We keep "File 1" = Mach9 lat/lon for fetching images)
    # Note: If you want the client lat/lon for matched signs, you'd adjust accordingly.
    examples_matched = build_examples_for_subset(df_matched, max_examples=10, dataset_label="Mach9_Matched")

    # 15. Appendix text
    appendix_text = """
<b>Extended Methodology:</b><br/>
- Data standardization: Columns were aligned across datasets.<br/>
- Duplicate removal: Duplicates were removed based on geographic coordinates (5 decimal precision).<br/>
- Distance matching: A 5‑meter buffer was used with a BallTree and haversine metric for matching signs.<br/>
- Spatial and cluster analysis: Data was analyzed for clustering and coverage patterns.<br/>
- Quality metrics: Confidence scores and sign dimensions were evaluated.<br/><br/>
<b>Glossary:</b><br/>
- <i>Matched Signs (5 m Buffer):</i> Signs within 5 meters across the two datasets.<br/>
- <i>Unique Signs:</i> Signs present in one dataset but not in the other (by distance threshold).<br/>
- <i>Duplicates Removed:</i> Duplicate sign entries removed during preprocessing.<br/>
- <i>MUTCD Codes:</i> Standardized sign classification codes used by Mach9.
"""

    # ---------------------
    # Build the PDF
    # ---------------------
    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    story = []

    # --- Page 1: Cover Page ---
    story.append(Paragraph("Comparative Analysis of Sign Detections: Mach9 vs. Client Dataset", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Evaluating Accuracy, Coverage, and Completeness of Extracted Traffic Signs", styles['Heading2']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Prepared by: Your Company/Team Name", styles['Normal']))
    story.append(Paragraph("Date: Report Creation Date", styles['Normal']))
    story.append(Spacer(1, 24))
    story.append(create_logo_image(logo_path, max_width=200, max_height=80))
    story.append(Spacer(1, 48))
    story.append(PageBreak())

    # --- Page 2: Executive Summary & Approach ---
    story.append(Paragraph(summary_text, styles['BodyText']))
    story.append(Spacer(1, 24))
    story.append(PageBreak())

    # --- Page 3: Table 1 (Overview Table) & Percent Differences Chart ---
    story.append(Paragraph("Table 1: Overview Table", styles['Heading1']))
    table_overview = Table(data_overview, hAlign='CENTER')
    table_overview.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table_overview)
    story.append(Spacer(1, 12))

    # Insert the stacked bar chart showing duplicates vs unique as percentages
    story.append(Paragraph("Percent Differences Visualization", styles['Heading2']))
    if os.path.exists(percent_diff_chart_path):
        story.append(Image(percent_diff_chart_path, width=480, hAlign='CENTER'))
    story.append(Spacer(1, 24))
    story.append(PageBreak())

    # --- Page 4: Table 2 (Comparison Table) & Key Metrics Bar Chart ---
    story.append(Paragraph("Table 2: Comparison Table", styles['Heading1']))
    table_comparison = Table(data_comparison, hAlign='CENTER')
    table_comparison.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table_comparison)
    story.append(Spacer(1, 12))

    # Insert the key metrics comparison bar chart
    story.append(Paragraph("Key Metrics Comparison Bar Chart", styles['Heading2']))
    if os.path.exists(bar_chart_path):
        story.append(Image(bar_chart_path, width=480, hAlign='CENTER'))
    story.append(Spacer(1, 24))
    story.append(PageBreak())

    # --- Page 5: Sign Category Breakdown ---
    story.append(Paragraph("Sign Category Breakdown", styles['Heading1']))
    final_cat_table = Table(cat_table_data, colWidths=[300, 100, 100], hAlign='CENTER')
    final_cat_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(final_cat_table)
    story.append(Spacer(1, 12))
    if os.path.exists(sign_type_chart_path):
        story.append(Image(sign_type_chart_path, width=400, hAlign='CENTER'))
    story.append(PageBreak())

    # --- Page 6: Spatial Distribution and Cluster Analysis ---
    story.append(Paragraph("Spatial Distribution and Cluster Analysis", styles['Heading1']))
    spatial_text = "Below are the cluster analysis graphs for sign locations in each dataset."
    story.append(Paragraph(spatial_text, styles['BodyText']))
    story.append(Spacer(1, 12))
    row = []
    if cluster_chart_mach9 and os.path.exists(cluster_chart_mach9):
        cell_mach9 = [
            Paragraph("Mach9 Dataset", styles['Heading2']),
            preserve_aspect_image(cluster_chart_mach9, desired_width=200)
        ]
        row.append(cell_mach9)
    if cluster_chart_client and os.path.exists(cluster_chart_client):
        cell_client = [
            Paragraph("Client Dataset", styles['Heading2']),
            preserve_aspect_image(cluster_chart_client, desired_width=200)
        ]
        row.append(cell_client)
    if row:
        table = Table([row], colWidths=[250, 250])
        table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        story.append(table)
    story.append(PageBreak())

    # --- Page 9: Case Studies and Examples ---
    # Section A: Unique Signs in Mach9
    story.append(Paragraph(f"Unique Signs in Mach9 Dataset [{unmatched_mach9}]", styles['Heading1']))
    story.append(Spacer(1, 12))
    if examples_unique_mach9:
        for imgs, desc in examples_unique_mach9:
            image_table = Table([imgs], colWidths=[130, 130, 130], hAlign='CENTER')
            image_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('BOX', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(image_table)
            story.append(Spacer(1, 6))
            text_table = Table([[desc]], colWidths=[390], hAlign='CENTER')
            text_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('BOX', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(text_table)
            story.append(Spacer(1, 24))
    else:
        story.append(Paragraph("No unique Mach9 examples available.", styles['BodyText']))

    story.append(PageBreak())

    # --- Page 10: Unique Signs in Client, Matching Signs ---
    # Section B: Unique Signs in Client
    story.append(Paragraph(f"Unique Signs in Client Dataset [{unmatched_client}]", styles['Heading1']))
    story.append(Spacer(1, 12))
    if examples_unique_client:
        for imgs, desc in examples_unique_client:
            image_table = Table([imgs], colWidths=[130, 130, 130], hAlign='CENTER')
            image_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('BOX', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(image_table)
            story.append(Spacer(1, 6))
            text_table = Table([[desc]], colWidths=[390], hAlign='CENTER')
            text_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('BOX', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(text_table)
            story.append(Spacer(1, 24))
    else:
        story.append(Paragraph("No unique Client examples available.", styles['BodyText']))

    story.append(PageBreak())

    # Section C: Matching Signs
    story.append(Paragraph(f"Matching Signs [{matched_signs}]", styles['Heading1']))
    story.append(Spacer(1, 12))
    if examples_matched:
        for imgs, desc in examples_matched:
            image_table = Table([imgs], colWidths=[130, 130, 130], hAlign='CENTER')
            image_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('BOX', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(image_table)
            story.append(Spacer(1, 6))
            text_table = Table([[desc]], colWidths=[390], hAlign='CENTER')
            text_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
                ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('BOX', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(text_table)
            story.append(Spacer(1, 24))
    else:
        story.append(Paragraph("No matched sign examples available.", styles['BodyText']))

    story.append(PageBreak())

    # --- Page 11: Extended Methodology and Glossary (Appendix) ---
    story.append(Paragraph("Extended Methodology and Glossary (Appendix)", styles['Heading1']))
    story.append(Paragraph(appendix_text, styles['BodyText']))
    story.append(PageBreak())

    # Finally, build the PDF
    doc.build(story)
    print(f"Report generated successfully: {output_pdf}")


def main():
    mach9_file = os.path.join("..", "PTC", "PTC_Mach9_Signs_projected_lat_lon.csv")
    client_file = os.path.join("..", "PTC", "PTC Original Clipped.csv")
    output_pdf = os.path.join("..", "PTC", "mach9_comparative_report.pdf")
    logo_path = os.path.join("..", "Logos", "Mach9_Logo_Black 1.png")
    generate_report(mach9_file, client_file, output_pdf, logo_path)


if __name__ == "__main__":
    main()
