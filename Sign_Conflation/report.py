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
from distance_matching import run_conflation_for_threshold
from sklearn.cluster import KMeans

# ---------------------
# Global settings and schema mappings
# ---------------------
GOOGLE_API_KEY = "AIzaSyAPxlcLCy9eZBMFqr5IVPpiCvWBTkqsw6M"  # Replace with your actual API key

mach9_schema = {"lat": "Latitude", "lon": "Longitude", "mutcd": "MUTCD"}
client_schema = {"lat": "CC_LATITUDE", "lon": "CC_LONGITUDE", "mutcd": "MUTCD"}

REMOVE_DUPLICATES = True
BUFFER_LIST = [5, 10, 12]

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

from PIL import Image as PILImage
from reportlab.platypus import Image

def preserve_aspect_image(path, desired_width, hAlign='CENTER'):
    """
    Opens an image from `path`, calculates the appropriate height to
    maintain aspect ratio for the given `desired_width`, and returns
    a ReportLab Image object with that size.
    """
    if not os.path.exists(path):
        return Paragraph("Image not found", getSampleStyleSheet()['BodyText'])

    pil_img = PILImage.open(path)
    w, h = pil_img.size
    aspect_ratio = h / float(w)

    # Calculate new height to maintain the original aspect ratio
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
    """
    Fetches a Google Street View image for the specified latitude and longitude.
    Saves the image to output_path.
    """
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
    """
    Fetches metadata from Google Street View for the specified latitude and longitude.
    Returns metadata as a JSON dict, including the capture date if available.
    """
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
    """
    Fetches a Google Static Map image for the specified latitude and longitude.
    Saves the image to output_path.
    """
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
# Image helper function to fix vertical skew
# ---------------------
def create_fixed_image(image_path, fixed_width):
    """
    Opens an image from image_path, calculates the appropriate height to
    maintain aspect ratio for the given fixed_width, and returns a ReportLab Image.
    """
    if not os.path.exists(image_path):
        return None
    pil_img = PILImage.open(image_path)
    w, h = pil_img.size
    aspect = h / float(w)
    fixed_height = fixed_width * aspect
    return Image(image_path, width=fixed_width, height=fixed_height, hAlign='CENTER')


def create_logo_image(logo_path, max_width=200, max_height=80):
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
# Dummy Chart Generator (for charts not yet fully implemented)
# ---------------------
def generate_dummy_chart(chart_type, filename):
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
    """
    Generates a bar chart showing the actual counts per sign category for Mach9.
    """
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


# ---------------------
# Report Generation Function
# ---------------------
def generate_report(mach9_csv, client_csv, output_pdf, logo_path):
    # Load data
    df_mach9 = pd.read_csv(mach9_csv, low_memory=False)
    df_client = pd.read_csv(client_csv, low_memory=False)

    styles = getSampleStyleSheet()

    # Standardize client columns
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

    # Remove duplicates if enabled
    duplicates_info = {}
    if REMOVE_DUPLICATES:
        orig_count_mach9 = len(df_mach9)
        df_mach9, dup_removed_mach9 = remove_duplicate_signs(df_mach9, "Latitude", "Longitude", decimals=5)
        unique_count_mach9 = len(df_mach9)
        orig_count_client = len(df_client)
        df_client, dup_removed_client = remove_duplicate_signs(df_client, "Latitude", "Longitude", decimals=5)
        unique_count_client = len(df_client)
        duplicates_info["Mach9"] = {"original_count": orig_count_mach9, "duplicates_removed": dup_removed_mach9,
                                      "unique_count": unique_count_mach9}
        duplicates_info["Client"] = {"original_count": orig_count_client, "duplicates_removed": dup_removed_client,
                                       "unique_count": unique_count_client}

    # Compute total counts
    total_mach9 = len(df_mach9)
    total_client = len(df_client)

    # Compute location sets (using 5-decimal precision)
    set_mach9_locations = set()
    for _, row in df_mach9.iterrows():
        try:
            lat = round(float(row["Latitude"]), 5)
            lon = round(float(row["Longitude"]), 5)
            set_mach9_locations.add((lat, lon))
        except:
            continue
    set_client_locations = set()
    for _, row in df_client.iterrows():
        try:
            lat = round(float(row["Latitude"]), 5)
            lon = round(float(row["Longitude"]), 5)
            set_client_locations.add((lat, lon))
        except:
            continue

    missing_in_client_location = len(set_mach9_locations - set_client_locations)
    missing_in_mach9_location = len(set_client_locations - set_mach9_locations)
    perc_missing_in_client_location = (missing_in_client_location / total_mach9 * 100) if total_mach9 else 0
    perc_missing_in_mach9_location = (missing_in_mach9_location / total_client * 100) if total_client else 0

    # Categorize signs
    mach9_cat_counts = categorize_signs(df_mach9, mutcd_col="MUTCD")
    client_cat_counts = categorize_signs(df_client, mutcd_col="MUTCD")

    # Define Executive Summary + Approach text (updated with new metrics)
    summary_text = f"""
<b>Executive Summary</b><br/><br/>
This report compares Mach9's LiDAR-extracted sign features with a legacy client dataset.
Key findings include:<br/>
&nbsp;&nbsp;&bull;&nbsp; Total Mach9 Signs: {total_mach9}<br/>
&nbsp;&nbsp;&bull;&nbsp; Total Client Signs: {total_client}<br/>
&nbsp;&nbsp;&bull;&nbsp; Signs in Mach9 not in Client (by location): {missing_in_client_location} ({perc_missing_in_client_location:.2f}%)<br/>
&nbsp;&nbsp;&bull;&nbsp; Signs in Client not in Mach9 (by location): {missing_in_mach9_location} ({perc_missing_in_mach9_location:.2f}%)<br/>
&nbsp;&nbsp;&bull;&nbsp; Duplicates Removed - Mach9: {duplicates_info["Mach9"]["duplicates_removed"]}, Client: {duplicates_info["Client"]["duplicates_removed"]}<br/><br/>
<b>Approach and Methodology</b><br/>
1. Standardize column names for latitude, longitude, and MUTCD.<br/>
2. Fill missing MUTCD codes with 'Unknown' to avoid classification errors.<br/>
3. Remove duplicates using coordinate precision.<br/>
4. Compute key metrics based on location and MUTCD code comparisons.<br/>
5. Generate charts to illustrate distribution, clustering, and discrepancies.
"""

    # Key Metrics Table (slimmer with revised rows)
    data_metrics = [
        ['Metric', 'Mach9 Dataset', 'Client Dataset', 'Percentage'],
        ['Total Signs', total_mach9, total_client,
         f"{abs(total_mach9 - total_client) / max(total_mach9, total_client) * 100:.2f}%" if max(total_mach9, total_client) else "N/A"],
        ['Missing in Client (by location)', missing_in_client_location, '-', f"{perc_missing_in_client_location:.2f}%"],
        ['Missing in Mach9 (by location)', '-', missing_in_mach9_location, f"{perc_missing_in_mach9_location:.2f}%"],
        ['Duplicates Removed', duplicates_info["Mach9"]["duplicates_removed"], duplicates_info["Client"]["duplicates_removed"], '-']
    ]

    # Generate Total Sign Count Bar Chart with labels
    bar_chart_path = 'bar_chart_total_counts.png'
    plt.figure(figsize=(6, 4))
    bars = plt.bar(['Mach9', 'Client'], [total_mach9, total_client], color=['#1f77b4', '#ff7f0e'])
    plt.title('Total Signs Count Comparison', fontsize=14)
    plt.ylabel('Number of Signs', fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, int(yval), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(bar_chart_path, bbox_inches='tight')
    plt.close()

    # Generate other charts (using dummy generators where applicable)
    venn_chart_path = 'venn_diagram.png'
    generate_dummy_chart('venn', venn_chart_path)
    box_plot_path = 'box_plot_confidence.png'
    generate_dummy_chart('box', box_plot_path)
    # Instead of a dummy stacked bar, generate an actual sign type distribution chart (Mach9 only)
    sign_type_chart_path = generate_sign_type_distribution_chart(mach9_cat_counts)
    heatmap_path = 'heatmap_sign_density.png'
    generate_dummy_chart('heatmap', heatmap_path)
    line_chart_path = 'line_chart_temporal.png'
    generate_dummy_chart('line', line_chart_path)

    # Detailed Error Analysis Chart (using location-based metrics)
    error_chart_path = "error_analysis.png"
    plt.figure(figsize=(6, 4))
    error_labels = ["Missing in Client (by location)", "Missing in Mach9 (by location)"]
    error_values = [missing_in_client_location, missing_in_mach9_location]
    plt.bar(error_labels, error_values, color=["purple", "red"])
    plt.title("Error Analysis: Missing in Client vs Missing in Mach9", fontsize=14)
    for i, v in enumerate(error_values):
        plt.text(i, v + 0.5, str(v), ha="center")
    plt.tight_layout()
    plt.savefig(error_chart_path, bbox_inches='tight')
    plt.close()

    # Spatial Distribution and Cluster Analysis - generate charts for each dataset
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
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=cluster_labels, cmap="plasma", s=10)
        plt.title("Cluster Analysis of Client Sign Locations", fontsize=14)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.colorbar(scatter, label="Cluster")
        plt.tight_layout()
        plt.savefig(cluster_chart_client, bbox_inches='tight')
        plt.close()
    else:
        cluster_chart_client = None

    # Sign Dimensions and Condition Analysis (Page 6)
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

    # Quality Metrics and Performance Evaluation (Page 7)
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

    # Sign Category Breakdown Table (Page 4)
    cat_table_data = [['Category', 'Mach9 Count', 'Client Count']]
    for c in sorted(set(mach9_cat_counts.keys()).union(client_cat_counts.keys())):
        mach9_count = mach9_cat_counts.get(c, 0)
        client_count = client_cat_counts.get(c, 0)
        cat_table_data.append([c, str(mach9_count), str(client_count)])

    # ---------------------
    # Annotated Case Studies / Sample Comparisons (Page 8)
    # ---------------------
    # Create folders for images if they don't exist
    street_folder = "street_view"
    map_folder = "map_view"
    os.makedirs(street_folder, exist_ok=True)
    os.makedirs(map_folder, exist_ok=True)

    # Determine samples from Mach9 missing Client and Client missing Mach9 (based on location)
    df_mach9_missing = df_mach9[df_mach9.apply(lambda row: (round(row["Latitude"], 5), round(row["Longitude"], 5)) not in set_client_locations, axis=1)]
    df_client_missing = df_client[df_client.apply(lambda row: (round(row["Latitude"], 5), round(row["Longitude"], 5)) not in set_mach9_locations, axis=1)]

    sample_mach9 = df_mach9_missing.sample(n=min(5, len(df_mach9_missing)), random_state=42) if not df_mach9_missing.empty else pd.DataFrame()
    sample_client = df_client_missing.sample(n=min(5, len(df_client_missing)), random_state=42) if not df_client_missing.empty else pd.DataFrame()

    annotated_df = pd.concat([sample_mach9.assign(Source="Mach9"), sample_client.assign(Source="Client")])
    # If fewer than 10 examples, add some from the intersection (present in both datasets)
    if len(annotated_df) < 10:
        df_both = df_mach9[df_mach9.apply(lambda row: (round(row["Latitude"], 5), round(row["Longitude"], 5)) in set_client_locations, axis=1)]
        extra_needed = 10 - len(annotated_df)
        if not df_both.empty:
            extra_samples = df_both.sample(n=min(extra_needed, len(df_both)), random_state=42)
            extra_samples = extra_samples.assign(Source="Both")
            annotated_df = pd.concat([annotated_df, extra_samples])
    annotated_df = annotated_df.sample(n=min(10, len(annotated_df)), random_state=42)

    annotated_examples = []
    for idx, row in annotated_df.iterrows():
        code = row.get('MUTCD', 'Unknown')
        text_info = row.get("Text", "No additional text")
        lat = row.get("Latitude", None)
        lon = row.get("Longitude", None)
        source = row.get("Source", "Unknown")
        # Build file paths for street and map images (include index to ensure uniqueness)
        street_img_path = os.path.join(street_folder, f"{code}_{idx}_street.jpg")
        map_img_path = os.path.join(map_folder, f"{code}_{idx}_map.jpg")
        # Fetch images via Google APIs if coordinates available
        if lat and lon and not pd.isna(lat) and not pd.isna(lon):
            fetch_street_view_image(lat, lon, street_img_path, heading=180, pitch=-10, fov=90, size="400x400")
            fetch_static_map_image(lat, lon, map_img_path, zoom=16, size="400x400")
            metadata = fetch_street_view_metadata(lat, lon, heading=180, pitch=-10, fov=90, size="400x400")
            capture_date = metadata.get("date", "Unknown") if metadata else "Unknown"
        else:
            capture_date = "Unknown"
        # Get MUTCD sign image from folder "mutcd_signs"
        mutcd_sign_path = os.path.join("../mutcd_signs", f"{code}.png")
        # Create fixed images; render MUTCD sign image at 25% size (fixed width=25)
        street_img = create_fixed_image(street_img_path, 120) if os.path.exists(street_img_path) else Paragraph(f"No Street View for <b>{code}</b>", styles['BodyText'])
        map_img = create_fixed_image(map_img_path, 120) if os.path.exists(map_img_path) else Paragraph(f"No Map View for <b>{code}</b>", styles['BodyText'])
        mutcd_img = create_fixed_image(mutcd_sign_path, 25) if os.path.exists(mutcd_sign_path) else Paragraph(f"No MUTCD Sign Image for <b>{code}</b>", styles['BodyText'])
        # Build Google Street View URL
        street_view_url = f"https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon}"
        # Build descriptive text with additional details
        description = (f"<b>MUTCD:</b> {code}<br/>"
                       f"<b>Text:</b> {text_info}<br/>"
                       f"<b>Source:</b> {source}<br/>"
                       f"<b>Lat/Lon:</b> {lat}, {lon}<br/>"
                       f"<b>Street View URL:</b> <a href='{street_view_url}'>{street_view_url}</a><br/>"
                       f"<b>Capture Date:</b> {capture_date}<br/>")
        if source == "Mach9":
            description += "Sign present in Mach9 dataset only (not found in Client dataset)."
        elif source == "Client":
            description += "Sign present in Client dataset only (not found in Mach9 dataset)."
        else:
            description += "Sign present in both datasets."
        annotated_examples.append(( [street_img, map_img, mutcd_img], Paragraph(description, styles['BodyText']) ))

    # Appendix text
    appendix_text = """
<b>Extended Methodology:</b><br/>
- Data standardization: Columns were aligned across datasets.<br/>
- Duplicate removal: Duplicates were removed based on geographic coordinates (5 decimal precision).<br/>
- Spatial and cluster analysis: Data was analyzed for clustering and coverage patterns.<br/>
- Quality metrics: Confidence scores and sign dimensions were evaluated.<br/><br/>
<b>Glossary:</b><br/>
- <i>Missing in Client (by location):</i> Signs present in Mach9 but absent in Client dataset based on location.<br/>
- <i>Missing in Mach9 (by location):</i> Signs present in Client but absent in Mach9 dataset based on location.<br/>
- <i>Duplicates Removed:</i> Duplicate sign entries removed during preprocessing.<br/>
- <i>MUTCD Codes:</i> Standardized sign classification codes used by Mach9.<br/>
- <i>Confidence Score:</i> A measure of detection certainty.<br/>
- <i>Cluster Analysis:</i> Grouping of similar sign locations.
"""

    # ---------------------
    # Build the PDF using ReportLab's Platypus
    # ---------------------
    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    story = []

    # Page 1: Cover Page
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

    # Page 2: Combined Executive Summary and Approach
    story.append(Paragraph(summary_text, styles['BodyText']))
    story.append(Spacer(1, 24))
    story.append(PageBreak())

    # Page 3: Key Metrics and Dataset Comparison (including Detailed Error Analysis)
    story.append(Paragraph("Key Metrics and Dataset Comparison", styles['Heading1']))
    table_metrics = Table(data_metrics, hAlign='CENTER')
    table_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table_metrics)
    story.append(Spacer(1, 12))
    if os.path.exists(bar_chart_path):
        story.append(Image(bar_chart_path, width=400, hAlign='CENTER'))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Detailed Error Analysis", styles['Heading2']))
    story.append(Paragraph("The graph below shows errors: missing in Client vs missing in Mach9 based on location.", styles['BodyText']))
    if os.path.exists(error_chart_path):
        story.append(Image(error_chart_path, width=400, hAlign='CENTER'))
    story.append(PageBreak())

    # Page 4: Sign Category Breakdown
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

    # Page 5: Spatial Distribution and Cluster Analysis (both datasets on one page)
    story.append(Paragraph("Spatial Distribution and Cluster Analysis", styles['Heading1']))
    spatial_text = "Below are the cluster analysis graphs for sign locations in each dataset."
    story.append(Paragraph(spatial_text, styles['BodyText']))
    story.append(Spacer(1, 12))
    row = []
    if cluster_chart_mach9 and os.path.exists(cluster_chart_mach9):
        if cluster_chart_mach9 and os.path.exists(cluster_chart_mach9):
            cell = [
                Paragraph("Mach9 Dataset", styles['Heading2']),
                preserve_aspect_image(cluster_chart_mach9, desired_width=200)
            ]
            row.append(cell)
    if cluster_chart_client and os.path.exists(cluster_chart_client):
        if cluster_chart_mach9 and os.path.exists(cluster_chart_mach9):
            cell = [
                Paragraph("Client Dataset", styles['Heading2']),
                preserve_aspect_image(cluster_chart_client, desired_width=200)
            ]
            row.append(cell)
    if row:
        table = Table([row], colWidths=[250, 250])
        table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
        story.append(table)
    story.append(PageBreak())

    # Page 6: Sign Dimensions and Condition Analysis
    story.append(Paragraph("Sign Dimensions and Condition Analysis", styles['Heading1']))
    dim_text = "Box plots below illustrate the distribution of sign widths and heights as captured in Mach9."
    story.append(Paragraph(dim_text, styles['BodyText']))
    story.append(Spacer(1, 12))
    if dimension_chart_path and os.path.exists(dimension_chart_path):
        story.append(Image(dimension_chart_path, width=400, hAlign='CENTER'))
    story.append(PageBreak())

    # Page 7: Quality Metrics and Performance Evaluation
    story.append(Paragraph("Quality Metrics and Performance Evaluation", styles['Heading1']))
    if quality_chart_path and os.path.exists(quality_chart_path):
        story.append(Paragraph("Confidence Score Distribution (Mach9).", styles['BodyText']))
        story.append(Image(quality_chart_path, width=400, hAlign='CENTER'))
        story.append(Spacer(1, 12))
        story.append(Paragraph(quality_metrics_text, styles['BodyText']))
    else:
        story.append(Paragraph(quality_metrics_text, styles['BodyText']))
    story.append(PageBreak())

    # Page 8: Annotated Case Studies / Sample Comparisons
    # Page 8: Annotated Case Studies / Sample Comparisons
    story.append(Paragraph("Annotated Case Studies / Sample Comparisons", styles['Heading1']))
    story.append(Spacer(1, 12))

    if annotated_examples:
        for imgs, desc in annotated_examples:
            # Build a 3-column table for images with padding and a black border
            # Reduce the column widths so the images do not overflow
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

            # Wrap the descriptive text in its own table with a black border
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
        story.append(Paragraph("No annotated examples available.", styles['BodyText']))

    story.append(PageBreak())

    # Page 9: Extended Methodology and Glossary (Appendix)
    story.append(Paragraph("Extended Methodology and Glossary (Appendix)", styles['Heading1']))
    story.append(Paragraph(appendix_text, styles['BodyText']))
    story.append(PageBreak())

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
