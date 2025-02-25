import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)
from duplicates import remove_duplicate_signs
from distance_matching import run_conflation_for_threshold
from sklearn.cluster import KMeans

# -------------------------------
# Configure your Google API Key here
GOOGLE_API_KEY = "AIzaSyAPxlcLCy9eZBMFqr5IVPpiCvWBTkqsw6M"
# -------------------------------

def fetch_street_view_image(lat, lon, output_path, heading=0, pitch=0, fov=90, size="400x400"):
    """
    Fetches a Google Street View image for the specified lat/lon and saves it to output_path.
    - heading, pitch, and fov can be adjusted.
    - size is "WIDTHxHEIGHT" in pixels.
    Returns True if successful, False otherwise.
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
            print(f"Street View API error: status={resp.status_code}, msg={resp.text}")
            return False
    except Exception as e:
        print(f"Street View request failed: {e}")
        return False

def fetch_static_map_image(lat, lon, output_path, zoom=16, size="400x400", maptype="roadmap"):
    """
    Fetches a Google Static Map image for the specified lat/lon and saves it to output_path.
    - zoom determines zoom level (1=world, 20=building).
    - size is "WIDTHxHEIGHT" in pixels.
    - maptype can be 'roadmap', 'satellite', 'terrain', or 'hybrid'.
    Returns True if successful, False otherwise.
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
            print(f"Static Map API error: status={resp.status_code}, msg={resp.text}")
            return False
    except Exception as e:
        print(f"Static Map request failed: {e}")
        return False

def generate_report_with_google_images(df_mach9, output_pdf, google_images_folder="street_view"):
    """
    Example function that:
      - Iterates over Mach9 data,
      - For each sign, fetches a Street View & Map image from Google,
      - Saves them locally,
      - Then builds a small PDF showing those images for demonstration.
    """

    # Ensure output folder exists
    os.makedirs(google_images_folder, exist_ok=True)
    map_folder = os.path.join(google_images_folder, "maps")
    os.makedirs(map_folder, exist_ok=True)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    story = []

    story.append(Paragraph("Google Images Demo", styles['Title']))
    story.append(Spacer(1, 24))

    # Let's limit ourselves to 5 examples
    df_subset = df_mach9.head(5)  # or use .sample(5, random_state=42) if you prefer random
    for i, row in df_subset.iterrows():
        lat = row.get("Latitude", None)
        lon = row.get("Longitude", None)
        code = row.get("MUTCD", f"Sign{i}")

        # Skip if lat/lon missing
        if pd.isna(lat) or pd.isna(lon):
            continue

        # Build file paths
        street_img_path = os.path.join(google_images_folder, f"{code}_street.jpg")
        map_img_path = os.path.join(map_folder, f"{code}_map.jpg")

        # Attempt to fetch images
        fetch_street_view_image(lat, lon, street_img_path, heading=0, pitch=0, fov=90, size="400x400")
        fetch_static_map_image(lat, lon, map_img_path, zoom=16, size="400x400")

        # Now add them to the PDF story
        story.append(Paragraph(f"<b>Sign MUTCD:</b> {code}", styles['Heading2']))
        story.append(Spacer(1, 6))

        images_row = []
        if os.path.exists(street_img_path):
            images_row.append(Image(street_img_path, width=200, hAlign='CENTER'))
        else:
            images_row.append(Paragraph("No Street View image", styles['BodyText']))

        if os.path.exists(map_img_path):
            images_row.append(Image(map_img_path, width=200, hAlign='CENTER'))
        else:
            images_row.append(Paragraph("No Map image", styles['BodyText']))

        # Put them side by side in a table
        from reportlab.platypus import Table
        table = Table([images_row], colWidths=[250, 250])
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))
        story.append(table)
        story.append(Spacer(1, 24))

    doc.build(story)
    print(f"Demo PDF with Google images created at: {output_pdf}")


def main():
    # Example usage:
    # 1. Load your Mach9 CSV
    # 2. Generate a small PDF that demonstrates Street View + Map images
    mach9_file = "PTC_Mach9_Signs_projected_lat_lon.csv"  # adjust path
    df_mach9 = pd.read_csv(mach9_file)

    # We'll just create a mini demo PDF
    demo_pdf = "mach9_google_images_demo.pdf"
    generate_report_with_google_images(df_mach9, demo_pdf)

if __name__ == "__main__":
    main()
