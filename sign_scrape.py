import requests  # Used for making HTTP requests to fetch web content
from bs4 import BeautifulSoup  # Used for parsing HTML content
import os  # Provides functions to interact with the file system
import time  # Used to add delays between requests


def scrape_us_road_signs():
    """
    Scrapes road sign images from Wikipedia's 'Road signs in the United States' page.

    **Functionality:**
    - Fetches the webpage containing US road signs.
    - Extracts sign codes and corresponding images from the page.
    - Downloads and saves the images in a local folder.

    **Input:**
    - No direct input parameters.
    - The function automatically retrieves data from a predefined Wikipedia URL.

    **Output:**
    - A folder named `sign_images` containing downloaded road sign images (.png, .jpg, or .svg).
    - Log messages indicating download progress or failures.
    """

    url = "https://en.wikipedia.org/wiki/Road_signs_in_the_United_States"

    # Create or use the existing folder for images
    folder = "sign_images"
    os.makedirs(folder, exist_ok=True)

    # Optional: use a custom User-Agent header
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; RoadSignScraper/1.0)"
    }

    # Make the request to the page
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # ensure we got a valid response

    soup = BeautifulSoup(response.text, "html.parser")

    # Find all gallery boxes
    gallery_boxes = soup.find_all("li", class_="gallerybox")

    for box in gallery_boxes:
        # Each gallerybox should have a 'gallerytext' div
        text_div = box.find("div", class_="gallerytext")
        if not text_div:
            continue

        # Split the gallerytext by <br> or line breaks
        lines = text_div.get_text("\n").split("\n")

        # The first line should be the sign code (possibly in quotes)
        raw_code = lines[0].strip()
        sign_code = raw_code.strip('"').strip("'")

        # Find the <img> to get the thumbnail src
        img_tag = box.find("img")
        if not img_tag:
            continue

        img_src = img_tag.get("src", "")
        # Wikipedia often uses protocol-relative URLs (starting with //)
        if img_src.startswith("//"):
            img_src = "https:" + img_src

        if not img_src:
            continue

        print(f"Downloading image for {sign_code} from {img_src} ...")
        try:
            # Request the image data with headers
            img_response = requests.get(img_src, headers=headers)
            img_response.raise_for_status()
            img_data = img_response.content

            # Determine the file extension based on the Content-Type header.
            content_type = img_response.headers.get("Content-Type", "").lower()
            if "png" in content_type:
                ext = ".png"
            elif "svg" in content_type:
                ext = ".svg"
            elif "jpeg" in content_type or "jpg" in content_type:
                ext = ".jpg"
            else:
                # Fallback: try to extract from URL; default to .jpg if unknown
                _, ext = os.path.splitext(img_src)
                if not ext:
                    ext = ".jpg"

            # Build the full path inside the sign_images folder
            filename = os.path.join(folder, sign_code + ext)

            with open(filename, "wb") as f:
                f.write(img_data)
            print(f"Saved {filename}")

        except requests.exceptions.RequestException as e:
            print(f"Failed to download {sign_code}: {e}")

        # Add a delay between requests to be polite
        time.sleep(1)

    print("Done!")


if __name__ == "__main__":
    scrape_us_road_signs()
