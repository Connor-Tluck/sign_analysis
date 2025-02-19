# Road Sign Processing and Matching

This repository provides tools for processing, projecting, and matching road sign data using Python. It includes functions for:
- Converting coordinate projections (State Plane to WGS84), This is necessary to project perception data to WGS84 for viewing in web maps.  
- Matching road signs using spatial proximity for analysis between processing runs. 
- Scraping road sign images from Wikipedia

## Installation

Ensure you have Python installed and install the required dependencies using:

```bash
pip install pandas pyproj numpy tqdm scikit-learn geopandas shapely requests beautifulsoup4
```

## Features

### 1. Convert Projection (State Plane to WGS84)
This function converts X, Y coordinates from **EPSG:6599 (State Plane)** to **EPSG:4326 (WGS84)**.

#### **Usage**
```python
convert_projection(input_file, output_folder)
```

#### **Inputs**
- `input_file` (str): Path to the CSV file containing X, Y coordinates.
- `output_folder` (str): Folder where the converted CSV file will be saved.

#### **Outputs**
- A new CSV file in `output_folder` named `{original_filename}_projected_lat_lon.csv`, containing `Longitude` and `Latitude` columns.

### 2. Process Multiple CSV Files
Processes multiple CSV files in a directory and outputs a **combined** CSV file with all projected coordinates.

#### **Usage**
```python
process_input(input_path, output_folder)
```

#### **Inputs**
- `input_path` (str): Path to a CSV file or a folder containing multiple CSV files.
- `output_folder` (str): Folder where converted files will be saved.

#### **Outputs**
- If `input_path` is a single file, the converted CSV is saved.
- If `input_path` is a folder, all files are projected and merged into **`combined_projected_lat_lon.csv`**.

### 3. Match Road Signs Using BallTree
This function matches road signs from two datasets based on **spatial proximity** using BallTree and the Haversine metric.

#### **Usage**
```python
match_signs_with_balltree()
```

#### **Inputs**
- Two CSV files containing road sign data with `Latitude` and `Longitude` columns.
- A distance threshold (e.g., **12 meters**) for matching.

#### **Outputs**
- **Matched Signs**:
  - `comparison_data/matched.csv`
  - `comparison_data/matched.shp` (Shapefile)
- **Unmatched Signs**:
  - `comparison_data/unmatched.csv`
  - `comparison_data/unmatched.shp` (Shapefile)

#### **Summary Metrics**
- Total signs in both datasets
- Number of matched/unmatched signs
- File output locations

### 4. Scrape US Road Signs from Wikipedia
Fetches road sign images from Wikipedia and saves them locally.

#### **Usage**
```python
scrape_us_road_signs()
```

#### **Outputs**
- A folder **`sign_images`** containing road sign images in `.png`, `.jpg`, or `.svg` format.

## Running the Scripts

To run the scripts, execute:

```bash
python your_script.py
```

or run individual functions inside a Python session.

## Example Usage

### **Convert Projections**
```python
process_input("./WDOT/TESTING", "./WDOT/BULK_TEST")
```

### **Match Signs**
```python
match_signs_with_balltree()
```

### **Scrape Road Signs**
```python
scrape_us_road_signs()
```

## Requirements

- **Python 3.7+**
- **Libraries**:
  - `pandas`
  - `pyproj`
  - `numpy`
  - `tqdm`
  - `scikit-learn`
  - `geopandas`
  - `shapely`
  - `requests`
  - `beautifulsoup4`

## License

This project is licensed under the **MIT License**.

---
*Developed for geospatial data processing and sign matching.*
