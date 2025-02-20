import pandas as pd
import pyproj
import os


def convert_projection(input_file, output_folder):
    """
    Converts X, Y coordinates from EPSG:6599 (State Plane) to EPSG:4326 (WGS84)
    for a given CSV file, and saves the transformed file to the specified output folder.

    **Functionality:**
    - Reads a CSV file containing X, Y coordinates.
    - Projects the coordinates to latitude/longitude (WGS84).
    - Saves the transformed data to a new CSV file in the output folder.

    **Input:**
    - input_file (str): Path to the CSV file with X, Y coordinates.
    - output_folder (str): Folder where the converted CSV file will be saved.

    **Output:**
    - A new CSV file saved in output_folder named `{original_filename}_projected_lat_lon.csv`,
      with added `Longitude` and `Latitude` columns.
    """
    # Ensure the output folder exists.
    os.makedirs(output_folder, exist_ok=True)

    # Generate output filename dynamically.
    base_name, ext = os.path.splitext(os.path.basename(input_file))
    output_file = os.path.join(output_folder, f"{base_name}_projected_lat_lon.csv")

    # Read the CSV file.
    df = pd.read_csv(input_file)

    # Create a Transformer from EPSG:6599 (State Plane) to EPSG:4326 (WGS84).
    transformer = pyproj.Transformer.from_crs("EPSG:32615", "EPSG:4326", always_xy=True)

    # Transform the X, Y columns into Longitude, Latitude.
    df["Longitude"], df["Latitude"] = transformer.transform(df["X"].values, df["Y"].values)

    # Save the transformed data to a new CSV file.
    df.to_csv(output_file, index=False)

    # Print confirmation message.
    print(f"File saved as: {output_file}")


def process_input(input_path, output_folder):
    """
    Processes a single CSV file or all CSV files in a folder.

    **Input:**
    - input_path (str): Path to a CSV file or a folder containing CSV files.
    - output_folder (str): Folder where converted CSV files will be saved.

    **Output:**
    - Converted CSV file(s) saved to output_folder.
    """
    if os.path.isdir(input_path):
        # Process all CSV files in the folder.
        csv_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(".csv")]
        if not csv_files:
            print("No CSV files found in the directory.")
            return
        for file in csv_files:
            convert_projection(file, output_folder)
    elif os.path.isfile(input_path):
        # Process a single CSV file.
        convert_projection(input_path, output_folder)
    else:
        print("Invalid input path. Please provide a valid CSV file or directory containing CSV files.")


if __name__ == "__main__":
    # Define input path (can be a single CSV file or a folder containing CSV files).
    input_path = "./TxDot/Signs.csv"  # Update this to your CSV file or folder path.

    # Define the output folder where converted CSVs will be saved.
    output_folder = "./TxDot"  # Update if needed.

    process_input(input_path, output_folder)
