import pandas as pd
import pyproj
import os

def convert_projection(input_file, output_folder=None, return_df=False):
    """
    Converts X, Y coordinates from EPSG:6599 (State Plane) to EPSG:4326 (WGS84)
    for a given CSV file, and optionally saves the transformed file to the specified output folder.

    **Functionality:**
    - Reads a CSV file containing X, Y coordinates.
    - Projects the coordinates to latitude/longitude (WGS84).
    - If `output_folder` is provided, saves the transformed data to a new CSV file in that folder.
    - If `return_df` is True, returns the transformed DataFrame (instead of or in addition to saving).

    **Inputs:**
    - input_file (str): Path to the CSV file with X, Y coordinates.
    - output_folder (str, optional): Folder where the converted CSV file will be saved. Defaults to None.
    - return_df (bool, optional): Whether to return the transformed DataFrame. Defaults to False.

    **Outputs:**
    - If `output_folder` is provided, a new CSV file named `{original_filename}_projected_lat_lon.csv`
      is saved in the output folder, with added `Longitude` and `Latitude` columns.
    - If `return_df` is True, returns the transformed DataFrame.
    """
    # Create a Transformer from EPSG:6599 to EPSG:4326 (WGS84).
    transformer = pyproj.Transformer.from_crs("EPSG:6599", "EPSG:4326", always_xy=True)

    # Read the CSV file.
    df = pd.read_csv(input_file)

    # Transform the X, Y columns into Longitude, Latitude.
    df["Longitude"], df["Latitude"] = transformer.transform(df["X"].values, df["Y"].values)

    # If an output folder is specified, save the transformed CSV.
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        base_name, _ = os.path.splitext(os.path.basename(input_file))
        output_file = os.path.join(output_folder, f"{base_name}_projected_lat_lon.csv")
        df.to_csv(output_file, index=False)
        print(f"File saved as: {output_file}")

    if return_df:
        return df


def process_input(input_path, output_folder):
    """
    Processes a single CSV file or all CSV files in a folder.
    If a folder is provided, all CSV files are combined into a single output CSV.

    **Inputs:**
    - input_path (str): Path to a CSV file or a folder containing CSV files.
    - output_folder (str): Folder where converted CSV file(s) will be saved.

    **Outputs:**
    - If `input_path` is a single CSV file, a single converted CSV is written to `output_folder`.
    - If `input_path` is a folder containing multiple CSV files:
        - Each file is converted in memory.
        - A "Layer" column is added (based on the filename).
        - All converted data is combined into one CSV file: `combined_projected_lat_lon.csv`.
    """
    if os.path.isfile(input_path):
        # Process a single CSV file -> same as original behavior
        convert_projection(input_path, output_folder=output_folder, return_df=False)

    elif os.path.isdir(input_path):
        # Process all CSV files in the folder, combine them into a single DataFrame
        csv_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(".csv")
        ]

        if not csv_files:
            print("No CSV files found in the directory.")
            return

        combined_dfs = []
        for file in csv_files:
            df_transformed = convert_projection(file, return_df=True)
            if df_transformed is not None:
                # Add a "Layer" column to differentiate each file in your viewer
                layer_name = os.path.splitext(os.path.basename(file))[0]
                df_transformed["Layer"] = layer_name
                combined_dfs.append(df_transformed)

        if not combined_dfs:
            print("No data to combine.")
            return

        # Combine all converted DataFrames into one
        final_df = pd.concat(combined_dfs, ignore_index=True)

        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Save the combined CSV
        combined_file = os.path.join(output_folder, "combined_projected_lat_lon.csv")
        final_df.to_csv(combined_file, index=False)
        print(f"Combined CSV saved as: {combined_file}")

    else:
        print("Invalid input path. Please provide a valid CSV file or directory containing CSV files.")


if __name__ == "__main__":
    # Example usage:
    #  - If 'input_path' is a single CSV file, one converted CSV is created.
    #  - If 'input_path' is a folder of CSV files, one combined CSV is created with all features.
    input_path = "./WDOT/TESTING"  # CSV file or folder containing CSV files
    output_folder = "./WDOT/BULK_TEST"  # Where converted CSV(s) will be saved

    process_input(input_path, output_folder)
