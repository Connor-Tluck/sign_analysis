import os
import pandas as pd
from duplicates import remove_duplicate_signs
from distance_matching import run_conflation_for_threshold
from report_generation import generate_conflation_report_all
from direct_results import generate_direct_results

# Define schema mapping for each dataset
# Mach9 schema (assumed already correct)
mach9_schema = {"lat": "Latitude", "lon": "Longitude", "mutcd": "MUTCD"}
# Client schema – these columns will be renamed to match Mach9
client_schema = {"lat": "CC_LATITUDE", "lon": "CC_LONGITUDE", "mutcd": "MUTCD"}

REMOVE_DUPLICATES = True
BUFFER_LIST = [5, 10, 12]
# Flags to choose which reports to generate
REPORT_CONFLATION = False  # not used in this example
REPORT_DIRECT_RESULTS = True


def main():
    # Read input CSV files (adjust path as needed)
    mach9_file = os.path.join("..", "PTC", "PTC_Mach9_Signs_projected_lat_lon.csv")
    client_file = os.path.join("..", "PTC", "PTC Original Clipped.csv")
    df_mach9 = pd.read_csv(mach9_file, low_memory=False)
    df_client = pd.read_csv(client_file, low_memory=False)

    # Rename client columns to match Mach9 schema
    df_client.rename(columns={
        client_schema["lat"]: "Latitude",
        client_schema["lon"]: "Longitude",
        client_schema["mutcd"]: "MUTCD"
    }, inplace=True)

    # (Optional) Remove duplicates if desired
    duplicates_info = {}
    if REMOVE_DUPLICATES:
        orig_count_f1 = len(df_mach9)
        df_mach9, dup_removed_f1 = remove_duplicate_signs(df_mach9, mach9_schema["lat"], mach9_schema["lon"],
                                                          decimals=5)
        unique_count_f1 = len(df_mach9)
        orig_count_f2 = len(df_client)
        df_client, dup_removed_f2 = remove_duplicate_signs(df_client, "Latitude", "Longitude", decimals=5)
        unique_count_f2 = len(df_client)
        duplicates_info["File 1"] = {"original_count": orig_count_f1, "duplicates_removed": dup_removed_f1,
                                     "unique_count": unique_count_f1}
        duplicates_info["File 2"] = {"original_count": orig_count_f2, "duplicates_removed": dup_removed_f2,
                                     "unique_count": unique_count_f2}

    # (Optional) Run conflation per buffer (if needed for other reports)
    base_output_folder = os.path.join("..", "PTC", "conflation_results")
    os.makedirs(base_output_folder, exist_ok=True)
    results = []
    for buf in BUFFER_LIST:
        result = run_conflation_for_threshold(df_mach9, df_client, buf,
                                              mach9_schema["lat"], mach9_schema["lon"],
                                              "Latitude", "Longitude",
                                              base_output_folder)
        results.append(result)

    logo_path = os.path.join("..", "Logos", "Mach9_Logo_Black 1.png")

    # Generate Direct Results Report (which compares Mach9 vs. Client directly)
    pdf_path_direct = os.path.join(base_output_folder, "mach9_direct_results.pdf")
    if REPORT_DIRECT_RESULTS:
        generate_direct_results(df_mach9, df_client, logo_path=logo_path,
                                output_pdf=pdf_path_direct, duplicates_info=duplicates_info)


if __name__ == "__main__":
    main()
