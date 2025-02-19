import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Specify the input CSV file path and the output shapefile path
csv_file = "WDOT_Noise_Reduction_Test_Sign_Face_projected_lat_lon.csv"  # Replace with your CSV file path
output_shapefile = "WDOT_Noise_Reduction_Test_Sign_Face_projected_lat_lon.shp"  # Replace with desired output path

# Read the CSV file
df = pd.read_csv(csv_file)

# Create a geometry column using the Longitude and Latitude columns.
# Note: The CSV header should contain 'Longitude' and 'Latitude'
df['geometry'] = df.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)

# Convert the DataFrame to a GeoDataFrame.
# Here, we assume the coordinates are in WGS84 (EPSG:4326).
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

# Write the GeoDataFrame to a shapefile.
gdf.to_file(output_shapefile, driver="ESRI Shapefile")

print("Shapefile created successfully:", output_shapefile)
