import pandas as pd
import os
import ast  # Import ast for safely evaluating string to list

# Load the CSVs
csv1_path = 'ra-dec_new.csv'  # CSV with image paths, RAAN, and DEC
csv2_path = 'angles_between_points.csv'  # CSV with image names and angles
csv3_path = "angles_between_polygons_new.csv"
# Read the CSVs into DataFrames
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv3_path)



# Ensure image names in df2 have the full path by adding the root directory to the image names
# Assuming the images in the first CSV are from the same root directory
root_directory = "content/processed-imagesp_new"
df2['image_path'] = df2['image_name'].apply(lambda x: os.path.join(root_directory, x))

df1['image_path'] = df1['Filename'].apply(lambda x: os.path.join(root_directory, x))


df2['angles'] = df2['angles'].apply(lambda x: ast.literal_eval(x))

df2['polygon_angles'] = df2['polygon_angles'].apply(lambda x: ast.literal_eval(x))
# Merge the DataFrames on the image path
print(df1.columns)
print(df2.columns)
merged_df = pd.merge(df1, df2, left_on='image_path', right_on='image_path')

# Save the merged DataFrame to a new CSV file
merged_csv_path = 'merged_data_new.csv'
merged_df.to_csv(merged_csv_path, index=False)
print(len(merged_df))
print(f"Merged CSV saved to {merged_csv_path}")
