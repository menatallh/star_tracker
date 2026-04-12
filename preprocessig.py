import pandas as pd
import os
import ast  # Import ast for safely evaluating string to list

# Load the CSVs
csv1_path = 'ra-dec.csv'  # CSV with image paths, RAAN, and DEC
csv2_path = 'angles_between_points.csv'  # CSV with image names and angles
csv3_path = 'angles_between_polygons.csv'
# Read the CSVs into DataFrames
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv3_path)
df2= pd.read_csv('angles_between_polygons.csv')

print(df1,df2)
# Ensure image names in df2 have the full path by adding the root directory to the image names
# Assuming the images in the first CSV are from the same root directory


root_directory = "/processed-imagesp2/"
df1['image_path'] = df1['image_path'].apply(lambda x: os.path.join(root_directory, x.split('/')[-1]))


df2['image_path'] = df2['image_name'].apply(lambda x: os.path.join(root_directory, x))
df2['angles'] = df2['angles'].apply(lambda x: ast.literal_eval(x))

df2['polygon_angles'] = df2['polygon_angles'].apply(lambda x: ast.literal_eval(x))
# Merge the DataFrames on the image path
merged_df = pd.merge(df1, df2, left_on='image_path', right_on='image_path')
print(merged_df)
# Save the merged DataFrame to a new CSV file
merged_csv_path = 'merged_data.csv'
merged_df.to_csv(merged_csv_path, index=False)

print(f"Merged CSV saved to {merged_csv_path}")
