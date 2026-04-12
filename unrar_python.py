import rarfile
import os

rar_path = "star-images.rar"
extract_to = "extracted_files"

os.makedirs(extract_to, exist_ok=True)

with rarfile.RarFile(rar_path) as rf:
    rf.extractall(extract_to)

print("RAR extracted successfully.")
