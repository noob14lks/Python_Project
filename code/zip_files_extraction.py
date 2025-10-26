import os
import zipfile

source_folder = r"./data/"

destination_folder = os.path.join(source_folder, "extracteddata")
os.makedirs(destination_folder, exist_ok=True)

for filename in os.listdir(source_folder):
    if filename.endswith(".zip"):
        zip_path = os.path.join(source_folder, filename)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(destination_folder)
        
        print(f"Extracted: {filename}")

print(f"All zip files extracted into: {destination_folder}")
