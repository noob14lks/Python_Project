import os
import zipfile
import hashlib

SOURCE_FOLDER = r"./data/"
DESTINATION_FOLDER = os.path.join(SOURCE_FOLDER, "extracteddata")
MAX_NAME_LEN = 50
FLATTEN_PATHS = False 

def shorten_name(name, max_len=MAX_NAME_LEN):
    """Shorten overly long names safely with an MD5 hash suffix."""
    if len(name) <= max_len:
        return name
    hash_suffix = hashlib.md5(name.encode()).hexdigest()[:8]
    return name[:max_len - 9] + "_" + hash_suffix

def win_long_path(path):
    """Return a Windows-compatible long path (\\?\\ prefix)."""
    path = os.path.abspath(path)
    if not path.startswith("\\\\?\\"):
        path = "\\\\?\\" + path
    return path

def safe_extract(zip_ref, extract_path):
    """Safely extract files from a zip into the given path."""
    for member in zip_ref.infolist():
        if member.is_dir():
            continue

        orig_path = member.filename
        if FLATTEN_PATHS:
            parts = [os.path.basename(orig_path)]
        else:
            parts = [shorten_name(p) for p in orig_path.split('/') if p and p != "."]

        safe_path = os.path.join(extract_path, *parts)

        if not os.path.abspath(safe_path).startswith(os.path.abspath(extract_path)):
            print(f" Skipping unsafe path: {safe_path}")
            continue

        safe_path = win_long_path(safe_path)
        os.makedirs(os.path.dirname(safe_path), exist_ok=True)

        with zip_ref.open(member) as source, open(safe_path, "wb") as target:
            target.write(source.read())

def extract_nested_zips(base_path):
    """Recursively extract any zip files inside base_path."""
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith('.zip'):
                nested_zip_path = os.path.join(root, file)
                nested_extract_path = root  

                try:
                    with zipfile.ZipFile(nested_zip_path, 'r') as nested_zip:
                        safe_extract(nested_zip, nested_extract_path)
                    print(f"Extracted nested ZIP: {nested_zip_path}")

                    extract_nested_zips(nested_extract_path)

                    os.remove(nested_zip_path)
                except zipfile.BadZipFile:
                    print(f"Skipping bad ZIP file: {nested_zip_path}")

def main():
    os.makedirs(win_long_path(DESTINATION_FOLDER), exist_ok=True)

    for filename in os.listdir(SOURCE_FOLDER):
        if filename.lower().endswith('.zip'):
            zip_path = os.path.join(SOURCE_FOLDER, filename)
            dataset_name = shorten_name(os.path.splitext(filename)[0])
            extract_path = os.path.join(DESTINATION_FOLDER, dataset_name)
            os.makedirs(win_long_path(extract_path), exist_ok=True)

            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    safe_extract(zip_ref, extract_path)
                extract_nested_zips(extract_path)
                print(f"Extracted: {filename} → {extract_path}")

                os.remove(zip_path)
                print(f"Deleted top-level ZIP: {zip_path}")

            except zipfile.BadZipFile:
                print(f"Invalid ZIP: {filename}")

    print(f"\All zip files extracted into: {DESTINATION_FOLDER}")

if __name__ == "__main__":
    main()
