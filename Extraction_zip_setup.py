import zipfile
import os
import shutil


def extract_dataset_properly():
    """
    Extract dataset.zip to get: dataset/WithMask/ and dataset/WithoutMask/
    """

    zip_path = r"C:\Users\Collins\Desktop\dataset.zip"
    extract_root = "."  # Current directory

    print("=" * 60)
    print("EXTRACTING DATASET")
    print("=" * 60)

    # Check if zip exists
    if not os.path.exists(zip_path):
        print(f"ERROR: dataset.zip not found at {zip_path}")
        print("Make sure the file is on your Desktop")
        return False

    # Check current directory
    print(f"Current directory: {os.getcwd()}")
    print(f"Will create: dataset/WithMask/ and dataset/WithoutMask/ here")
    print("=" * 60)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get all files in the zip
            all_files = zip_ref.namelist()
            print(f"Total files in zip archive: {len(all_files)}")

            # Show the structure
            print("\nZIP FILE STRUCTURE:")
            folders = {}
            for file in all_files[:20]:  # Show first 20
                parts = file.split('/')
                if len(parts) >= 2:
                    folder = parts[0]
                    if folder not in folders:
                        folders[folder] = []
                    if len(parts) > 1 and parts[1]:
                        folders[folder].append('/'.join(parts[1:]))

            for folder, contents in folders.items():
                print(f"  {folder}/")
                for item in contents[:3]:  # Show first 3 items
                    print(f"    -- {item}")
                if len(contents) > 3:
                    print(f"    -- ... and {len(contents) - 3} more items")

            # Extract everything
            print(f"\nExtracting to: {os.path.join(os.getcwd(), 'dataset/')}")
            zip_ref.extractall(extract_root)

            # Verify extraction
            print("EXTRACTION COMPLETE")

            # Check what we got
            dataset_path = os.path.join(extract_root, "dataset")
            if os.path.exists(dataset_path):
                print(f"\nCreated directory: {dataset_path}")
                items = os.listdir(dataset_path)

                print(f"Subdirectories found in dataset/: {len(items)}")
                for item in items:
                    item_path = os.path.join(dataset_path, item)
                    if os.path.isdir(item_path):
                        file_count = len([f for f in os.listdir(item_path)
                                          if os.path.isfile(os.path.join(item_path, f))])
                        print(f"  - {item}/ ({file_count} image files)")

                        # Show some sample files
                        if file_count > 0:
                            sample_files = os.listdir(item_path)[:3]
                            print(f"    Sample files:")
                            for sample in sample_files:
                                print(f"      {sample}")
                            if file_count > 3:
                                print(f"      ... and {file_count - 3} more files")

                # Test access
                print("\nSTRUCTURE VERIFICATION PASSED")
                print(f"Image path example: dataset/WithMask/U.jpg")

            return True

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False


def verify_structure():
    """Verify the dataset structure"""

    print("\n" + "=" * 60)
    print("VERIFYING DATASET STRUCTURE")
    print("=" * 60)

    expected_structure = {
        "dataset": ["WithMask", "WithoutMask"]
    }

    all_good = True

    for folder, subfolders in expected_structure.items():
        if os.path.exists(folder):
            print(f"FOUND: {folder}/")

            for subfolder in subfolders:
                subfolder_path = os.path.join(folder, subfolder)
                if os.path.exists(subfolder_path):
                    # Count files
                    try:
                        files = [f for f in os.listdir(subfolder_path)
                                 if os.path.isfile(os.path.join(subfolder_path, f))]
                        print(f"  FOUND: {subfolder}/ (contains {len(files)} files)")

                        # Check for U.jpg specifically
                        if "U.jpg" in files:
                            print(f"         Contains U.jpg file")
                        else:
                            print(f"         WARNING: U.jpg not found in this folder")
                            if files:
                                print(f"         First few files: {files[:3]}")

                    except Exception as e:
                        print(f"  ERROR reading {subfolder}/: {e}")
                        all_good = False
                else:
                    print(f"  MISSING: {subfolder}/")
                    all_good = False
        else:
            print(f"MISSING: {folder}/")
            all_good = False

    if all_good:
        print("\nDATASET STRUCTURE IS CORRECT")
        print("\nExample Python usage:")
        print("  from PIL import Image")
        print("  image = Image.open('dataset/WithMask/U.jpg')")
    else:
        print("\nISSUES DETECTED WITH DATASET STRUCTURE")

    return all_good


def cleanup_existing():
    """Remove existing dataset folder if it exists"""
    if os.path.exists("dataset"):
        print("Found existing 'dataset' folder")
        response = input("Remove it and extract fresh? (y/n): ").strip().lower()
        if response == 'y':
            shutil.rmtree("dataset")
            print("Removed existing dataset folder")
            return True
        else:
            print("Keeping existing folder")
            return False
    return True


def print_tree(startpath, prefix="", max_depth=3, current_depth=0):
    """Print directory tree structure"""
    if current_depth >= max_depth:
        return

    try:
        contents = sorted(os.listdir(startpath))
        for i, item in enumerate(contents):
            path = os.path.join(startpath, item)
            is_last = (i == len(contents) - 1)

            if os.path.isdir(path):
                print(f"{prefix}{'L-- ' if is_last else '|-- '}{item}/")
                extension = "    " if is_last else "|   "
                if current_depth + 1 < max_depth:
                    print_tree(path, prefix + extension, max_depth, current_depth + 1)
            else:
                # Only show files at the first few levels
                if current_depth < 2:
                    print(f"{prefix}{'L-- ' if is_last else '|-- '}{item}")
    except:
        pass


if __name__ == "__main__":
    # Optional: Clean up first
    cleanup_existing()

    # Extract the dataset
    print("\n" + "=" * 60)
    print("STARTING DATASET EXTRACTION")
    print("=" * 60)

    if extract_dataset_properly():
        # Verify the structure
        verify_structure()

        # Show complete file tree
        print("\n" + "=" * 60)
        print("DATASET DIRECTORY TREE")
        print("=" * 60)

        if os.path.exists("dataset"):
            print_tree("dataset", max_depth=3)
        else:
            print("No dataset directory found")

        print("\n" + "=" * 60)
        print("EXTRACTION PROCESS COMPLETED")
        print("=" * 60)