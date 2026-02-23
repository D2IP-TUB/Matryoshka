import zipfile
import sys
import tempfile
import shutil
from pathlib import Path


def process_zip(input_zip, output_zip, target_file, data_dir="datasets_USA"):
    # 1) Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # 2) Extract ZIP
        with zipfile.ZipFile(input_zip, 'r') as zip_ref:
            zip_ref.extractall(tmp_path)
        
        dataset_path = tmp_path / data_dir
        if not dataset_path.exists() or not dataset_path.is_dir():
            raise FileNotFoundError(f"Directory '{data_dir}' not found in ZIP")
        
        # 3) List and sort files in dataset_path
        files = sorted([f for f in dataset_path.iterdir() if f.is_file()])
        
        # 4) Remove all files up to and including target_file
        found = False
        for f in files:
            f.unlink()
            if f.name == target_file:
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Target file '{target_file}' not found in ZIP")
        
        # 5) Re-zip remaining files
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zip_out:
            for path in tmp_path.rglob('*'):
                zip_out.write(path, path.relative_to(tmp_path))
    
    print(f"Created {output_zip}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_zip.py input.zip output.zip")
        sys.exit(1)
    
    input_zip = sys.argv[1]
    output_zip = sys.argv[2]
    TARGET_FILE = "USA_CSV0000000000035766.csv"
    
    process_zip(input_zip, output_zip, TARGET_FILE)
