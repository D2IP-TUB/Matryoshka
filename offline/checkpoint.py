#!/usr/bin/env python3
import os
import sys
import subprocess
import re
import tempfile
import shutil
import tarfile
from pathlib import Path
from datetime import datetime

def extract_paths_from_log(log_file):
    """Extract file paths from log entries matching the specified pattern."""
    paths = []
    pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - \w+ - INFO - Processing table \d+ - (.+)'
    
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                paths.append(match.group(1).strip())
    
    return paths

def find_latest_log(directory):
    """Find the latest pipeline error log file."""
    log_pattern = re.compile(r'pipeline_(\d{8}_\d{6})_error\.log')
    logs = []
    
    for filename in os.listdir(directory):
        match = log_pattern.match(filename)
        if match:
            timestamp_str = match.group(1)
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            logs.append((timestamp, filename))
    
    if not logs:
        raise FileNotFoundError("No pipeline error logs found")
    
    logs.sort(reverse=True)
    return os.path.join(directory, logs[0][1])

def copy_from_container(container_path, host_dest):
    """Copy directory contents from Docker container to host."""
    container_name = os.environ.get('DOCKER_CONTAINER', 'your_container_name')
    
    # Ensure destination exists
    os.makedirs(host_dest, exist_ok=True)
    
    # Add trailing slash to copy contents, not the directory itself
    # This makes docker cp copy the contents of container_path into host_dest
    source_with_slash = container_path.rstrip('/') + '/.'
    
    cmd = ['docker', 'cp', f'{container_name}:{source_with_slash}', host_dest]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"Copied contents of {container_path} from container to {host_dest}")

def remove_files_from_tar(tar_path, files_to_remove, output_path):
    """Create a new tar.gz archive excluding specified files."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract the archive
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(temp_dir)
        
        # Remove specified files
        for file_path in files_to_remove:
            full_path = os.path.join(temp_dir, file_path)
            if os.path.exists(full_path):
                os.remove(full_path)
                print(f"Removed: {file_path}")
            else:
                print(f"Warning: File not found in archive: {file_path}")
        
        # Create new archive
        print(f"Creating new archive at {output_path}...")
        with tarfile.open(output_path, 'w:gz') as tar:
            tar.add(temp_dir, arcname='.')

def process_logs_and_clean_data(host_dest, container_source, host_data_path):
    """
    Main function to process logs and clean data archive.
    
    Args:
        host_dest: Host directory path (destination for container copy)
        container_source: Docker container directory path (source)
        host_data_path: Host data directory path (.tar.gz archive)
    """
    
    # Step 1 & 2: Copy source directory from container to host
    print("Step 1-2: Copying from container...")
    os.makedirs(host_dest, exist_ok=True)
    copy_from_container(container_source, host_dest)
    
    # Step 3: Find latest log file
    print("\nStep 3: Finding latest log file...")
    latest_log = find_latest_log(host_dest)
    print(f"Latest log: {latest_log}")
    
    # Step 4: Extract paths from log
    print("\nStep 4: Extracting paths from log...")
    paths = extract_paths_from_log(latest_log)
    print(f"Found {len(paths)} file paths to remove")
    for path in paths[:5]:  # Show first 5
        print(f"  - {path}")
    if len(paths) > 5:
        print(f"  ... and {len(paths) - 5} more")
    
    # Step 5: Create cleaned copy of data archive
    print("\nStep 5: Creating cleaned data archive...")
    if not os.path.exists(host_data_path):
        raise FileNotFoundError(f"Data archive not found: {host_data_path}")
    
    # Create output path for cleaned archive
    base_name = os.path.basename(host_data_path)
    output_path = os.path.join(
        os.path.dirname(host_data_path),
        base_name.replace('.tar.gz', '_cleaned.tar.gz')
    )
    
    remove_files_from_tar(host_data_path, paths, output_path)
    print(f"\nCleaned archive created: {output_path}")
    
    return output_path

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <host_dest_dir> <container_source_dir> <host_data_archive>")
        print("\nExample:")
        print("  python script.py /tmp/logs /app/logs /data/archive.tar.gz")
        print("\nNote: Set DOCKER_CONTAINER environment variable with container name/ID")
        sys.exit(1)
    
    host_dest = sys.argv[1]
    container_source = sys.argv[2]
    host_data_path = sys.argv[3]
    
    try:
        output_path = process_logs_and_clean_data(host_dest, container_source, host_data_path)
        print(f"\n✓ Success! Cleaned archive: {output_path}")
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()