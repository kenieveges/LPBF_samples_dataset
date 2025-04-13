# DirParse.py

import os
import pandas as pd
from typing import List, Dict


def is_tiff_file(filename: str) -> bool:
    """Check if the file is a TIFF image."""
    return filename.lower().endswith(('.tif', '.tiff'))


def normalize_filename(filename: str) -> str:
    """
    Removes the .tif/.tiff extension from filename.
    Keeps dot-separated identifiers (e.g., '2.1.tif' -> '2.1').
    """
    return os.path.splitext(filename)[0]


def parse_metadata_from_path(path: str, root_dir: str) -> Dict:
    """
    Extract metadata from file path:
    year, test_type, company, material, file_type, filename, and full path.
    Adds a unique ID per file.
    """
    try:
        relative_path = os.path.relpath(path, root_dir).replace("\\", "/")
        parts = relative_path.split("/")
        base_dir = parts[0]  # e.g., 2024_bend_mai_rs300
        file_type = parts[1]  # cropped, filled, masked
        file_name = parts[-1]
        name_only = normalize_filename(file_name)
        
        year, test_type, company, material = base_dir.split("_")

        # Unique ID pattern: year_testType_company_material_fileType_filename
        uid = f"{year[-2:]}_{test_type[0].upper()}_{company.upper()}_{material.upper()}_{file_type[-1:].upper()}_{name_only}"
        short_id = name_only.replace('.', '')
        return {
            "uid": uid,
            "id": short_id,
            "year": year,
            "test_type": test_type,
            "material": material,
            "company": company,
            "file_type": file_type,
            "path": os.path.abspath(path)
        }
    except Exception as e:
        raise ValueError(f"Failed to parse metadata from path: {path}") from e


def collect_tiff_files(root_dir: str) -> List[str]:
    """Walk through the directory tree and collect all .tif/.tiff file paths."""
    tiff_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if is_tiff_file(file):
                full_path = os.path.join(root, file)
                tiff_paths.append(full_path)
    return tiff_paths


def create_tiff_dataframe(root_dir: str) -> pd.DataFrame:
    """
    Create a DataFrame of TIFF files with extracted metadata and unique IDs.
    Columns: id, year, test_type, material, company, file_type, path
    """
    all_paths = collect_tiff_files(root_dir)
    records = []

    for path in all_paths:
        try:
            record = parse_metadata_from_path(path, root_dir)
            records.append(record)
        except ValueError as e:
            print(e)

    return pd.DataFrame(records)
