import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, Dict, List

def load_images(mask_path: str, filled_path: str, cropped_path: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
    try:
        # Convert absolute paths to relative paths (relative to current working directory)
        cwd = os.getcwd()
        mask_path = os.path.relpath(os.path.normpath(mask_path), cwd)
        filled_path = os.path.relpath(os.path.normpath(filled_path), cwd)
        if cropped_path:
            cropped_path = os.path.relpath(os.path.normpath(cropped_path), cwd)

        images = {
            'mask': cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE),
            'crossec': cv2.imread(filled_path, cv2.IMREAD_GRAYSCALE)
        }
        if cropped_path and os.path.exists(cropped_path):
            images['cropped'] = cv2.imread(cropped_path, cv2.IMREAD_GRAYSCALE)
        return images
    except Exception as e:
        print(f"Error loading images with OpenCV: {e}")
        return None

def calculate_center_of_mass(image: np.ndarray) -> (int, int):
    moments = cv2.moments(image)
    if moments['m00'] != 0:
        return int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])
    return 0, 0

def analyze_pores(mask: np.ndarray, crossec_contour: List[np.ndarray], center: (int, int), coef_px2mm: float) -> Dict[str, List]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cx, cy = center
    data = {"areas": [], "distances": [], "edge_distances": [], "x_coords": [], "y_coords": []}

    for contour in contours:
        m = cv2.moments(contour)
        if m['m00'] == 0:
            continue
        x = int(m['m10'] / m['m00'])
        y = int(m['m01'] / m['m00'])
        area = cv2.contourArea(contour)
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        edge_dist = cv2.pointPolygonTest(crossec_contour[0], (x, y), True)

        data["areas"].append(area)
        data["distances"].append(dist)
        data["edge_distances"].append(edge_dist)
        data["x_coords"].append(x * coef_px2mm)
        data["y_coords"].append(y * coef_px2mm)

    return data

def process_single_sample(df: pd.DataFrame, unique_id: str, coef_px2mm: float) -> Optional[pd.DataFrame]:
    sample_data = df[df['id'] == unique_id].iloc[0]
    year, test_type, company, material = sample_data["year"], sample_data["test_type"], sample_data["company"], sample_data["material"]

    try:
        mask_path = df[(df['id'] == unique_id) & (df['file_type'] == 'masked')].iloc[0]['path']
        filled_path = df[(df['id'] == unique_id) & (df['file_type'] == 'filled')].iloc[0]['path']
    except IndexError:
        print(f"Missing images for {unique_id}")
        return None

    cropped_row = df[(df['id'] == unique_id) & (df['file_type'] == 'cropped')]
    cropped_path = cropped_row.iloc[0]['path'] if not cropped_row.empty else ""

    images = load_images(mask_path, filled_path, cropped_path)
    if not images or images['mask'] is None or images['crossec'] is None:
        print(f"Failed to load required images for {unique_id}")
        return None

    cube_contour, _ = cv2.findContours(images['crossec'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cube_contour:
        print(f"No contour in crossec for {unique_id}")
        return None

    center = calculate_center_of_mass(images['crossec'])
    pores = analyze_pores(images['mask'], cube_contour, center, coef_px2mm)

    return pd.DataFrame({
        "sample_ID": unique_id,
        "year": year,
        "test_type": test_type,
        "company": company,
        "material": material,
        "dist": pores["distances"],
        "edge_dist": pores["edge_distances"],
        "area": np.array(pores["areas"]) * (coef_px2mm ** 2),
        "x": pores["x_coords"],
        "y": pores["y_coords"],
        "masked_path": mask_path,
        "filled_path": filled_path,
        "cropped_path": cropped_path,
        "s_area": images['crossec'].sum() * (coef_px2mm ** 2),
        "num_pores": len(pores["areas"])
    })

def export_results(df: pd.DataFrame, current_date: str, tables_dir: str) -> str:
    """
    Exports the final results DataFrame to a CSV file for further analysis.
    """
    os.makedirs(tables_dir, exist_ok=True)
    output_path = os.path.join(tables_dir, f"{current_date}_processed_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Exported processed data to: {output_path}")
    return output_path

def process_sample_images(df: pd.DataFrame, coef_px2mm: float, current_date: str, tables_dir: str) -> pd.DataFrame:
    results = []
    for uid in tqdm(df['id'].unique(), desc="Processing samples"):
        result = process_single_sample(df, uid, coef_px2mm)
        if result is not None:
            results.append(result)

    if not results:
        print("No samples processed.")
        return pd.DataFrame()

    output_df = pd.concat(results, ignore_index=True)
    export_results(output_df, current_date, tables_dir)

    print(f"\nSamples processed: {output_df['sample_ID'].nunique()}")
    print(f"Total pores analyzed: {len(output_df)}")
    print(f"Companies in dataset: {output_df['company'].unique()}")

    return output_df
