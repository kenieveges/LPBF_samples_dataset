# plottingFuncs.py

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import pandas as pd
from typing import Optional, Dict


def plot_sample_images(
    df: pd.DataFrame,
    figure_dir: str,
    current_date: str,
    sample_id: Optional[str] = None,
    save: bool = False
):
    """
    Plot a collage of TIFF images (masked, filled, cropped) for a given sample ID.
    
    Args:
        df (pd.DataFrame): DataFrame with columns ['id', 'file_type', 'path', ...]
        figure_dir (str): Directory to save figures
        current_date (str): Used in filename for saving
        sample_id (str, optional): ID of the sample to plot. If None, random is used.
        save (bool): Whether to save the output figure
    """
    if sample_id:
        sample_data = df[df['id'] == sample_id]
        if sample_data.empty:
            print(f"Sample ID '{sample_id}' not found in DataFrame.")
            return
        base_metadata = sample_data.iloc[0]
    else:
        # Randomly choose a sample id
        sample_id = df['id'].sample(1).values[0]
        sample_data = df[df['id'] == sample_id]
        base_metadata = sample_data.iloc[0]

    # Extract all image types for this sample
    img_types = ['masked', 'filled', 'cropped']
    image_paths = []

    for img_type in img_types:
        row = sample_data[sample_data['file_type'] == img_type]
        if not row.empty:
            image_paths.append((img_type, row.iloc[0]['path']))

    if not image_paths:
        print(f"No available images found for sample ID '{sample_id}'")
        return

    # Plotting
    fig, axs = plt.subplots(1, len(image_paths), figsize=(5 * len(image_paths), 5))
    axs = [axs] if len(image_paths) == 1 else axs

    title = f"Sample ID: {sample_id}\n{base_metadata['test_type'].capitalize()}, {base_metadata['material']}, {base_metadata['company'].upper()}, {base_metadata['year']}"
    plt.suptitle(title, fontsize=14)

    for ax, (img_type, path) in zip(axs, image_paths):
        try:
            img = io.imread(path)
            ax.imshow(img, cmap='gray')
            ax.set_title(f"{img_type.capitalize()}", fontsize=12)
            ax.axis('off')
        except Exception as e:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Error loading image', ha='center', va='center')
            print(f"Failed to load {path}: {e}")

    plt.tight_layout()
    plt.subplots_adjust(top=0.8)

    if save:
        os.makedirs(figure_dir, exist_ok=True)
        output_path = os.path.join(figure_dir, f"{current_date}_{sample_id}_collage.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")

    plt.show()
