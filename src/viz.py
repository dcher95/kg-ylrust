import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

def plot_images_with_labels(df, num_images, file_path_col, label_col, bands):
    """
    Plot a selected number of images with their labels, visualizing specified bands.

    Parameters:
    df (pd.DataFrame): DataFrame containing the file paths and labels.
    num_images (int): Number of images to plot.
    file_path_col (str): Column name for the file paths.
    label_col (str): Column name for the labels.
    bands (list): List of band names to visualize (e.g., ["B4", "B3", "B2"]).
    """
    # Select a subset of the DataFrame
    subset_df = df.sample(n=num_images)

    # Create subplots
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    for i, (idx, row) in enumerate(subset_df.iterrows()):
        # List to hold the bands
        band_images = []

        for band in bands:
            # Construct the band file path
            band_path = row[file_path_col] + f"/{band}.tif"
            
            # Read the band
            with rasterio.open(band_path) as src:
                band_images.append(src.read(1))

        if len(bands) > 1:
            # Stack the bands into an image
            image = np.dstack(band_images)
            
            # Normalize the pixel values to the range [0, 1] for display
            image = image / np.max(image)
        else:
            # Only one band, use it directly
            image = band_images[0]
            
            # Normalize the pixel values to the range [0, 1] for display
            image = image / np.max(image)
        
        # Plot the image
        if len(bands) > 1:
            axes[i].imshow(image)
        else:
            axes[i].imshow(image, cmap='gray')
        
        axes[i].set_title(row[label_col])
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
