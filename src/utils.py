import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def download_competition_data(competition_name, input_folder):
    """
    Download and unzip competition data from Kaggle.

    Parameters:
    competition_name (str): The name of the Kaggle competition.
    input_folder (str): The folder where the competition data should be stored.

    Returns:
    None
    """
    if competition_name == "":
        return

    os.chdir(input_folder)
    os.system(f'rm -rf {competition_name} && echo "Removed existing version"')
    os.system(f'kaggle competitions download -c {competition_name} -p {competition_name} --force && echo "Downloaded"')
    os.system(f'unzip {competition_name}/{competition_name}.zip -d {competition_name} && echo "Unzipped"')

def band_plot(images, names, num_rows, num_cols, title):
    # Create a subplot
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

    # Plot each image
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images[i])
        ax.set_title(names[i])
        ax.axis("off")  # Turn off axis numbers and ticks

    # Add an overall title to the plot
    fig.suptitle(title, fontsize=16)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust top to make room for suptitle
    plt.show()

def count_subfolders(directory, depth=1):
    try:
        if depth < 1:
            raise ValueError("Depth must be 1 or greater.")
        
        def count_folders_at_depth(dir_path, current_depth):
            if current_depth > depth:
                return 0
            count = 0
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                if os.path.isdir(item_path):
                    count += 1
                    if current_depth < depth:
                        count += count_folders_at_depth(item_path, current_depth + 1)
            return count

        # Start counting from the initial directory
        subfolder_count = count_folders_at_depth(directory, 1)
        return subfolder_count

    except Exception as e:
        print(f"An error occurred: {e}")
        return 0

def create_crop_classification_df(base_path):
    data = {
        "Crop": [],
        "Plot ID": [],
        "Date": [],
        "Timestamp": [],
        "Tile": [],
        "Band": [],
        "File Path": [],
    }

    # Traverse the directory structure
    for crop in os.listdir(base_path):
        crop_path = os.path.join(base_path, crop)
        if os.path.isdir(crop_path):
            for plot_id in os.listdir(crop_path):
                plot_path = os.path.join(crop_path, plot_id)
                if os.path.isdir(plot_path):
                    for date_time_tile in os.listdir(plot_path):
                        dt_tile_path = os.path.join(plot_path, date_time_tile)
                        if os.path.isdir(dt_tile_path):
                            for band_file in os.listdir(dt_tile_path):
                                if band_file.endswith(".tif"):
                                    # Construct the shortened file path
                                    short_file_path = os.path.join(crop, plot_id, date_time_tile, band_file)

                                    # Extract information from date_time_tile
                                    date, timestamp_tile = (
                                        date_time_tile.rsplit("_", 2)[0],
                                        date_time_tile.rsplit("_", 2)[1] + "_" + date_time_tile.rsplit("_", 2)[2],
                                    )
                                    timestamp, tile = timestamp_tile.split("_T")
                                    band = band_file.split(".")[0]

                                    # Append to data
                                    data["Crop"].append(crop)
                                    data["Plot ID"].append(plot_id)
                                    data["Date"].append(date)
                                    data["Timestamp"].append(timestamp)
                                    data["Tile"].append(tile)
                                    data["Band"].append(band)
                                    data["File Path"].append(short_file_path)

    # Create DataFrame
    df = pd.DataFrame(data)
    return df

import pandas as pd

def find_duplicate_plot_ids(df):
    # Check for duplicate Plot IDs across different crops
    duplicates = df[df.duplicated(subset='Plot ID', keep=False)]
    duplicate_plot_ids = duplicates.groupby('Plot ID')['Crop'].unique().reset_index()

    # Filter out Plot IDs that are associated with multiple crops
    multiple_crops = duplicate_plot_ids[duplicate_plot_ids['Crop'].apply(lambda x: len(x) > 1)]

    if not multiple_crops.empty:
        print("Plot IDs found in multiple crops:")
        return multiple_crops
    else:
        print("No Plot IDs found in multiple crops.")

def assign_holdout_plot_ids(df, seed=42):
    # Initialize a new column for holdout indicator
    df['Holdout'] = False
    
    # Get unique crops
    crops = df['Crop'].unique()
    
    # Iterate over unique crops
    for crop in crops:
        crop_df = df[df['Crop'] == crop]
        
        # Get unique Plot IDs for the current crop
        unique_plot_ids = crop_df['Plot ID'].unique()
        
        # Determine the number of Plot IDs to select for holdout (20%)
        holdout_count = int(0.2 * len(unique_plot_ids))
        
        # Randomly sample Plot IDs with a seed for reproducibility
        holdout_plot_ids = np.random.RandomState(seed).choice(unique_plot_ids, size=holdout_count, replace=False)
        
        # Update the Holdout column for the selected Plot IDs
        df.loc[df['Plot ID'].isin(holdout_plot_ids), 'Holdout'] = True
    
    return df