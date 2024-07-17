import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
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

def create_crop_classification_df(directory):

    roots = []
    for root, dirs, files in os.walk(directory):
        # Don't care about earlier parts of hierarchy. Just right before bands.
        if len(root.split("/")) > 8:
            roots.append(root)

    # Create a DataFrame
    df = pd.DataFrame(roots, columns=["File Path"])


    # Apply the extraction function to each file path
    df[["Crop", "Plot ID", "Date"]] = df["File Path"].apply(extract_columns)
    return df

def extract_columns(file_path):
        parts = file_path.split("/")
        crop_classification = parts[-3]  # 'soybean'
        plot_id = parts[-2]  # '000919'
        file_name = os.path.basename(file_path)  # Get the file name
        date = file_name.split("_")[0][:8]  # '20201109' from '20201109T...'

        return pd.Series(
            [crop_classification, plot_id, date],
            index=["Crop", "Plot ID", "Date"],
        )

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

def stratified_partition(df, id_col, train_size=0.7, val_size=0.15, holdout_size=0.15, random_state=42):
    if train_size + val_size + holdout_size != 1.0:
        raise ValueError("The sum of train_size, val_size, and test_size must be 1.0")
    
    # Prepare the Stratified Shuffle Split for train and remaining (validation + test)
    stratified_split_1 = StratifiedShuffleSplit(n_splits=1, test_size=(val_size + holdout_size), random_state=random_state)
    
    # Prepare the Stratified Shuffle Split for validation and test
    stratified_split_2 = StratifiedShuffleSplit(n_splits=1, test_size=(holdout_size / (val_size + holdout_size)), random_state=random_state)
    
    # Get the unique plot IDs and their associated labels
    unique_plot_ids = df[id_col].unique()
    plot_id_labels = df.groupby(id_col).size().values
    
    # First split: Train and (Validation + Test)
    train_index, remaining_index = next(stratified_split_1.split(unique_plot_ids, plot_id_labels))
    train_plot_ids = unique_plot_ids[train_index]
    remaining_plot_ids = unique_plot_ids[remaining_index]
    
    # Subset the original DataFrame into Train and Remaining DataFrames
    train_df = df[df[id_col].isin(train_plot_ids)]
    remaining_df = df[df[id_col].isin(remaining_plot_ids)]
    
    # Get the labels for the remaining plot IDs
    remaining_plot_id_labels = remaining_df.groupby(id_col).size().values
    
    # Second split: Validation and Test
    val_index, test_index = next(stratified_split_2.split(remaining_plot_ids, remaining_plot_id_labels))
    val_plot_ids = remaining_plot_ids[val_index]
    test_plot_ids = remaining_plot_ids[test_index]
    
    # Subset the Remaining DataFrame into Validation and Test DataFrames
    val_df = remaining_df[remaining_df[id_col].isin(val_plot_ids)]
    test_df = remaining_df[remaining_df[id_col].isin(test_plot_ids)]
    
    return train_df, val_df, test_df

def move_files(df, new_base_path = None, split = 'Holdout'):
    '''
    Move train folders to holdout. Make sure to name new folders based on plot + timestamp as might have multiple timeframes for single plot.
    '''
    move_df = df.copy()

    for index, row in move_df.iterrows():

        
        if split == 'Holdout':
            new_path = os.path.join(new_base_path, row["Plot ID"] + "-" + row["Date"])

        if split == 'Validation':
            new_path = row['File Path'].replace("train", "validation")

        shutil.copytree(
            row["File Path"],
            new_path,
            dirs_exist_ok=True,
        )
        shutil.rmtree(row["File Path"])

        # Update the DataFrame with the new path
        move_df.at[index, "File Path"] = new_path

    return move_df

def remove_empty_folders(directory):
    """
    Recursively remove empty folders in the specified directory.
    """
    # Traverse the directory tree
    for root, dirs, files in os.walk(directory, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # Check if the directory is empty
            if not os.listdir(dir_path):  # True if the directory is empty
                os.rmdir(dir_path)  # Remove the empty directory
                print(f"Removed empty directory: {dir_path}")