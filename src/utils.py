import os
import matplotlib.pyplot as plt

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
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

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