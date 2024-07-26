from pathlib import Path

data_path = Path(
    "/workspaces/kg-ylrust/data/input/beyond-visible-spectrum-ai-for-agriculture-2023-p2/share/"
)

train_dir = data_path / "train/"
validation_dir = data_path / "validation/"

rgb_bands = ['B4.tif', 'B3.tif', 'B2.tif']