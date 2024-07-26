import os
import glob
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Dict, List, Tuple, Optional

class CropDataset(Dataset):
    """
    Custom Dataset for loading and processing crop images stored in .tif files.

    Attributes:
        root_dir (str): Root directory containing crop image folders.
        transform (Optional[transforms.Compose]): Transformations to apply to the images.
        samples (List[Dict[str, Dict[str, str]]]): List of dictionaries containing paths to RGB band files and labels.
        bands (str): identifier (e.g. 'rgb', 'evi')
    """

    def __init__(self, root_dir: str, bands: str, transform: Optional[transforms.Compose] = None) -> None:
        """
        Initializes the dataset by loading sample paths and labels.

        Args:
            root_dir (str): Root directory containing crop image folders.
            transform (Optional[transforms.Compose]): Transformations to apply to the images.
            bands (str): identifier (e.g. 'rgb', 'evi')
        """
        self.root_dir = root_dir
        self.transform = transform
        self.bands = bands
        self.samples = self._load_samples()

    def _load_samples(self):
        """
        Loads sample paths and labels from the root directory.

        Returns:
            List[Dict[str, Dict[str, str]]]: List of dictionaries containing paths to RGB band files and labels.
        """
        samples = []
        crop_types = os.listdir(self.root_dir)
        for crop_type in crop_types:
            crop_dir = os.path.join(self.root_dir, crop_type)
            for sample_dir in glob.glob(os.path.join(crop_dir, '*', '*')):
                if self.bands == 'rgb':
                    sample = {
                        'paths': {
                            'B4': os.path.join(sample_dir, 'B4.tif'),
                            'B3': os.path.join(sample_dir, 'B3.tif'),
                            'B2': os.path.join(sample_dir, 'B2.tif')
                        },
                        'label': crop_type
                    }
                samples.append(sample)
        return samples

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves an image and its label for a given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, str]: Tuple containing the RGB image tensor and its label.
        """
        sample = self.samples[idx]
        if self.bands == 'rgb':
            try:
                image = self._load_rgb_image(sample['paths'])
            except Exception as e:
                print(f"Error loading image {sample['paths']}: {e}")
                # Handle the error or return a default value
                image = torch.zeros((3, 256, 256))  # Example default value

        label = sample['label']
        
        if self.transform:
            image = self.transform(image)

        return image, label

    def _load_rgb_image(self, paths):
        """
        Loads and combines B4, B3, and B2 bands into a single RGB image tensor.

        Args:
            paths (Dict[str, str]): Dictionary containing paths to B4, B3, and B2 .tif files.

        Returns:
            torch.Tensor: RGB image tensor.
        """
        try:
            with rasterio.open(paths['B4']) as src:
                B4 = src.read(1)
            with rasterio.open(paths['B3']) as src:
                B3 = src.read(1)
            with rasterio.open(paths['B2']) as src:
                B2 = src.read(1)
        except Exception as e:
            print(f"Error reading files: {paths}, {e}")
            raise e
        
        rgb_image = np.dstack((B4, B3, B2))
        rgb_image = rgb_image / np.max(rgb_image)  # Normalize to [0, 1]
        rgb_image = torch.tensor(rgb_image, dtype=torch.float32).permute(2, 0, 1)  # Convert to Tensor and rearrange to CxHxW

        return rgb_image
