import os
import glob
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from config.configure import mask_images_path
from src import logger
def get_dataframe(path: str) -> pd.DataFrame:
    """
    Create a DataFrame containing image paths, mask paths, and labels.

    Args:
        path (str): path [mask_images]

    Returns:
        pd.DataFrame: DataFrame with image paths, mask paths, and labels.
    """

    image_masks = glob.glob(path)
    image_paths = [file_path.replace("_mask", '') for file_path in image_masks]

    def labels(mask_path):
        label = []
        for mask in mask_path:
            img = Image.open(mask)
            label.append(1) if np.array(img).sum() > 0 else label.append(0)
        return label

    mask_labels = labels(image_masks)

    df = pd.DataFrame({
        'image_path': image_paths,
        'mask_path': image_masks,
        'label': mask_labels
    })

    return df

class MRIDataset(Dataset):
    def __init__(self, paths, transform):
        """
        Custom dataset for MRI images.

        Args:
            paths (pd.DataFrame): DataFrame containing mask paths.
            transform: Data augmentation and transformation pipeline.
        """
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_path, mask_path = self.paths.iloc[idx]
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        image = np.array(image).astype(np.float32) / 255.
        mask = np.array(mask).astype(np.float32) / 255.

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            return transformed['image'], transformed['mask'].unsqueeze(0)
        else:
            transformed = ToTensorV2()(image=image, mask=mask)
            return transformed['image'], transformed['mask'].unsqueeze(0)


def data_loaders(batch_size,num_workers, train_split=False) -> DataLoader:
    
    logger.info(f"Preprocessing Data")
    df = get_dataframe(mask_images_path)

    train_transforms = A.Compose([
    A.Resize(224, 224, p=1.0),
    A.RandomBrightnessContrast(p=0.2),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    ToTensorV2(),
    ])

    # Only reshape val and test data
    val_transforms = A.Compose([
        A.Resize(224, 224, p=1.0),
        ToTensorV2(),
    ])

    # splitting the dataset
    train_x, val_x, train_y, val_y = train_test_split(df.drop('label',axis=1), df.label,test_size=0.3)
    val_x , test_x, val_y, test_y = train_test_split(val_x, val_y, test_size = 0.2)

    train_data = MRIDataset(train_x, train_transforms)
    val_data = MRIDataset(val_x, val_transforms)
    test_data = MRIDataset(test_x[test_y == 1], val_transforms)


    # train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    if train_split:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        return train_loader, val_loader
    else:
        test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
        return test_loader 