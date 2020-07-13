"""
Module for Pytorch dataset representations
"""

import torch
from torch.utils.data import Dataset

class SlicesDataset(Dataset):
    """
    This class represents an indexable Torch dataset
    which could be consumed by the PyTorch DataLoader class
    """
    def __init__(self, data):
        self.data = data

        self.slices = []

        for i, d in enumerate(data):
            for j in range(d["image"].shape[0]):
                self.slices.append((i, j))

    def __getitem__(self, idx):
        """
        This method is called by PyTorch DataLoader class to return a sample with id idx

        Arguments: 
            idx {int} -- id of sample

        Returns:
            Dictionary of 2 Torch Tensors of dimensions [1, W, H]
        """
        slc = self.slices[idx]
        sample = dict()
        sample["id"] = idx

        # slc is a tuple of two integers
        # self.data contains the {'image', 'seg', 'filename'} dicts from HippocampusDatasetLoader
        # length of self.data depends but it will be either the train, val, or test lengths

        # Really it just looks like a way to flatten and batch the images. Since we are treating this as
        # a 2D dataset of images and labels rather than a set of 3D volumes at some point we need to flatten
        # the 3D portion and here we are returning an image and label for a single slice of a 3D volume. So `slc`
        # is the two-dimensional index for what we're looking for.

        # You could implement caching strategy here if dataset is too large to fit
        # in memory entirely
        # Also this would be the place to call transforms if data augmentation is used
        
        # TASK: Create two new keys in the "sample" dictionary, named "image" and "seg"
        imgslc = self.data[slc[0]]['image'][slc[1]]
        lblslc = self.data[slc[0]]['seg'][slc[1]]

        sample['image'] = torch.tensor(imgslc).unsqueeze(0)
        sample['seg']   = torch.tensor(lblslc).unsqueeze(0)

        return sample

    def __len__(self):
        """
        This method is called by PyTorch DataLoader class to return number of samples in the dataset

        Returns:
            int
        """
        return len(self.slices)
