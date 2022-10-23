import os, glob, math
import h5py
import numpy as np
import torch
import pandas as pd
import math
from torch.utils import data

class HDF5Dataset(data.Dataset):
    """An HDF5 Dataset class derived from torch.utils.Dataset.

    Input params:
        dir_path: Path to the directory containing the dataset (one or more HDF5 files).
        metadata_file_path: An optional master_file which may contain the labels associated with the dataset. This must be set for GalaxyZoo dataset.
    """
    def __init__(self, dir_path, recursive=False, load_data=False, min_pixel_dims=None, max_pixel_dims=None, data_cache_size=3, pad=False, transform=None, label_keys=None):
        super().__init__()
        self.label_keys = label_keys
        self.transform = transform
        self.total_num_imgs = 0
        self.imgs_per_file = []
        self.filenames = []
        self.min_pixel_dims = min_pixel_dims
        self.max_pixel_dims = max_pixel_dims
        self.pad = pad
        self.dataset = []
        self.dataset_labels = []
        valid_imgs_count = 0
        assert(os.path.isdir(dir_path))

        if(recursive):
            files = glob.glob(dir_path+'**/*.hdf5')
        else:
            files = glob.glob(dir_path+'*.hdf5')
        if len(files) < 1:
            raise RuntimeError('No hdf5 files found in directory.')
        for idx, hdf5filename in enumerate(sorted(files)):
            hdf5file = h5py.File(hdf5filename, 'r')
            dset_keys = list(hdf5file.keys())

            self.filenames.append(hdf5filename)
            for i in dset_keys:
                if(hdf5file[i].shape[1] >= min_pixel_dims and hdf5file[i].shape[1] <= max_pixel_dims):
                    valid_imgs_count += 1
                    sample = (hdf5file[i][()])
                    
                    if(self.pad):
                        amt_pad = self.max_pixel_dims-sample.shape[1];
                        rem = amt_pad % 2;
                        amt_pad = math.floor(amt_pad/2)
                        
                        if(amt_pad > 0 or rem > 0):
                            npad = [(0,0), (amt_pad+rem, amt_pad), (amt_pad+rem, amt_pad)]
                            sample = np.pad(sample, pad_width=npad, mode='constant', constant_values=0)
                    self.dataset.append(sample.astype(float))
                    #print(hdf5file[i].attrs.keys())
                    label = list(map(lambda x: hdf5file[i].attrs.get(x), self.label_keys))
                    label /= np.sum(label)
                    self.dataset_labels.append(label.astype(float))
                    #print(len(sample))
            self.imgs_per_file.append(valid_imgs_count)
        #print(self.dset_file_size)
        self.total_num_imgs = valid_imgs_count
        #print(len(self.dataset))
        #print(len(self.dataset_labels))
        assert(self.total_num_imgs == len(self.dataset))
        assert(len(self.dataset) == len(self.dataset_labels))
        print('Total number of images: %d' % self.total_num_imgs)
        if self.total_num_imgs < 1:
            raise RuntimeError('No datasets found in hdf5 files.')

    def __len__(self):
        return self.total_num_imgs

    """Overloaded function to get a sample and label pair. Called by DataLoader class during training/testing.

        Input params:
            index: The index of the sample and its associated label that should be loaded.
        Output params:
            sample: The requested sample from the dataset.
            label: The label corresponding to sample.
    """

    #TODO: Should cache data to improve performance.

    def __getitem__(self, index):
        sample = self.dataset[index]
        label = self.dataset_labels[index]
        return (sample, label)
