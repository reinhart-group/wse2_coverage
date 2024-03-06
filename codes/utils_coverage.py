
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
import torch.optim as optim

from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms


import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.rcParams.update(mpl.rcParamsDefault) # this is to clear figure settings
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pylab import *
from matplotlib import  style


import time
import os
import copy
import random


import pandas as pd
from torchvision.io import read_image

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader, TensorDataset

import json
import pickle

import pretrained_microscopy_models as pmm
import torch.utils.model_zoo as model_zoo
import warnings


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score  
from sklearn.metrics import mean_squared_error as mse

from timeit import default_timer as timer



def lightness(im):
    """Implement the lightness transform found here:
    https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color
    """
    R, G, B = np.array(im).transpose([2, 0, 1])  # get each color channel
    L = (0.299*R + 0.587*G + 0.114*B)  # one possible formula for luminance

    return L


def autocrop(afm):
    # convert to lightness only
    L = lightness(afm)
    # find bounds
    afm_bounds = np.zeros(4, dtype=int)
    for i in range(2):
        min_pixels = np.argwhere(L.min(axis=i) < L.max()).flatten()
        gap_idx = np.argwhere(np.diff(min_pixels) > 1).flatten()[0]
        min_pixels = min_pixels[:gap_idx]
        if len(min_pixels) < 1:
            min_pixels = [L.shape[i]]

        median_pixels = np.argwhere(np.median(L, axis=i) < L.max()).flatten()
        if len(median_pixels) < 1:
            median_pixels = [L.shape[i]]

        afm_bounds[0+i] = min(min_pixels[0], median_pixels[0])
        afm_bounds[2+i] = min(min_pixels[-1], median_pixels[-1])

    return afm.crop(box=afm_bounds)


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, max_index=None, transform=None):
        """
        Args:
            root_dir (str): Directory containing the image files.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        if max_index is None:
            self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]
            self.image_files = sorted(self.image_files)
        else:
            self.image_files = []
            for f in os.listdir(root_dir):
                patch_num = int(f.split('_patch_')[1].split('_label_')[0])
                if f.endswith('.png') and patch_num < max_index:
                    self.image_files.append(f)
        
        # Extract and store groups from filenames
        self.groups = [f.split('sample_')[1].split('_image')[0] for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_filename)
        
        # Extract label from the filename
        label_str = os.path.splitext(self.image_files[idx])[0].split('label_')[1].split('.png')[0]
        label = float(label_str)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, self.groups[idx]
        
        

def grouped_train_test_split(train_group, val_group, test_group, dataset, test_size=0.1, random_state=0):
    groups = dataset.groups
    unique_groups = list(set(groups))
    print('len of groups: ', len(groups))
    print('len of unique groups: ', len(unique_groups))
    train_val_indices, test_indices = [], []
    
 
    for index, group in enumerate(groups):
        if group in train_group:
            train_val_indices.append(index)
        elif group in val_group:
            train_val_indices.append(index)
        elif group in test_group:
            test_indices.append(index)
            
    train_indices, val_indices = train_test_split(train_val_indices, test_size=test_size, random_state=random_state)
        
   
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset
    
    

# Load pre-trained ResNet model

# Choose a specific layer to extract features from
def get_features_from_model(model, data_loader):
    model.eval()
    # Choose a specific layer to extract features from
    target_layer = model.avgpool  # can be layer1 - layer4, avgpool

    # Define a hook to capture the activations from the target layer

    activations = []
    labels_list = []

    def hook_fn(module, input, output):
        activations.append(output)
        #activations += output



    hook = target_layer.register_forward_hook(hook_fn)

    # Load and preprocess an image


    #def get_features(images):
    for inputs, labels, groups in data_loader:

        inputs, labels = inputs.float(), labels.float().unsqueeze(1)
        
        labels_list.append(labels.tolist())
    
#    for image in images:
   
#        item_tensor = torch.tensor(image, dtype=torch.float32)
#        input_batch = item_tensor.unsqueeze(0)     
    # Pass the input through the model to trigger the hook and capture activations

        with torch.no_grad():

            _ = model(inputs)


    # Remove the hook to prevent memory leaks

    hook.remove()
 

# The activations list now contains the convolutional activations from the target layer
    return activations, labels_list

# Use 

    
    
def model_inference(model, dataloader):
    #model.eval()
    true_label = []
    pred_label = []

    with torch.no_grad():
        for inputs, labels, groups in dataloader:

            inputs, labels = inputs.float().to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(inputs)


            true_label += labels.cpu().tolist()
            pred_label += outputs.cpu().tolist()

        pred_mae = mae(true_label, pred_label)
        pred_rmse = np.sqrt(mse(true_label,pred_label))
        
    return true_label, pred_label, pred_mae, pred_rmse
   


import pretrained_microscopy_models as pmm

#-----helper function to split data into batches
def divide_batch(l, n): 
    for i in range(0, l.shape[0], n):  
        yield l[i:i + n,::] 

# https://github.com/choosehappy/PytorchDigitalPathology
def coverage_models_inference(io, model, device = None, batch_size = 1, patch_size = 224, num_classes=1):

    # This will not output the first class and assumes that the first class is wherever the other classes are not!

    #io = preprocessing_fn(io)
    io = io[0].numpy().transpose(1, 2, 0)
    io_shape_orig = np.array(io.shape)
    stride_size = patch_size // 2
    if device is None:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # add half the stride as padding around the image, so that we can crop it away later
    io = np.pad(io, [(stride_size // 2, stride_size // 2), (stride_size // 2, stride_size // 2), (0, 0)],
                mode="reflect")

    io_shape_wpad = np.array(io.shape)

    # pad to match an exact multiple of unet patch size, otherwise last row/column are lost
    npad0 = int(np.ceil(io_shape_wpad[0] / patch_size) * patch_size - io_shape_wpad[0])
    npad1 = int(np.ceil(io_shape_wpad[1] / patch_size) * patch_size - io_shape_wpad[1])

    io = np.pad(io, [(0, npad0), (0, npad1), (0, 0)], mode="constant")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        arr_out = pmm.segmentation_training.extract_patches(io, (patch_size, patch_size, 3), stride_size)

    arr_out_shape = arr_out.shape
    
    
    arr_out = arr_out.reshape(-1, patch_size, patch_size, 3)

    # in case we have a large network, lets cut the list of tiles into batches
    #output = np.zeros((0, num_classes, patch_size, patch_size))
    output = []
    for batch_arr in divide_batch(arr_out, batch_size):
        arr_out_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2).astype('float32')).to(device)

        # ---- get results
        output_batch = model(arr_out_gpu)

       
        output_batch = output_batch.detach().cpu().numpy()
        crystal = np.prod(arr_out_gpu.cpu().numpy().size)*output_batch # np.prod(rec.size)
        ##output = np.append(output, output_batch, axis=0)
        output.append(crystal)
    
    coverage = np.sum(output)/np.prod(arr_out.size)
    
    return coverage
