import glob
import shutil
import os
import cv2
import imageio
import numpy as np
import pandas as pd


from tqdm import tqdm
import imageio.v2 as imageio

#from glob import glob

import torch

import random


import pretrained_microscopy_models as pmm
import segmentation_models_pytorch as smp
import albumentations as albu
from skimage import img_as_ubyte
from skimage.color import gray2rgb

from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

import json
import pickle


from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score  
from sklearn.metrics import mean_squared_error as mse

import matplotlib as mpl# random_state = 36
import matplotlib.pyplot as plt
from matplotlib import  style


def make_inference(data_set, data_set_vis):
    
    pr = np.zeros(0)
    gt = np.zeros(0)

    # for n in random_index:
    pbar = tqdm.tqdm(np.arange(len(data_set)))
    for n in pbar:

        image_vis = data_set_vis[n][0].astype('uint8')
        image, gt_mask = data_set[n]
        gt_mask = gt_mask[0, :, :] # remove single dimension to visualize

        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        pr_cov = np.sum(pr_mask) / np.prod(pr_mask.shape)
        gt_cov = np.sum(gt_mask) / np.prod(gt_mask.shape)

        pr = np.hstack([pr, pr_cov])
        gt = np.hstack([gt, gt_cov])
    rmse = np.sqrt(np.mean((pr-gt)**2))
    r2 = r2_score(pr, gt)
    print('RMSE: ', rmse, 'R2: ', r2)
    return gt, pr, rmse, r2
    
import warnings


#-----helper function to split data into batches
def divide_batch(l, n): 
    for i in range(0, l.shape[0], n):  
        yield l[i:i + n,::] 

# https://github.com/choosehappy/PytorchDigitalPathology
def segmentation_models_inference(io, model, preprocessing_fn, device = None, batch_size = 8, patch_size = 512,
                                  num_classes=1, probabilities=None):

    # This will not output the first class and assumes that the first class is wherever the other classes are not!

    io = preprocessing_fn(io)
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
    output = np.zeros((0, num_classes, patch_size, patch_size))
    for batch_arr in divide_batch(arr_out, batch_size):
        arr_out_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2).astype('float32')).to(device)

        # ---- get results
        output_batch = model.predict(arr_out_gpu)

        # --- pull from GPU and append to rest of output
        if probabilities is None:
            output_batch = output_batch.detach().cpu().numpy().round()
        else:
            output_batch = output_batch.detach().cpu().numpy()

        output = np.append(output, output_batch, axis=0)

    output = output.transpose((0, 2, 3, 1))

    # turn from a single list into a matrix of tiles
    output = output.reshape(arr_out_shape[0], arr_out_shape[1], patch_size, patch_size, output.shape[3])

    # remove the padding from each tile, we only keep the center
    output = output[:, :, stride_size // 2:-stride_size // 2, stride_size // 2:-stride_size // 2, :]

    # turn all the tiles into an image
    output = np.concatenate(np.concatenate(output, 1), 1)

    # incase there was extra padding to get a multiple of patch size, remove that as well
    output = output[0:io_shape_orig[0], 0:io_shape_orig[1], :]  # remove paddind, crop back

    if probabilities is None:
        if num_classes == 1:
            return output.astype('bool')
        else:
            return output[:, :, 1:].astype('bool')
    else:
        if num_classes == 1:
            output[:,:,0] = output[:,:,0] > probabilities
            return output.astype('bool')
        else:
            for i in range(num_classes-1): #don't care about background class
                output[:,:,i+1] = output[:,:,i+1] > probabilities[i]
            return output[:, :, 1:].astype('bool')
            
            



def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def iou(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union


def iou_value(model, preprocessing_fn,data_images, data_masks):
    IOU_list = []
    for k in range(len(data_images)):

        im_path = data_images[k]
        annot_path = data_masks[k]
        im = imageio.imread(im_path)
        im = img_as_ubyte(im)


        pred = segmentation_models_inference(im, model, preprocessing_fn, batch_size=8, patch_size=512, num_classes=1)

        truth = imageio.imread(annot_path)
        truth = img_as_ubyte(truth)

        pred1 = torch.from_numpy(pred)
        truth1 = torch.from_numpy(truth[:, :, 0]).unsqueeze(2)
        #print(truth1.shape, pred1.shape)
        IOU = iou(pred1, truth1).tolist() #[0]
        IOU_list.append(IOU)
    
    return IOU_list

def cov_value(model, preprocessing_fn,data_images, data_masks):
    cov_truth_list = []
    cov_pred_list = []
    for k in range(len(data_images)):

        im_path = data_images[k]
        annot_path = data_masks[k]
        im = imageio.imread(im_path)
        im = img_as_ubyte(im)


        pred = segmentation_models_inference(im, model, preprocessing_fn, batch_size=4, patch_size=512, num_classes=1)

        truth = imageio.imread(annot_path)
        truth = img_as_ubyte(truth)

        cov_truth = np.sum(np.array(truth)==1)/truth.size #/np.prod(truth1.size)
        cov_pred = np.sum(np.array(pred)==1)/pred.size #/np.prod(pred.size)

        cov_truth_list.append(cov_truth)
        cov_pred_list.append(cov_pred)
    
    return cov_truth_list, cov_pred_list            
