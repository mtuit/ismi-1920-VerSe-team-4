# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 11:55:48 2020

@author: Casper
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import SimpleITK as sitk
import random
from src.data.data_loader import generate_heatmap
import os
from skimage import transform



def visualize_graph(path, prediction, threshold = 0.5):
    """
        This function visualizes the centroid mask (true) and the output heatmap (pred)
        Channels First on the output heatmap, path variable + extension
    """
    itkimg1 = sitk.ReadImage(os.path.join("data/processed/normalized-images/centroid_masks/", path))
    results_true = np.array(sitk.GetArrayFromImage(itkimg1))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    goalshape = results_true.shape
    (xs_lbl, ys_lbl, zs_lbl) = _set_truth_plot(ax, results_true)
    (xs_hmp, ys_hmp, zs_hmp) = _set_prediction_plot(ax, prediction, goalshape, threshold)

    xs_all = xs_lbl + xs_hmp
    ys_all = ys_lbl + ys_hmp
    zs_all = zs_lbl + zs_hmp
    _set_dims(ax,xs_all,ys_all,zs_all)
    plt.show()


def visualize(results_true, prediction, threshold = 0.5):
    """
    Visualize the output from a model (batch size = 1)
    """
    #itkimg1 = sitk.ReadImage(os.path.join("data/processed/normalized-images/images/", path))
    #results_true = np.array(sitk.GetArrayFromImage(itkimg1))
    fig = plt.figure()
    coords = _get_locations_output(prediction[0], results_true.shape, threshold = 0.5)
    average = [a_tuple[2] for a_tuple in coords]
    average = sum(average)/len(average)
    locs = [(a_tuple[0],a_tuple[1]) for a_tuple in coords]
    #x,y = zip(*locs)
    for (x,y) in locs:
        plt.scatter(y,x)
        
    plt.imshow(results_true[:,:, int(average)])
    plt.show()
    

def distances(path, heatmaps, threshold = 0.5):
    itkimg1 = sitk.ReadImage(os.path.join("data/processed/normalized-images/centroid_masks/", path))
    results_true = np.array(sitk.GetArrayFromImage(itkimg1))
    for x in range(25):
        output = "v{}\t".format(x+1)
        coord_true = None;
        coord_pred = None;
        labels_one_hot = np.where(results_true == x+1, 1, 0)
        if np.max(labels_one_hot) > 0.1:
            coord_true = np.unravel_index(labels_one_hot.argmax(), labels_one_hot.shape)
            output += " at {}.{}.{}\t".format(*coord_true)
        else:
            output += " absent;\t    "
        if np.max(heatmaps[x]) > threshold:
            coord_pred = _single_heatmap_to_loc(heatmaps[x], results_true.shape) 
            output += " is spotted at loc: {}.{}.{}".format(*coord_pred)
        else:
            output += " has no prediction."
        if(coord_true is not None) and (coord_pred is not None):
            dist = int(np.linalg.norm(np.array(coord_true) - np.array(coord_pred)))
            output += "\t dist: {}".format(dist)
      

def _single_heatmap_to_loc(heatmap, goalshape):
    """
        input is single heatmap
        location is extracted with the argmax policy
    """
    # upscaling of the image back to its original shape:
    heatmap_upscale = transform.resize(heatmap, goalshape)
    x = np.unravel_index(heatmap_upscale.argmax(), heatmap_upscale.shape)
    return x


def _get_locations_output(heatmap, goalshape, threshold, return_labels=False):
    """
        input is 25-dim heatmap
        returns list of predicted locations of vertebrae
    """
    locations = []
    for index, hm in enumerate(np.rollaxis(heatmap, 3)):
        if np.max(hm) > threshold:
            # If we want to return the labels as well we have to add those to the tuple, incremented by 1
            # since vertebrae's start at 1 and indexing at 0
            if return_labels:
                location = _single_heatmap_to_loc(hm, goalshape)
                locations.append((index + 1, *location))
            else:
                locations.append(_single_heatmap_to_loc(hm, goalshape))
                
    return locations


def _get_locations_labels(labels):
    """
        input: original labels
        returns locations of ground truth
    """
    locations = []
    for x in range(25):
        labels_one_hot = np.where(labels == x+1, 1, 0)
        if np.max(labels_one_hot) > 0.1:
            locations.append(np.unravel_index(labels_one_hot.argmax(), labels_one_hot.shape))
    return locations


    
def _set_prediction_plot(ax, prediction, goalshape, threshold = 0.5):
    coords = _get_locations_output(prediction, goalshape, threshold)
    xs_hmp = [a_tuple[0] for a_tuple in coords]
    ys_hmp = [a_tuple[1] for a_tuple in coords]
    zs_hmp = [a_tuple[2] for a_tuple in coords]
    ax.plot(xs_hmp, ys_hmp, zs_hmp)
    ax.scatter(xs_hmp, ys_hmp, zs_hmp)
    return (xs_hmp, ys_hmp, zs_hmp)
    

def _set_truth_plot(ax, results_true):
    coords = _get_locations_labels(results_true)
    xs_lbl = [a_tuple[0] for a_tuple in coords]
    ys_lbl = [a_tuple[1] for a_tuple in coords]
    zs_lbl = [a_tuple[2] for a_tuple in coords]
    ax.plot(xs_lbl, ys_lbl, zs_lbl)
    ax.scatter(xs_lbl, ys_lbl, zs_lbl)
    return (xs_lbl, ys_lbl, zs_lbl)
    

def _set_dims(ax, xs,ys,zs):
    """
        Changes scale symmetrically of axis of Axes3d object for 3d data
    """
    x_size = np.max(xs) - np.min(xs)
    y_size = np.max(ys) - np.min(ys)
    z_size = np.max(zs) - np.min(zs)
    margin = np.max([x_size,y_size,z_size])/2
    x_origin = (np.min(xs) + np.max(xs))/2
    y_origin = (np.min(ys) + np.max(ys))/2
    z_origin = (np.min(zs) + np.max(zs))/2
    ax.set_xlim(x_origin - margin, x_origin + margin)
    ax.set_ylim(y_origin - margin, y_origin + margin)
    ax.set_zlim(z_origin - margin, z_origin + margin)

if __name__ == '__main__':
    
    # fake prediction variable:
    itkimg2 = sitk.ReadImage("data/processed/normalized-images/centroid_masks/verse007.mha")
    prediction = generate_heatmap(np.array(sitk.GetArrayFromImage(itkimg2)), (128,64,64), 25, debug=False) 
   
    # visualize input with prediction (example) (includes reshaping of prediction):
    #visualize_graph("verse005.mha", prediction, threshold = 0.5)
    #distances("verse005.mha", prediction, threshold = 0.5)
    visualize("verse007.mha", prediction, threshold = 0.5)