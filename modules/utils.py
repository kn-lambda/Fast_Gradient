import os
import sys

import numpy as np
from PIL import Image
from numpy import random
import subprocess
import datetime


#########################################################
# parameters ############################################

#directory of the ImageNet image files
#sub-directories must correspond to the categories
IMAGE_DIR = "/home/kouichi/Data/Imagenet/images"


#########################################################
# functions #############################################

def execute(cmd_str):
    cmd = cmd_str.split(' ')
    return subprocess.check_call(cmd)


def get_images_in_dir(dir_path, index=None):
    
    image_paths = []
    image_indices = []
    
    for image in os.listdir(dir_path):

        if len(image.split('.')) <= 1: # for the case image is directory
            continue

        path = os.path.join(dir_path, image)
        image_paths.append(path)
        if index is None:
            image_indices.append(image.split('__')[0])
        else:
            image_indices = [index] * len(image_paths)
            
    return image_paths, image_indices


def sampling_categories(index_list, n_samples = 20, index_must = []):
    
    remain_index = [i for i in index_list if i not in index_must]
    random.shuffle(remain_index)
    
    n_remain = n_samples - len(index_must)
    index_samples = sorted(index_must + remain_index[:n_remain])
    
    return index_samples


def sampling_images(index_samples, n_pics, index_dirname):

    image_paths = []
    image_indices = []
    
    sampling_dir = "./sampling_images/" + datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
    execute("mkdir -p {}".format(sampling_dir))
    
    for i in index_samples:
        category_dir = os.path.join(IMAGE_DIR, index_dirname[i])
        image_list = os.listdir(category_dir)
        if len(image_list) == 0:
            continue
        random.shuffle(image_list)
        
        for image in image_list[:n_pics]:
            image_path = os.path.join(category_dir, image)
            copy_path = os.path.join(sampling_dir, "{0}__{1}".format(i, image))
            execute("cp {0} {1}".format(image_path, copy_path))
            image_paths.append(copy_path)
            image_indices.append(i)
    
    return sampling_dir, image_paths, image_indices


