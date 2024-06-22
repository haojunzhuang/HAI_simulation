import os, pickle, torch, json
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Compose, ToTensor, Resize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

def png_to_tensor(image_path:str):
    image = torch.tensor(plt.imread(image_path)[:,:,0])
    value_map = {0.1254902:0, 0.22745098:1, 0.26666668:2, 0.36862746:3, 0.99215686:4}
    transformed_image = torch.empty_like(image, dtype=torch.float)

    for key,value in value_map.items():
        transformed_image[image == key] = value

    return transformed_image

def interpolate(partial_images:torch.Tensor, full_images:torch.Tensor, ratio:float):
    """
    Interpolate between parial and full image datasets with the extent specified by "ratio"
    Return the whole set of new interpolated partial image of shape (num_images, height, width)

    Input: 
    - partial_images: tensor of size (num_images, height, width) with each entry being a value in 0,1,2,3,4
    - full_images: tensor of size (num_images, height, width) with each entry being a value in 0,2,3,4
    - ratio: A number between 0 and 1, with 0 returning the partial image and 1 returning the full image

    Output:
    - interpolated_images: tensor of size (num_images, height, width)
    """
    num_images, height, width = partial_images.shape
    interpolated_images = partial_images.clone()
    
    for i in tqdm(range(num_images)):
        partial_image = partial_images[i]
        full_image = full_images[i]
        
        # Check if the images are pairs
        assert torch.all(partial_image[partial_image != 1] == full_image[partial_image != 1]), \
            "The partial and full images do not match at positions with values 0/2/3/4"

        ones_indices = (partial_image == 1).nonzero(as_tuple=False)
        num_to_replace = int(ratio * len(ones_indices))
        
        if num_to_replace > 0:
            selected_indices = ones_indices[torch.randperm(len(ones_indices))[:num_to_replace]]
            for idx in selected_indices:
                interpolated_images[i, idx[0], idx[1]] = full_image[idx[0], idx[1]]
    
    return interpolated_images

def prepare_tensor_images(data_path:str):
    """
    Given a folder of full and partial images in png, return two tensor of size (num_sample, height, width),
    as well as their order in terms of filenames
    """
    full_image_order = []
    partial_image_order = []
    full_images = []
    partial_images = []
    for image_path in tqdm(sorted(os.listdir(data_path))):
        if "full" in image_path:
            full_image_order.append(image_path)
            full_images.append(png_to_tensor(data_path+'/'+image_path))
        else:
            partial_image_order.append(image_path)
            partial_images.append(png_to_tensor(data_path+'/'+image_path))

    full_images = torch.stack(full_images)
    partial_images = torch.stack(partial_images)

    return full_image_order, partial_image_order, full_images, partial_images


def normalize_input(input_image_tensor:torch.Tensor):
    """
    Given a input image in [0,4], scale it to [-1, 1]
    """
    return (input_image_tensor - 2) / 2


if __name__ == "__main__":
    FULL_IMAGE_PATH = "generated_matrices_cutted/sw_3_501_full_1.png"
    PARTIAL_IMAGE_PATH = "generated_matrices_cutted/sw_3_501_partial_1.png"
    FOLDER_PATH = "generated_matrices_cutted"
    # full_image_tensor = png_to_tensor(FULL_IMAGE_PATH)
    # partial_image_tensor = png_to_tensor(PARTIAL_IMAGE_PATH)
    # print(full_image_tensor)
    # print(normalize_input(full_image_tensor))
    # print(np.unique(full_image_tensor, return_counts=True))
    # print(np.unique(normalize_input(full_image_tensor), return_counts=True))
    # full_image_order, partial_image_order, full_images, partial_images = prepare_tensor_images(FOLDER_PATH)
    # with open('data/full_image_order.json', 'w') as file:
    #     json.dump(full_image_order, file)
    # with open('data/partial_image_order.json', 'w') as file:
    #     json.dump(partial_image_order, file)
    # torch.save(full_images, 'data/full_images.pth')
    # torch.save(partial_images, 'data/partial_images.pth')

    # partial_images = torch.tensor([
    #     [[0, 1, 2], [3, 1, 4], [1, 1, 1]],
    #     [[0, 1, 2], [3, 1, 4], [1, 1, 1]]
    # ])
    # full_images = torch.tensor([
    #     [[0, 2, 2], [3, 4, 4], [2, 2, 2]],
    #     [[0, 2, 2], [3, 4, 4], [2, 2, 2]]
    # ])
    # ratio = 0

    # interpolated_images = interpolate(partial_images, full_images, ratio)
    # print(interpolated_images)

    # full_images = torch.load('data/full_images.pth')
    # partial_images = torch.load('data/partial_images.pth')
    # interpolated_images_20 = interpolate(partial_images, full_images, ratio=0.2)
    # interpolated_images_50 = interpolate(partial_images, full_images, ratio=0.5)
    # interpolated_images_75 = interpolate(partial_images, full_images, ratio=0.75)
    # interpolated_images_90 = interpolate(partial_images, full_images, ratio=0.9)
    # interpolated_images_95 = interpolate(partial_images, full_images, ratio=0.95)
    # torch.save(interpolated_images_20, 'data/interpolated_images_20.pth')
    # torch.save(interpolated_images_50, 'data/interpolated_images_50.pth')
    # torch.save(interpolated_images_75, 'data/interpolated_images_75.pth')
    # torch.save(interpolated_images_90, 'data/interpolated_images_90.pth')
    # torch.save(interpolated_images_95, 'data/interpolated_images_95.pth')