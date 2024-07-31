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

def train(model, optimizer, loss_fn, num_epochs, device, train_loader, val_loader, save_path):
    training_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (x, y) in enumerate(tqdm(train_loader)):
            y = F.pad(input=y, pad=(0, 0, 6, 0), mode='constant', value=0)  # Ensure y's dimension is a multiple of 8
            x, y = x.to(device), y.type(torch.LongTensor).to(device)

            optimizer.zero_grad()
            y_pred, mu, logvar = model(x)
            loss = loss_fn(y, y_pred, mu, logvar)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / (i + 1):.4f}')
        
        avg_train_loss = running_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        
        avg_val_loss = validate(model, loss_fn, device, val_loader, save_path, epoch)
        validation_losses.append(avg_val_loss)
        
    return training_losses, validation_losses

def validate(model, loss_fn, device, dataloader, save_path, epoch):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            y = F.pad(input=y, pad=(0, 0, 6, 0), mode='constant', value=0)  # Ensure y's dimension is a multiple of 8
            x, y = x.to(device), y.type(torch.LongTensor).to(device)

            y_pred, mu, logvar = model(x)
            loss = loss_fn(y, y_pred, mu, logvar)
            val_loss += loss.item()

            # Save the first batch's reconstructed image for visualization
            if i == 0:
                save_reconstructed_images(x, y_pred, save_path, epoch)

    avg_val_loss = val_loss / len(dataloader)
    print(f'Validation Loss: {avg_val_loss:.4f}')
    return avg_val_loss

def save_reconstructed_images(input_images, reconstructed_images, save_path, epoch):
    input_images = input_images.cpu().numpy()
    reconstructed_images = reconstructed_images.cpu().numpy()
    # reconstructed_images = reconstructed_images.argmax(axis=1)
    # print(input_images.shape, reconstructed_images.shape)

    fig, axs = plt.subplots(2, input_images.shape[0], figsize=(15, 5))

    for i in range(input_images.shape[0]):
        axs[0, i].imshow(input_images[i], interpolation='none')
        axs[0, i].set_title("Original")
        axs[0, i].axis('off')

        axs[1, i].imshow(reconstructed_images[i], interpolation='none')
        axs[1, i].set_title("Reconstructed")
        axs[1, i].axis('off')

    plt.savefig(f'{save_path}/reconstructed_epoch_{epoch}.png')
    plt.close()
