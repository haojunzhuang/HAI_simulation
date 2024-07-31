import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import train
from torch.utils.data import DataLoader
from dataset import MatrixDataset
from vae import VAE, vae_loss

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    X = torch.load('data/interpolated_images_75.pth')
    Y = torch.load('data/full_images.pth')

    # assume dataset is in the order of full_image_order.json
    train_X, test_X = X[:4000], X[4000:]
    train_Y, test_Y = Y[:4000], Y[4000:]

    train_dataset = MatrixDataset(train_X, train_Y)
    val_dataset = MatrixDataset(test_X, test_Y)

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True)

    # sanity checks
    # for x, y in train_loader:
    #     fig, axs = plt.subplots(1,2)
    #     print(x.shape, torch.unique(x, return_counts=True))
    #     axs[0].imshow(x[0, :100, :100], interpolation='none')
    #     print(y.shape, torch.unique(y, return_counts=True))
    #     axs[1].imshow(y[0, :100, :100], interpolation='none')
    #     break

    print('Finished importing data')

    save_path = '/data/richard/HAI_simulation/algorithm/reconstructed'
    num_epochs = 20

    model = VAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = vae_loss

    # Train the model
    training_losses, validation_losses = train(model, optimizer, loss_fn, num_epochs, device, train_loader, val_loader, save_path)
    
    # Plot training and validation losses
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Time')
    plt.legend()
    plt.show()