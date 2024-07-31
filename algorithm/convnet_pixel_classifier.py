import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import MatrixDataset

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # 3 input channels (RGB), 64 output channels
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )
        
        # Flatten and create latent representation
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256*93*50, 512)  # Adjust the size according to the encoder output
        self.fc2 = nn.Linear(512, 256*93*50)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Use sigmoid for normalized output
        )

    def forward(self, x):
        # Encoding
        x = self.encoder(x)
        # print(x.shape)
        
        # Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Reshape for decoding
        x = x.view(x.size(0), 256, 93, 50)
        
        # Decoding
        x = self.decoder(x)
        # print("Decoded x shape:", x.shape)
        return x
    
if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print('Using device:', device)

    print("Start Importing Data")
    X = torch.load('data/observed_matrices.pt').float()
    Y = torch.load('data/real_matrices.pt').float()
    # X = X.unsqueeze(1)  # Shape will be (batch_size, 1, 738, 400)
    # Y = Y.unsqueeze(1)
    print(X.shape, Y.shape)

    # assume dataset is in the order of full_image_order.json
    train_X, test_X = X[0:4000], X[4000:]
    train_Y, test_Y = Y[:4000], Y[4000:]
    # train_X, test_X = X[0:20], X[20:40]
    # train_Y, test_Y = Y[:20], Y[20:40]

    # train_dataset = MatrixDataset(train_X, train_Y)
    # val_dataset = MatrixDataset(test_X, test_Y)

    # train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True)

    # sanity checks
    # for x, y in train_loader:
    #     # fig, axs = plt.subplots(1,2)
    #     print(x.shape, torch.unique(x, return_counts=True))
    #     # axs[0].imshow(x[0, :100, :100], interpolation='none')
    #     print(y.shape, torch.unique(y, return_counts=True))
    #     # axs[1].imshow(y[0, :100, :100], interpolation='none')
    #     break

    # print('Finished importing data')

    # save_path = '/data/richard/HAI_simulation/algorithm/reconstructed'
    # num_epochs = 20

    # model = ConvAutoencoder().to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # loss_fn = nn.MSELoss()

    # print("Start Training")

    # Train the model
    # training_losses, validation_losses = train(model, optimizer, loss_fn, num_epochs, device, train_loader, val_loader, save_path)
    
    # training_losses = []
    # validation_losses = []

    # for epoch in range(num_epochs):
    #     model.train()
    #     running_loss = 0.0

    #     with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
    #         for i, (x, y) in enumerate(train_loader):
    #             y = F.pad(input=y, pad=(0, 0, 6, 0), mode='constant', value=0)  # Ensure y's dimension is a multiple of 8
    #             x, y = x.float().to(device), y.float().to(device)
    #             # print(x.shape)

    #             optimizer.zero_grad()
    #             y_pred = model(x)
    #             loss = loss_fn(y, y_pred)
    #             loss.backward()
    #             optimizer.step()

    #             running_loss += loss.item()

    #             # Update the progress bar with the current loss
    #             pbar.set_postfix(train_loss=running_loss / (i + 1))
    #             pbar.update(1)
        
    #             # if (i + 1) % 10 == 0:
    #             #     print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / (i + 1):.4f}')
            
    #         avg_train_loss = running_loss / len(train_loader)
    #         training_losses.append(avg_train_loss)
        
    #     model.eval()
    #     val_loss = 0.0

    #     with torch.no_grad():
    #         for i, (x, y) in enumerate(val_loader):
    #             y = F.pad(input=y, pad=(0, 0, 6, 0), mode='constant', value=0)  # Ensure y's dimension is a multiple of 8
    #             x, y = x.float().to(device), y.float().to(device)

    #             y_pred = model(x)
    #             loss = loss_fn(y, y_pred)
    #             val_loss += loss.item()
                
    #             # Save the first batch's reconstructed image for visualization
    #             if i == 0:
    #                 print(x.shape)
    #                 print(y_pred.shape)
    #                 save_reconstructed_images(torch.squeeze(x, dim=1), torch.squeeze(y_pred, dim=1), save_path, epoch)

    #     avg_val_loss = val_loss / len(val_loader)
    #     print(f'Validation Loss: {avg_val_loss:.4f}')
    #     validation_losses.append(avg_val_loss)
        
    # # Plot training and validation losses
    # plt.plot(training_losses, label='Training Loss')
    # plt.plot(validation_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss over Time')
    # plt.legend()
    # plt.show()