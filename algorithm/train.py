import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import train, normalize_input
from torch.utils.data import DataLoader
from dataset import MatrixDataset
from pythae.data import BaseDataset
from pythae.models import VAE, VAEConfig
from pythae.models import VQVAE, VQVAEConfig
from pythae.trainers import BaseTrainerConfig, BaseTrainer
from pythae.pipelines import TrainingPipeline
import numpy as np

def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    X = torch.load('data/observed_matrices.pt')
    Y = torch.load('data/real_matrices.pt')
    X = normalize_input(X)
    Y = normalize_input(Y)
    
    print('sanity check')
    print(X.shape, Y.shape)
    fig, axs = plt.subplots(1,3)
    x0 = X[0, 0, :100, :100]
    x1 = X[0, 1, :100, :100]
    y  = Y[0, 0, :100, :100]
    print(x0.shape, torch.unique(x0, return_counts=True))
    axs[0].imshow(x0, interpolation='none')
    print(x1.shape, torch.unique(x1, return_counts=True))
    axs[1].imshow(x1, interpolation='none')
    print(y.shape, torch.unique(y, return_counts=True))
    axs[2].imshow(y, interpolation='none')
    plt.show(block=True)

    # assume dataset is in the order of full_image_order.json
    dataset_length = X.shape[0]
    cutoff = int(dataset_length * 0.8)
    train_X, test_X = X[:cutoff], X[cutoff:]
    train_Y, test_Y = Y[:cutoff], Y[cutoff:]

    # pythae basedataset
    train_dataset = BaseDataset(train_X, train_Y)
    val_dataset = BaseDataset(test_X, test_Y)

    # train_dataset = MatrixDataset(train_X, train_Y)
    # val_dataset = MatrixDataset(test_X, test_Y)
    # train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True)

    print('Finished importing data')
    save_path = '/data/richard/HAI_simulation/algorithm/reconstructed'

    # Define Model

    # VAE
    # model_config = VAEConfig(
    #     input_dim=(2, X.shape[2], X.shape[3]),
    #     latent_dim=1e5
    # )
    # model = VAE(model_config)

    # VQ-VAE
    model_config = VQVAEConfig(
        input_dim=(2, X.shape[2], X.shape[3]),
        num_embeddings=1e5
    )
    model = VQVAE(model_config)

    # trainer and pipeline
    trainer_config = BaseTrainerConfig(
        num_epochs=10,
        learning_rate=1e-5,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        output_dir=save_path
    )
    pipeline = TrainingPipeline(
        training_config=trainer_config,
        model=model
    )
    # execute pipeline
    pipeline(
        train_data=train_dataset,
        eval_data=val_dataset
    )

    # reconstruction
    reconstructions: np.ndarray = model.reconstruct(test_X[:25].to(device)).detach().cpu()
    reconstructed_infection = reconstructions[:,0].squeeze()
    reconstructed_movement = reconstructions[:,1].squeeze()
    
    plt.rcParams["figure.figsize"] = (20,10)
    # bs = reconstructions.shape[0]
    bs = 5

    _, axs = plt.subplots(1, bs)
    for i in range(bs):
        axs[i].imshow(reconstructed_infection[i], interpolation='none')
    plt.savefig('reconstructed/reconstructed_infection.png')

    _, axs = plt.subplots(1, bs)
    for i in range(bs):
        axs[i].imshow(reconstructed_movement[i], interpolation='none')
    plt.savefig('reconstructed/reconstructed_movement.png')

    # Plot training and validation losses
    # training_losses = trainer.train_losses
    # validation_losses = trainer.eval_losses

    # num_epochs = 20
    # model = VAE().to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # loss_fn = vae_loss
    # Train the model
    # training_losses, validation_losses = train(model, optimizer, loss_fn, num_epochs, device, train_loader, val_loader, save_path)
    
    # Plot training and validation losses
    # plt.plot(training_losses, label='Training Loss')
    # plt.plot(validation_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss over Time')
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()