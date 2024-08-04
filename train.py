import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import numpy as np
import torch.nn as nn
from dataset import AutoencoderDataset
from model import Autoencoder
from unet import UNet
import wandb

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 1000

def load_data(img_dir, batch_size):
    dataset = AutoencoderDataset(img_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train(device, model, dataloader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
        wandb.log({
            "Loss": epoch_loss
        })
        # if i == 0:
        #         fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        #         axs[0].imshow(np.transpose(inputs[0].numpy(), (1, 2, 0)))
        #         axs[0].set_title('Original')
        #         axs[1].imshow(np.transpose(outputs[0].numpy(), (1, 2, 0)))
        #         axs[1].set_title('Reconstructed')
        #         plt.show()
                

def main():
    img_dir = "C:/Users/jecel/Downloads/Imagens para autoencoder-20240719T144106Z-001/imgAutoencoder"

    wandb.init(
        project="Autoencoder",
        config={
            "learning_rate": 0.0001,
            "architecture": "UNet",
            "dataset": "2020-2022",
            "epochs": 1000,
        }
    )
    
    model = UNet(23, depth=5, merge_mode='concat')

    print("GPU disponível:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Nome da GPU:", torch.cuda.get_device_name(0))
        print("Versão do CUDA:", torch.version.cuda)
        print("Versão do cuDNN:", torch.backends.cudnn.version())
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    dataloader = load_data(img_dir, BATCH_SIZE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    train(device, model, dataloader, criterion, optimizer, EPOCHS)


if __name__ == "__main__":
    main()
