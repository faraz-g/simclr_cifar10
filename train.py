from data_loader import get_cifar_dataset
import torch
from model import SimCLR
from loss import NT_Xent
from pathlib import Path
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

BATCH_SIZE = 512
TEMPERATURE = 0.5
NUM_EPOCHS = 200
EXPERIMENT_NAME = "512bshlr"


def train(train_loader, model, criterion, optimizer, device):
    loss_epoch = 0

    for (x_i, x_j), _ in tqdm(train_loader):
        optimizer.zero_grad()
        x_i = x_i.to(device)
        x_j = x_j.to(device)

        z_i, z_j = model(x_i, x_j)
        loss = criterion(z_i, z_j)

        loss.backward()

        optimizer.step()

        loss_epoch += loss.item()

    return loss_epoch


def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    dataset = get_cifar_dataset(root_folder="/home/faraz/Documents/code/simclr_cifar10/cifar10", train=True, aug=True)
    dataloader = DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count() - 1, drop_last=True
    )
    model = SimCLR()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=NUM_EPOCHS)
    criterion = NT_Xent(batch_size=BATCH_SIZE, temperature=TEMPERATURE)

    model = model.to(device=device)
    checkpoints_path = Path(__file__).parent / "checkpoints"
    os.makedirs(checkpoints_path, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        loss_epoch = train(
            train_loader=dataloader, model=model, criterion=criterion, optimizer=optimizer, device=device
        )
        scheduler.step()

        print(f"Epoch [{epoch}/{NUM_EPOCHS}]\t Loss: {loss_epoch / len(dataloader)}\t lr: {scheduler.get_last_lr()}")
        if epoch % 40 == 0 or epoch == NUM_EPOCHS - 1:
            torch.save(model.state_dict(), os.path.join(checkpoints_path, f"model_epoch_{epoch}_{EXPERIMENT_NAME}"))


if __name__ == "__main__":
    main()
