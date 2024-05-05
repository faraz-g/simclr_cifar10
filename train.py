from data_loader import CIFARSimCLRDataLoader, simclr_transform_pipeline
import torch
from model import SimCLR
from loss import NT_Xent
from pathlib import Path
import os

BATCH_SIZE = 128
TEMPERATURE = 0.5
NUM_EPOCHS = 100


def train(train_loader, model, criterion, optimizer):
    loss_epoch = 0

    for step, (x_i, x_j) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)

        z_i, z_j = model(x_i, x_j)
        loss = criterion(z_i, z_j)

        loss.backward()

        optimizer.step()

        print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        loss_epoch += loss.item()

    return loss_epoch


def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    dataloader = CIFARSimCLRDataLoader(transforms=simclr_transform_pipeline(32, 32))
    model = SimCLR()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100)
    criterion = NT_Xent(batch_size=BATCH_SIZE, temperature=TEMPERATURE, world_size=1)

    model = model.to(device=device)
    checkpoints_path = Path(__file__) / "checkpoints"
    os.makedirs(checkpoints_path, exist_ok=True)

    for epoch in range(0, NUM_EPOCHS):
        loss_epoch = train(dataloader, model, criterion, optimizer)
        scheduler.step()
        torch.save(model.state_dict(), os.path.join(checkpoints_path, f"model_epoch_{epoch}"))

        print(f"Epoch [{epoch}/{NUM_EPOCHS}]\t Loss: {loss_epoch / len(dataloader)}\t lr: {scheduler.get_last_lr()}")
