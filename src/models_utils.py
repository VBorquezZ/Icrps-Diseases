"""
Functions used to train and validate models.
"""
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np

def predict(model_path, sample_image, my_transforms):
    model = torch.load(model_path)
    model.eval()
    image = Image.open(sample_image)
    image = my_transforms(image)[None, :, :, :]
    x = model(image)
    return "Mask" if x[0].argmax(dim=0) else "No Mask"


def check_accuracy(loader, model, device):
    print("Checking accuracy on given data")
    ground_truth = []
    predicted = []
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            ground_truth.append(y)
            predicted.append(predictions)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()
    return torch.cat(ground_truth, dim=0).tolist(), torch.cat(predicted, dim=0).tolist()

def train_model(model,train_loader, val_loader, criterion, optimizer,scheduler, epochs, device, early_stopping):
    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []
    for epoch in range(epochs):

        losses = []

        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device=device)
            targets = targets.to(device=device)

            scores = model(data)
            loss = criterion(scores, targets)

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        for data, target in val_loader:
            data = data.to(device=device)
            target = target.to(device=device)

            output = model(data)

            loss = criterion(output, target)

            valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        mean_loss = sum(losses) / len(losses)

        scheduler.step(mean_loss)
        print(
            f"Cost at epoch {epoch+1} is {mean_loss} | valid_loss: {valid_loss:.5f} | train_loss: {train_loss:.5f}")

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        early_stopping(valid_loss, model, optimizer)

        if early_stopping.early_stop:
            print("Early stopping")
            break
