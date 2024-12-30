import torch

def train(model, train_loader, optimizer, criterion, device):
    """
    Single epoch of training. Returns (avg_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in train_loader:
        optimizer.zero_grad()

        # Move each Data object to device
        batch = [data.to(device) for data in batch]
        y_true = torch.stack([data.y for data in batch]).to(device)

        # Forward pass
        output = model(batch)

        # Compute loss & backprop
        loss = criterion(output, y_true)
        loss.backward()
        optimizer.step()

        # Stats
        total_loss += loss.item()
        total_correct += (output.argmax(dim=1) == y_true).sum().item()
        total_samples += len(y_true)

    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy
