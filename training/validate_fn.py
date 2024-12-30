import torch

def validate(model, val_loader, criterion, device):
    """
    Single epoch of validation. Returns (avg_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = [data.to(device) for data in batch]
            y_true = torch.stack([data.y for data in batch]).to(device)

            output = model(batch)
            loss = criterion(output, y_true)

            total_loss += loss.item()
            total_correct += (output.argmax(dim=1) == y_true).sum().item()
            total_samples += len(y_true)

    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy
