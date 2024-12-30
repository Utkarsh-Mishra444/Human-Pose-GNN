import torch
from torch_geometric.loader import DataListLoader
import torch.nn.functional as F
import argparse

from src.data.dataset_mpose import load_mpose_data
from src.data.prepare_data import prepare_data
from src.model.gcn_lstm import GCN_LSTM
from src.training.train_fn import train
from src.training.validate_fn import validate
from src.visualization.plots import plot_training_curves
from src.visualization.confusion import compute_and_plot_confusion_matrix

def main(args):
    # -------------------------------
    # 1. Load dataset
    # -------------------------------
    X_train, y_train, X_val, y_val = load_mpose_data()
    print(f"Train data shape: {X_train.shape}")
    print(f"Val data shape:   {X_val.shape}")

    # -------------------------------
    # 2. Prepare PyG Data objects
    # -------------------------------
    train_data, val_data = prepare_data(X_train, y_train, X_val, y_val)
    train_loader = DataListLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataListLoader(val_data,   batch_size=args.batch_size, shuffle=False, drop_last=True)

    # -------------------------------
    # 3. Model, Criterion, Optimizer
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN_LSTM().to(device)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # -------------------------------
    # 4. Training Loop
    # -------------------------------
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float('inf')

    print("Starting Training ...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc     = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.4f}")

    # -------------------------------
    # 5. Plot Training Curves
    # -------------------------------
    plot_training_curves(args.epochs, train_losses, val_losses, train_accuracies, val_accuracies)

    # -------------------------------
    # 6. Evaluate Best Model
    # -------------------------------
    best_model = GCN_LSTM().to(device)
    best_model.load_state_dict(torch.load("best_model.pt"))
    compute_and_plot_confusion_matrix(best_model, val_loader, device, num_classes=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GCN-LSTM for pose classification.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    args = parser.parse_args()
    main(args)