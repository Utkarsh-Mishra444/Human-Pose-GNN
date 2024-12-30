import matplotlib.pyplot as plt

def plot_training_curves(epochs, train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plots the loss and accuracy curves for train/val sets.
    """
    plt.figure(figsize=(12,4))

    # ---------- Loss Plot ----------
    plt.subplot(1,2,1)
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()

    # ---------- Accuracy Plot ----------
    plt.subplot(1,2,2)
    plt.plot(range(1, epochs+1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, epochs+1), val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()

    plt.tight_layout()
    plt.show()
