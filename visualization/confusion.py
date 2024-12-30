import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def compute_and_plot_confusion_matrix(model, val_loader, device, num_classes=20):
    """
    Evaluates the trained model on the validation set, prints final accuracy,
    and plots a confusion matrix.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            batch = [data.to(device) for data in batch]
            y_true = torch.stack([data.y for data in batch]).to(device)

            output = model(batch)
            preds = output.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_true.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Final validation accuracy for best model
    final_val_accuracy = (all_preds == all_labels).mean()
    print(f"Final Validation Accuracy (best model): {final_val_accuracy:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Validation Set)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
