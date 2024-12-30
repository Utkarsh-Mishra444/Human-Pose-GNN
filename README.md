# Spatio-temporal Human Pose Classification using Graph Neural Network and LSTMs

## Overview

This repository contains the code and documentation for classifying human poses in **video sequences** using a **Graph Convolutional Network (GCN)** integrated with **Long Short-Term Memory (LSTM)**. By combining **OpenPose** (or other pose-estimation frameworks) to detect keypoints with a GCN for spatial relationships, followed by LSTM for temporal dynamics, we can effectively perform pose classification tasks.

### Key Highlights
1. **Pose Estimation**: We use [MPOSE](https://github.com/lcjmmp/mpose) with OpenPose under the hood for joint detection.  
2. **GCN**: Learns spatial relationships between joints as a graph, where nodes represent human joints and edges represent biomechanical connections.  
3. **LSTM**: Captures temporal dependencies across multiple frames in a sequence.  
4. **Classification**: A final feed-forward network (linear layers) outputs the pose class probabilities.

---

## Project Poster
<p align="center">
  <img src="Poster.jpg" alt="Poster Snapshot" />
</p>

## Repository Structure
```
Human-Pose-GNN/
├── main.py               # Entry point for training + evaluation
├── requirements.txt      # Python dependencies
├── README.md             
│
├── data/
│   ├── dataset_mpose.py   # Loads MPOSE data
│   ├── edge_index.py      # Defines the graph edges
│   └── prepare_data.py    # Creates torch_geometric Data objects
│
├── model/
│   └── gcn_lstm.py        # GCN+LSTM model definition
│
├── training/
│   ├── train_fn.py        
│   ├── validate_fn.py     
│   └── __init__.py
│
├── visualization/
│   ├── plots.py           # Plots training curves
│   └── confusion.py       # Confusion matrix generation

```

---

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/<your-username>/my_pose_project.git
   cd my_pose_project
   ```

2.  **Create and activate a conda environment**:
    ```bash
    conda create -n venv python=3.10n
    conda activate venv
    ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) MPOSE Setup**  
   - By default, `mpose` uses OpenPose internally. If needed, consult the [MPOSE](https://github.com/PIC4SeR/MPOSE2021_Dataset) repository. 

---

## Usage

### 1. Data Loading
We rely on [MPOSE](https://github.com/PIC4SeR/MPOSE2021_Dataset) to download/open MPOSE data automatically.  
- `data/dataset_mpose.py` initializes the MPOSE dataset.

### 2. Prepare Graph Data
- `data/prepare_data.py` converts each sample’s keypoints and labels into `torch_geometric.data.Data` objects, attaching the correct `edge_index` from `data/edge_index.py`.

### 3. Train the Model
- Run with command-line arguments:
  ```bash
  python main.py --epochs 200 --batch_size 64 --learning_rate 0.0005 
  ```
  - **Arguments**:
    - `--epochs`: Number of training epochs (default: 100).
    - `--batch_size`: Batch size for training and validation (default: 32).
    - `--learning_rate`: Learning rate for the optimizer (default: 0.001).

### 5. Visualize Results
- After training completes, `main.py` calls:
  - `plot_training_curves(...)` to produce **loss** and **accuracy** plots.  
  - `compute_and_plot_confusion_matrix(...)` to display a **confusion matrix** for the best model.

---

## Model Architecture

1.  **Graph Convolutional Network**:
    -   **Layers**: Multiple GCNConv layers.
    -   **Pooling**: Uses `global_mean_pool`.

2.  **LSTM**:
    -   **Stacked**: Two LSTM modules in parallel.
    -   Extracts and concatenates the last hidden states of all LSTMs.

3.  **Classification**:
    -   **Fully Connected**: Multiple fully connected layers.

## References

1. **MPOSE**: [https://github.com/lcjmmp/mpose](https://github.com/lcjmmp/mpose)  
2. **OpenPose**: [https://github.com/CMU-Perceptual-Computing-Lab/openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
3. **AlphaPose**: [https://github.com/MVIG-SJTU/AlphaPose](https://github.com/MVIG-SJTU/AlphaPose)
4. Si, Chenyang, et al. "An attention enhanced graph convolutional lstm network for skeleton-based action recognition." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2019.
5. Sofianos, Theodoros, et al. "Space-time-separable graph convolutional network for pose forecasting." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.  
6. Zou, Zhiming, and Wei Tang. "Modulated graph convolutional network for 3D human pose estimation." Proceedings of the IEEE/CVF international conference on computer vision. 2021.
