import torch

def get_edge_index():
    """
    Returns a Torch tensor of edges (source, target) specifying the
    body-part connections for our pose graph.
    """
    edge_index_list = [
        (0, 1),  # Nose -> Right Shoulder
        (0, 4),  # Nose -> Left Shoulder
        (1, 2),  # Right Shoulder -> Right Elbow
        (2, 3),  # Right Elbow -> Right Wrist
        (4, 5),  # Left Shoulder -> Left Elbow
        (5, 6),  # Left Elbow -> Left Wrist
        (1, 7),  # Right Shoulder -> Right Hip
        (7, 8),  # Right Hip -> Right Knee
        (8, 9),  # Right Knee -> Right Ankle
        (7, 10), # Right Hip -> Left Hip
        (4, 10), # Left Shoulder -> Left Hip
        (10, 11),# Left Hip -> Left Knee
        (11, 12),# Left Knee -> Left Ankle
        (0, 13), # Nose -> Right Eye
        (0, 14), # Nose -> Left Eye
        (13, 15),# Right Eye -> Right Ear
        (14, 16),# Left Eye -> Left Ear
        (12, 17),# Left Ankle -> Left Heel
        (17, 18),# Left Heel -> Left Big Toe
        (12, 19),# Left Big Toe -> Left Small Toe
        (9, 20), # Right Ankle -> Right Heel
        (20, 21),# Right Heel -> Right Big Toe
        (9, 22), # Right Big Toe -> Right Small Toe
    ]
    return torch.tensor(edge_index_list, dtype=torch.long).t()
