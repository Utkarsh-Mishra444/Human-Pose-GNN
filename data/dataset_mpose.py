from mpose import MPOSE

def load_mpose_data():
    """
    Creates an MPOSE2021 dataset object and returns (X_train, y_train, X_val, y_val).
    """
    dataset = MPOSE(pose_extractor='openpose',
                    split=1,
                    preprocess='scale_and_center',
                    velocities=True,
                    remove_zip=False)

    X_train, y_train, X_val, y_val = dataset.get_data()

    return X_train, y_train, X_val, y_val
