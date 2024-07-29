def binarize_targets(y, threshold=0.5):
    return (y > threshold).astype(int)
