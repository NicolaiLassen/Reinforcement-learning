def normalize_dist(t):
    # Normalize  # PLZ DON'T BLOW MY GRADIENT
    return (t - t.mean()) / (t.std() + 1e-10)
