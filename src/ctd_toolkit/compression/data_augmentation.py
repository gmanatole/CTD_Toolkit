import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import CubicSpline

def _smooth_curve(length: int, n_knots: int, scale: float, 
                  center: float = 1.0) -> np.ndarray:
    """
    Generate a smooth random curve by interpolating a few random anchor points with a cubic spline.
    center : value around which anchors are scattered (1.0 for warping, 0.0 for shifting)
    """
    knot_x = np.linspace(0, length - 1, n_knots)
    knot_y = center + np.random.uniform(-scale, scale, size=n_knots)
    cs = CubicSpline(knot_x, knot_y)
    return cs(np.arange(length))


def magnitude_warp(profile: np.ndarray,
                   n_knots: int = 4,
                   scale: float = 0.1) -> np.ndarray:
    """
    Multiply the profile by a smooth curve centered around 1.0. Preserves zero-crossings and shape topology.
    scale : max deviation from 1.0
    """
    curve = _smooth_curve(len(profile), n_knots, scale, center=1.0)
    return profile * curve


def baseline_shift(profile: np.ndarray,
                   n_knots: int = 4,
                   scale: float = 0.1) -> np.ndarray:
    """
    Add a smooth curve centered around 0.0 to the profile. Simulates sensor drift or slow ambient temperature variation.
    scale : max deviation (for normalized profiles)
    """
    curve = _smooth_curve(len(profile), n_knots, scale, center=0.0)
    return profile + curve


def build_pca_index(profiles: np.ndarray, depth = 200, n_components: int = 15, k: int = 10):
    """
    profiles : (N, 1000) — your padded profiles
    Returns the fitted PCA, the k-NN model, and the embeddings.
    """
    pca = PCA(n_components=n_components, random_state=42)
    embeddings = pca.fit_transform(profiles[:, :depth])

    knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="auto")
    knn.fit(embeddings)
    return pca, knn, embeddings

def mixup(idx: int,
                knn: NearestNeighbors,
                embeddings: np.ndarray,
                batch_size: int = 64,
                alpha: float = 0.4) -> np.ndarray:
    """
    Sample synthetic profiles by mixing each profile
    with one of its k nearest neighbours.
    alpha : Beta distribution parameter — lower = more extreme mixing
    """
    # For each sampled profile, pick a random neighbour
    _, neighbor_indices = knn.kneighbors(embeddings[idx])
    idx_j = neighbor_indices[np.random.randint(1, neighbor_indices.shape[1])]
    
    lam = np.random.beta(alpha, alpha, size=(batch_size, 1))
    return lam * profiles[idx_i] + (1 - lam) * profiles[idx_j]


def refresh_latent_index(encoder,
                         profiles: np.ndarray,
                         k: int = 10,
                         device: str = "cpu"):
    """
    Rebuild the k-NN index using the encoder's current latent vectors.
    Drop-in replacement for the PCA-based index after a few epochs.
    """
    encoder.eval()
    with torch.no_grad():
        t = torch.tensor(profiles, dtype=torch.float32).to(device)
        latents = encoder(t).cpu().numpy()           # (N, 10)

    knn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    knn.fit(latents)
    return knn, latents











