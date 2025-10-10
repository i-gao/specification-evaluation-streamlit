import numpy as np
from scipy.optimize import linear_sum_assignment
import PIL
import clip
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def compute_similarity_matrix(A, B, sim_fn: callable) -> np.ndarray:
    """
    Compute the pairwise similarity matrix between two sets,
    given a similarity function.
    """
    return np.array([[sim_fn(a, b) for b in B] for a in A])


def soft_jaccard(A, B, sim_fn: callable) -> float:
    """
    Compute the soft Jaccard similarity between two sets,
    given their pairwise similarity matrix.

    Given sets A, B
    1. Compute pairwise similarity matrix containing s(a, b) in [0, 1]
    2. Compute the optimal assignment of elements in A to elements in B; call this M*
    3. Compute the soft Jaccard similarity as the sum of the similarities of the assigned elements:
        let W* := sum_{i, j in M*} s(a_i, b_j)
        soft Jaccard := W* / (|A| + |B| - W*)
    Reduces to Jaccard similarity when s(a, b) = 1 if a == b, 0 otherwise.

    Parameters
    ----------
    sim_matrix : np.ndarray
        A matrix of shape (len(A), len(B)), where entry [i, j] is
        the similarity between element i in A and element j in B.
        Values must be in [0, 1].

    Returns
    -------
    float
        Soft Jaccard similarity in [0, 1].
        The optimal matching is returned as a list of tuples (i, j)
        where i is an index in A and j is an index in B.
        The similarity matrix.
    """
    sim_matrix = compute_similarity_matrix(A, B, sim_fn)

    if sim_matrix.size == 0:
        # both sets empty → similarity = 1
        return 1.0, [] if sim_matrix.shape == (0, 0) else 0, []

    m, n = sim_matrix.shape
    # Hungarian algorithm solves a *minimization* problem, so convert
    cost = -sim_matrix

    # Pad to square matrix if needed
    size = max(m, n)
    padded = np.zeros((size, size))
    padded[:m, :n] = cost

    row_ind, col_ind = linear_sum_assignment(padded)
    # Only count assignments within the original matrix
    valid = [(i, j) for i, j in zip(row_ind, col_ind) if i < m and j < n]

    W_star = sim_matrix[[i for i, _ in valid], [j for _, j in valid]].sum()

    # soft Jaccard
    return (
        W_star / (m + n - W_star),
        valid,
        sim_matrix, 
    )

def clip_score(img1: PIL.Image, img2: PIL.Image) -> float:
    """
    Compute the CLIP score between two images.
    """
    image1 = preprocess(img1).unsqueeze(0).to(device)
    image2 = preprocess(img2).unsqueeze(0).to(device)

    # Calculate CLIP features
    with torch.no_grad():
        image_features1 = model.encode_image(image1)
        image_features2 = model.encode_image(image2)

    # Normalize features for cosine similarity
    image_features1 /= image_features1.norm(dim=-1, keepdim=True)
    image_features2 /= image_features2.norm(dim=-1, keepdim=True)

    # Calculate cosine similarity
    similarity = (image_features1 @ image_features2.T).item()

    return similarity