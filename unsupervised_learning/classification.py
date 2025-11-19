import numpy as np
from sklearn.decomposition import PCA, FastICA

class BaseClassifier:
    def __init__(self):
        self.fit_per_trace = False
    def fit(self, X_batch: np.ndarray) -> np.ndarray:
        return self
    def predict(self, X_batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


# ===== Size-ordered 3-class K-Means (largest→1, second→2, smallest→0) =====
# deps: numpy, scikit-learn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


def _prepare_data(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N, D); got shape {X.shape}")

    # 1. Remove low-variance features
    selector = VarianceThreshold(threshold=1e-4)
    X = selector.fit_transform(X)

    # 2. Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # TODO test PCA/ICA
    return X


class KMeansClassifier(BaseClassifier):
    """
    Unsupervised 3-class classifier using K-Means, with class relabeling by cluster size.

    Fit:
      - Run KMeans(n_clusters=3) on X.
      - Measure cluster sizes and remap cluster indices so:
          largest  -> class 1
          second   -> class 2
          smallest -> class 0
      - Stores centroids for prediction.

    Predict:
      - Uses KMeans' nearest-centroid assignment, then applies the size-based remap.

    Probabilities:
      - Converts distances to centroids into probabilities via softmax over (-distance / temperature).
      - Returns a (N, 3) array with columns strictly in class order [0, 1, 2].

    Parameters
    ----------
    n_init : int, default 10
        KMeans restarts.
    max_iter : int, default 300
        Max iterations per run.
    tol : float, default 1e-4
        Convergence tolerance.
    random_state : int | None, default 42
        Reproducibility.
    algorithm : {'lloyd','elkan'}, default 'lloyd'
        KMeans algorithm variant.
    temperature : float, default 1.0
        Temperature for softmax in predict_proba (lower = sharper).

    Attributes after fit
    --------------------
    kmeans_ : sklearn.cluster.KMeans
        Fitted KMeans instance.
    centroids_ : np.ndarray, shape (3, D)
    cluster_to_class_ : dict[int,int]
        Mapping from KMeans cluster index -> class label {0,1,2}.
    classes_ : np.ndarray([0,1,2])
    """

    def __init__(self,
                 n_init: int = 10,
                 max_iter: int = 300,
                 tol: float = 1e-4,
                 random_state: int | None = 42,
                 algorithm: str = "lloyd",
                 temperature: float = 1.0):
        super().__init__()
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.algorithm = algorithm
        self.temperature = float(temperature)

        # Fitted attrs
        self.kmeans_ = None
        self.centroids_ = None
        self.cluster_to_class_ = None
        self.classes_ = np.array([0, 1, 2], dtype=int)

    # ---- BaseUnsupervisedClassifier API ----
    def fit(self, X: np.ndarray):
        X = _prepare_data(X)

        self.kmeans_ = KMeans(
            n_clusters=3,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            algorithm=self.algorithm,
            copy_x=False,
        ).fit(X)

        labels = self.kmeans_.labels_
        unique, counts = np.unique(labels, return_counts=True)
        if unique.size != 3:
            raise ValueError(
                f"KMeans did not yield 3 non-empty clusters (got {unique.size}). "
                "Consider different initialization or parameters."
            )

        # Sort clusters by size (desc) and map to desired class labels
        order = unique[np.argsort(-counts)]
        # largest->1, second->2, smallest->0
        self.cluster_to_class_ = {order[0]: 1, order[1]: 2, order[2]: 0}

        self.centroids_ = np.asarray(self.kmeans_.cluster_centers_, dtype=float)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = _prepare_data(X)
        clusters = self.kmeans_.predict(X)
        # Map cluster indices to size-ordered classes
        to_class = np.vectorize(self.cluster_to_class_.get)
        return to_class(clusters).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Soft probabilities derived from centroid distances.
        Columns are ordered [0, 1, 2] regardless of internal KMeans labels.
        """
        self._check_is_fitted()
        X = _prepare_data(X)
        # Distances to original KMeans cluster indices (columns 0..2 correspond to those)
        d = pairwise_distances(X, self.centroids_, metric="euclidean")  # (N, 3)

        # Convert distances to similarities; softmax over negative distance / temperature
        t = max(self.temperature, 1e-8)
        logits = -d / t
        logits -= logits.max(axis=1, keepdims=True)  # stabilize
        exp = np.exp(logits)
        p_cluster = exp / exp.sum(axis=1, keepdims=True)  # (N, 3) in KMeans-cluster order

        # Reorder probabilities into class columns [0,1,2]
        # Build mapping cluster_idx -> class_idx_position
        class_positions = {0: 0, 1: 1, 2: 2}
        proba = np.zeros_like(p_cluster)
        for kmeans_idx, cls in self.cluster_to_class_.items():
            proba[:, class_positions[cls]] = p_cluster[:, kmeans_idx]
        return proba

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Margin-like scores: probs minus uniform prior (1/3) for each class.
        """
        p = self.predict_proba(X)
        return p - (1.0 / 3.0)

    # ---- Utilities ----
    def _check_is_fitted(self):
        if self.kmeans_ is None or self.centroids_ is None or self.cluster_to_class_ is None:
            raise AttributeError("Model is not fitted yet. Call fit(X) first.")
