import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.pipeline import Pipeline
from umap import UMAP

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


class BaseClassifier:
    def __init__(self):
        self.fit_per_trace = False
        self.pipeline_ = Pipeline([
            ("selector", VarianceThreshold(threshold=1e-4)),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95)),
            #("umap", UMAP(n_components=10, n_neighbors=15, min_dist=0.3, verbose=True)),
            #("ica", FastICA(n_components=30))
        ])
    def fit(self, X_batch: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self
    def predict(self, X_batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


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
                 n_clusters = 3,
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
        self.kmeans_ = KMeans(
            n_clusters=n_clusters,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            algorithm=self.algorithm,
            copy_x=False,
        )
        self.centroids_ = None
        self.cluster_to_class_ = None
        self.classes_ = np.array([0, 1, 2], dtype=int)

    # ---- BaseUnsupervisedClassifier API ----
    def fit(self, X: np.ndarray, y: np.ndarray):
        orig_features = X.shape[1]
        X = self.pipeline_.fit_transform(X)
        print(f"Feature dimension after pipeline: {X.shape[1]}/{orig_features}")

        self.kmeans_.fit(X)
        self._assign_clusters_from_labels(X, y)

        self.centroids_ = np.asarray(self.kmeans_.cluster_centers_, dtype=float)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = self.pipeline_.transform(X)
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
        X = self.pipeline_.transform(X)

        # Distances to K centroids
        d = pairwise_distances(X, self.centroids_, metric="euclidean")  # (N, K)

        t = max(self.temperature, 1e-8)
        logits = -d / t
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        p_cluster = exp / exp.sum(axis=1, keepdims=True)  # (N, K)

        # Aggregate cluster probabilities into 3 class columns [0,1,2]
        n_samples = X.shape[0]
        n_classes = len(self.classes_)  # should be 3
        proba = np.zeros((n_samples, n_classes), dtype=float)

        for kmeans_idx, cls in self.cluster_to_class_.items():
            proba[:, cls] += p_cluster[:, kmeans_idx]

        # (optional) renormalize in case of numerical drift
        proba_sum = proba.sum(axis=1, keepdims=True)
        proba_sum[proba_sum == 0.0] = 1.0
        proba /= proba_sum

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


    def _assign_clusters_from_labels(self, X: np.ndarray, y_true: np.ndarray):
        """
        After self.kmeans_.fit(X), use ground-truth labels y_true (0/1/2)
        to derive cluster_to_class_ via majority vote.
        X is only used to recompute cluster assignments if needed.
        """
        cluster_labels = self.kmeans_.predict(X)
        self.cluster_to_class_ = _majority_vote_cluster_mapping(cluster_labels, y_true, classes=self.classes_)


def _majority_vote_cluster_mapping(
        cluster_labels: np.ndarray,
        y_true: np.ndarray,
        classes=(0, 1, 2),
) -> dict[int, int]:
    """
    For each cluster, assign the class that is most frequent among its points.

    Parameters
    ----------
    cluster_labels : (N,) int array
        KMeans cluster index per point.
    y_true : (N,) int array
        Ground truth window labels in {0,1,2}.
    classes : iterable
        Allowed class labels (default {0,1,2}).

    Returns
    -------
    mapping : dict[int,int]
        cluster_idx -> assigned_class
    """
    classes = np.array(classes, dtype=int)
    mapping: dict[int, int] = {}

    K = int(cluster_labels.max()) + 1
    for k in range(K):
        mask = (cluster_labels == k)
        if not mask.any():
            # If a cluster ended up empty, assign a default (e.g. class 2)
            mapping[k] = 2
            continue

        vals, counts = np.unique(y_true[mask], return_counts=True)
        # Restrict to known classes if needed
        valid = np.isin(vals, classes)
        vals = vals[valid]
        counts = counts[valid]
        if vals.size == 0:
            mapping[k] = 2
        else:
            mapping[k] = int(vals[counts.argmax()])

    return mapping


