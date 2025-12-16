import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


import numpy as np
import matplotlib.pyplot as plt


class BaseClassifier:
    def __init__(self):
        self.fit_per_trace = False
        self.pipeline_ = Pipeline([
            ("selector", VarianceThreshold(threshold=1e-4)),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2)),
        ])
    def fit(self, X_batch: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self
    def predict(self, X_batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


def plot_after_pipeline(Xp, y=None, max_points=200_000):
    """
    Plot features after pipeline in 1D / 2D / 3D.
    Assumes pipeline ends in PCA with n_components <= 3.
    """

    d = Xp.shape[1]
    if d < 1 or d > 3:
        return

    # subsample for speed / readability
    if Xp.shape[0] > max_points:
        idx = np.random.choice(Xp.shape[0], max_points, replace=False)
        Xp = Xp[idx]
        y = None if y is None else y[idx]

    if d == 1:
        plt.figure()
        if y is None:
            plt.hist(Xp[:, 0], bins=200)
        else:
            plt.hist(Xp[y == 1, 0], bins=200, alpha=0.5, label="class 1", color='lightblue')
            plt.hist(Xp[y == 0, 0], bins=200, alpha=0.5, label="class 0", color='red')
            plt.legend()
        plt.xlabel("PC1")

    elif d == 2:
        plt.figure()
        if y is None:
            plt.scatter(Xp[:, 0], Xp[:, 1], s=1)
        else:
            plt.scatter(Xp[y == 1, 0], Xp[y == 1, 1], s=1, label="class 1", alpha=1, color='lightblue')
            plt.scatter(Xp[y == 0, 0], Xp[y == 0, 1], s=1, label="class 0", alpha=0.1, c='red')

            plt.legend()
        plt.xlabel("PC1")
        plt.ylabel("PC2")

    else:  # d == 3
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        if y is None:
            ax.scatter(Xp[:, 0], Xp[:, 1], Xp[:, 2], s=1)
        else:
            ax.scatter(Xp[y == 1, 0], Xp[y == 1, 1], Xp[y == 1, 2], s=1, label="class 1", alpha=0.02, color='lightblue')
            ax.scatter(Xp[y == 0, 0], Xp[y == 0, 1], Xp[y == 0, 2], s=1, label="class 0", alpha=1, c='red')
            ax.legend()
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

    plt.tight_layout()
    plt.show()


class RadialThresholdClassifier(BaseClassifier):
    """
    Radial (norm-based) classifier in post-pipeline feature space:

        r = ||Z[:, :radius_dims]||_2

    Decision:
        if r <= r0 -> inner_class
        else       -> outer_class

    r0 is chosen using ground-truth labels on fit() by maximizing a metric.
    """

    def __init__(self,
                 inner_class: int = 0,
                 outer_class: int = 1,
                 radius_dims: int | None = None,      # None = use all dims
                 metric: str = "balanced_accuracy", # "f1" | "balanced_accuracy" | "youden"
                 positive_label: int = 0,           # used only when metric="f1"
                 prob_sharpness: float = 8.0):
        super().__init__()
        self.inner_class = int(inner_class)
        self.outer_class = int(outer_class)
        self.radius_dims = radius_dims
        self.metric = metric
        self.positive_label = int(positive_label)
        self.prob_sharpness = float(prob_sharpness)

        self.r0_ = None
        self._used_dims_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        orig_features = X.shape[1]
        Z = self.pipeline_.fit_transform(X)
        print(f"Feature dimension after pipeline: {Z.shape[1]}/{orig_features}")
        plot_after_pipeline(Z, y)

        r = self._radius(Z)
        y = np.asarray(y, dtype=int)

        self.r0_ = self._choose_threshold(r, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        Z = self.pipeline_.transform(X)
        r = self._radius(Z)

        y_pred = np.where(r <= self.r0_, self.inner_class, self.outer_class)
        return y_pred.astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns (N,2) probabilities in class order [0,1].
        Uses a smooth logistic around r0.
        """
        self._check_is_fitted()
        Z = self.pipeline_.transform(X)
        r = self._radius(Z)

        s = self.prob_sharpness
        p_inner = 1.0 / (1.0 + np.exp(-s * (self.r0_ - r)))

        proba = np.zeros((r.shape[0], 2), dtype=float)
        if self.inner_class == 0:
            proba[:, 0] = p_inner
            proba[:, 1] = 1.0 - p_inner
        else:
            proba[:, 1] = p_inner
            proba[:, 0] = 1.0 - p_inner
        return proba

    # ---------------- internals ----------------

    def _radius(self, Z: np.ndarray) -> np.ndarray:
        """
        Compute L2 norm using the first `radius_dims` columns (or all if None).
        """
        d = Z.shape[1]
        if d < 1:
            raise ValueError("Pipeline output has < 1 dimension.")

        if self.radius_dims is None:
            use_d = d
        else:
            use_d = int(self.radius_dims)
            if use_d < 1:
                raise ValueError(f"radius_dims must be >= 1 or None, got {self.radius_dims}")
            use_d = min(use_d, d)

        self._used_dims_ = use_d
        Zuse = Z[:, :use_d]
        return np.linalg.norm(Zuse, axis=1)

    def _check_is_fitted(self):
        if self.r0_ is None:
            raise AttributeError("Model is not fitted yet. Call fit(X, y) first.")

    def _choose_threshold(self, r: np.ndarray, y: np.ndarray) -> float:
        r = np.asarray(r, dtype=float)
        y = np.asarray(y, dtype=int)

        # Candidate thresholds = quantiles of r
        n_candidates = 512  # tune: 256, 512, 1024, 2048
        qs = np.linspace(0.0, 1.0, n_candidates)
        candidates = np.quantile(r, qs)

        # ensure strictly increasing candidates (quantiles can repeat on discrete r)
        candidates = np.unique(candidates)
        if candidates.size == 1:
            t = float(candidates[0])
            print(f"Chosen radius r0={t:.6f} (degenerate; all radii equal)")
            return t

        best_t = float(candidates[0])
        best_score = -np.inf

        # Evaluate all candidates
        for t in candidates:
            y_pred = np.where(r <= t, self.inner_class, self.outer_class)
            score = self._score_from_preds(y, y_pred)
            if score > best_score:
                best_score = score
                best_t = float(t)

        print(f"Chosen radius r0={best_t:.6f} using dims={self._used_dims_} with {self.metric}={best_score:.6f}")
        return best_t


        print(f"Chosen radius r0={best_t:.6f} using dims={self._used_dims_} with {self.metric}={best_score:.6f}")
        return best_t

    def _score_from_preds(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        TP = np.count_nonzero((y_true == 1) & (y_pred == 1))
        FP = np.count_nonzero((y_true == 0) & (y_pred == 1))
        TN = np.count_nonzero((y_true == 0) & (y_pred == 0))
        FN = np.count_nonzero((y_true == 1) & (y_pred == 0))

        if self.metric == "balanced_accuracy":
            tpr = TP / (TP + FN) if (TP + FN) else 0.0
            tnr = TN / (TN + FP) if (TN + FP) else 0.0
            return 0.5 * (tpr + tnr)

        if self.metric == "youden":
            tpr = TP / (TP + FN) if (TP + FN) else 0.0
            tnr = TN / (TN + FP) if (TN + FP) else 0.0
            return (tpr + tnr - 1.0)

        # metric == "f1"
        pos = self.positive_label
        if pos == 1:
            precision = TP / (TP + FP) if (TP + FP) else 0.0
            recall    = TP / (TP + FN) if (TP + FN) else 0.0
        else:
            # treat class 0 as positive
            TP0, FP0, FN0 = TN, FN, FP
            precision = TP0 / (TP0 + FP0) if (TP0 + FP0) else 0.0
            recall    = TP0 / (TP0 + FN0) if (TP0 + FN0) else 0.0

        return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0



class KMeansClassifier(BaseClassifier):
    """
    Unsupervised 2-class classifier using K-Means, with class relabeling by cluster size.

    Fit:
      - Run KMeans(n_clusters=2) on X.
      - Measure cluster sizes and remap cluster indices so:
          largest  -> class 1
          second   -> class 0
      - Stores centroids for prediction.

    Predict:
      - Uses KMeans' nearest-centroid assignment, then applies the size-based remap.

    Probabilities:
      - Converts distances to centroids into probabilities via softmax over (-distance / temperature).
      - Returns a (N, 2) array with columns strictly in class order [0, 1].

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
    centroids_ : np.ndarray, shape (2, D)
    cluster_to_class_ : dict[int,int]
        Mapping from KMeans cluster index -> class label {0,1}.
    classes_ : np.ndarray([0,1])
    """

    def __init__(self,
                 n_clusters = 2,
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
        self.classes_ = np.array([0, 1], dtype=int)

    # ---- BaseUnsupervisedClassifier API ----
    def fit(self, X: np.ndarray, y: np.ndarray):
        orig_features = X.shape[1]
        X = self.pipeline_.fit_transform(X)
        print(f"Feature dimension after pipeline: {X.shape[1]}/{orig_features}")
        plot_after_pipeline(X, y)

        self.kmeans_.fit(X)
        self._assign_clusters_from_labels(X, y)

        self.centroids_ = np.asarray(self.kmeans_.cluster_centers_, dtype=float)
        # TODO remove
        cluster_labels = self.kmeans_.labels_
        for k in range(self.kmeans_.n_clusters):
            m = cluster_labels == k
            if m.any():
                print(k, "size", m.sum(), "p(class0)", np.mean(y[m] == 0), "p(class1)", np.mean(y[m] == 1))

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
        Columns are ordered [0, 1] regardless of internal KMeans labels.
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

        # Aggregate cluster probabilities into 2 class columns [0,1]
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
        Margin-like scores: probs minus uniform prior (1/) for each class.
        """
        p = self.predict_proba(X)
        return p - (1.0 / 2.0)

    # ---- Utilities ----
    def _check_is_fitted(self):
        if self.kmeans_ is None or self.centroids_ is None or self.cluster_to_class_ is None:
            raise AttributeError("Model is not fitted yet. Call fit(X) first.")


    def _assign_clusters_from_labels(self, X: np.ndarray, y_true: np.ndarray):
        """
        After self.kmeans_.fit(X), use ground-truth labels y_true (0/1)
        to derive cluster_to_class_ via majority vote.
        X is only used to recompute cluster assignments if needed.
        """
        cluster_labels = self.kmeans_.predict(X)
        self.cluster_to_class_ = self._majority_vote_cluster_mapping(cluster_labels, y_true)

    @staticmethod
    def _majority_vote_cluster_mapping(
            cluster_labels: np.ndarray,
            y_true: np.ndarray,
            default_class: int = 1,
            class0_ratio_threshold: float = 0.1,
    ) -> dict[int, int]:
        """
        For each cluster:
          - Assign class 0 if (count(class0) / cluster_size) >= class0_ratio_threshold
          - Otherwise assign class 1

        Binary setting: y_true in {0,1}.
        """

        mapping: dict[int, int] = {}

        K = int(cluster_labels.max()) + 1
        for k in range(K):
            mask = (cluster_labels == k)
            if not mask.any():
                mapping[k] = default_class
                continue

            yk = y_true[mask]

            n = yk.size
            n0 = np.sum(yk == 0)

            ratio0 = n0 / n

            if ratio0 >= class0_ratio_threshold:
                mapping[k] = 0
            else:
                mapping[k] = 1

        return mapping


