import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

def _build_feature_preproc(n_features: int, pca_latent_components = 2, pca_generic_components = 1):
    # Generic block (works well for TSFresh)
    generic = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("var", VarianceThreshold()),  # safe default; drops constant cols
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=pca_generic_components, random_state=0)),
    ])

    # Hack: 33 == [error + 32 latent]
    if n_features == 33:
        err_idx = [0]
        latent_idx = list(range(1, 33))

        err_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),  # optional but recommended
            ("scale", StandardScaler()),
        ])

        latent_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=pca_latent_components, random_state=0)),
        ])

        return ColumnTransformer(
            transformers=[
                ("latent", latent_pipe, latent_idx),
                ("err", err_pipe, err_idx),

            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    # Fallback: TSFresh / other feature sets
    return generic


def _plot_after_pipeline(Xp, y=None, max_points=100_000, decision_radius=None):
    """
    Plot features after pipeline in 1D / 2D / 3D.
    Assumes pipeline ends in PCA with n_components <= 3.
    """
    d = Xp.shape[1]
    if d < 1 or d > 3:
        return
    fig = plt.figure()

    # subsample for speed / readability
    if Xp.shape[0] > max_points:
        idx = np.random.choice(Xp.shape[0], max_points, replace=False)
        Xp = Xp[idx]
        y = None if y is None else y[idx]

    if d == 1:
        if y is None:
            plt.hist(Xp[:, 0], bins=200)
        else:
            x1 = Xp[y == 1, 0]
            x0 = Xp[y == 0, 0]
            plt.hist(x0, bins=200, alpha=0.5, label="class 1", color='lightblue')
            plt.hist(x1, bins=200, alpha=0.5, label="class 0", color='red')
            plt.legend()

            # scaled class 0
            if len(x0) > 0 and len(x1) > 0:
                scale = len(x1) / len(x0)
                weights = np.full_like(x0, scale, dtype=float)

                plt.hist(
                    x0,
                    bins=200,
                    weights=weights,
                    histtype="step",
                    linewidth=2,
                    label="class 0 (scaled)",
                )

            if decision_radius is not None:
                plt.axvline(+decision_radius, color='black')
                plt.axvline(-decision_radius)
        plt.xlabel("PC1")

    elif d == 2:
        plt.axis("equal")
        if y is None:
            plt.scatter(Xp[:, 0], Xp[:, 1], s=1)
        else:
            plt.scatter(Xp[y == 1, 0], Xp[y == 1, 1], s=1, label="class 1", alpha=1, color='lightblue')
            plt.scatter(Xp[y == 0, 0], Xp[y == 0, 1], s=1, label="class 0", alpha=0.1, c='red')

            if decision_radius is not None:
                t = np.linspace(0.0, 2.0 * np.pi, 512)
                x = decision_radius * np.cos(t)
                y_ = decision_radius * np.sin(t)
                plt.plot(x, y_, color='black')

            plt.legend()
        plt.xlabel("PC1")
        plt.ylabel("PC2/Err")

    elif d == 3:  # d == 3
        ax = fig.add_subplot(111, projection="3d")
        if y is None:
            ax.scatter(Xp[:, 0], Xp[:, 1], Xp[:, 2], s=1)
        else:
            ax.scatter(Xp[y == 1, 0], Xp[y == 1, 1], Xp[y == 1, 2], s=1, label="class 1", alpha=0.1, color='lightblue')
            ax.scatter(Xp[y == 0, 0], Xp[y == 0, 1], Xp[y == 0, 2], s=1, label="class 0", alpha=1, c='red')

            if decision_radius is not None:
                u = np.linspace(0.0, 2.0 * np.pi, 64)
                v = np.linspace(0.0, np.pi, 32)
                uu, vv = np.meshgrid(u, v)

                x = decision_radius * np.cos(uu) * np.sin(vv)
                y_ = decision_radius * np.sin(uu) * np.sin(vv)
                z = decision_radius * np.cos(vv)

                ax.plot_wireframe(x, y_, z, color='black')
            ax.legend()
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3/Err")

    plt.tight_layout()
    plt.show()

    return fig



class BaseClassifier:
    def __init__(self):
        self.pipeline_ = None
        self.post_pipeline_data = None
    def fit(self, X_batch: np.ndarray, y: np.ndarray):
        return self
    def predict(self, X_batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


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
                 metric: str = "balanced_accuracy", # "f1" | "balanced_accuracy" | "youden"
                 positive_label: int = 1,           # used only when metric="f1"
                 prob_sharpness: float = 8.0):
        super().__init__()
        self.inner_class = int(inner_class)
        self.outer_class = int(outer_class)
        self.metric = metric
        self.positive_label = int(positive_label)
        self.prob_sharpness = float(prob_sharpness)

        self.r0_ = None
        self._used_dims_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        orig_features = X.shape[1]
        self.pipeline_ = _build_feature_preproc(orig_features)
        Z = self.pipeline_.fit_transform(X)
        print(f"Feature dimension after pipeline: {Z.shape[1]}/{orig_features}")

        y = np.asarray(y, dtype=int)
        if orig_features == 33:
            self.r0_ =self.choose_radius_from_top_error(Z[:, :-1], Z[:, -1])
        else:
            r = self._radius(Z)
            self.r0_ = self._choose_threshold(r, y)
        _plot_after_pipeline(Z, y, decision_radius=self.r0_)
        self.post_pipeline_data = (Z, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        Z = self.pipeline_.transform(X)
        if X.shape[1] == 33:
            r = self._radius(Z[:, :-1])
            err = Z[:, -1]
            y_pred = np.where(r <= self.r0_, self.inner_class, self.outer_class)
        else:
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
        if X.shape[1] == 33:
            r = self._radius(Z[:, :-1])
            err = Z[:, -1]
        else:
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

        self._used_dims_ = d
        Zuse = Z[:, :d]
        return np.linalg.norm(Zuse, axis=1)

    def _check_is_fitted(self):
        if self.r0_ is None:
            raise AttributeError("Model is not fitted yet. Call fit(X, y) first.")

    @staticmethod
    def choose_radius_from_top_error(
            Z: np.ndarray,
            err: np.ndarray,
            top_err_frac: float = 0.10,
            in_radius_frac: float = 0.95,
            eps: float = 1e-12,
    ) -> float:
        """
        Pick a radial threshold r0 using the highest-error windows.

        Procedure:
          1) Select the windows in the top `top_err_frac` by reconstruction error.
          2) Compute radii r = ||Z||_2 for those windows (same dims as Z).
          3) Choose r0 so that `in_radius_frac` of those radii are <= r0.

        Returns
        -------
        r0 : float
            The decision radius.
        """
        Z = np.asarray(Z)
        err = np.asarray(err).reshape(-1)
        if Z.shape[0] != err.shape[0]:
            raise ValueError(f"Z and err must have same length: {Z.shape[0]} vs {err.shape[0]}")

        n = err.shape[0]
        k = int(np.ceil(n * top_err_frac))
        k = max(k, 1)

        # indices of top-k errors (fast, no full sort)
        idx_top = np.argpartition(err, n - k)[n - k:]
        Z_top = Z[idx_top]

        # radii in the same feature space as Z
        r = np.linalg.norm(Z_top, axis=1)

        # radius containing `in_radius_frac` of top-error windows
        r0 = float(np.quantile(r, in_radius_frac))

        # avoid degenerate zero radius
        return max(r0, eps)

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
        self.pipeline_ = _build_feature_preproc(orig_features)
        X = self.pipeline_.fit_transform(X)
        print(f"Feature dimension after pipeline: {X.shape[1]}/{orig_features}")
        _plot_after_pipeline(X, y)
        self.post_pipeline_data = (X, y)

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
        n_classes = len(self.classes_)  # should be 2
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
