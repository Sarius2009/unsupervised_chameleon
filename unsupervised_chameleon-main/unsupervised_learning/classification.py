import numpy as np
from sklearn.cluster import KMeans

class BaseClassifier:
    def __init__(self):
        self.fit_per_trace = False
    def fit(self, X_batch: np.ndarray) -> np.ndarray:
        return self
    def predict(self, X_batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
class AutoencoderAnomalyClassifier(BaseClassifier):
    """
    Unsupervised anomaly detection classifier using KMeans on AE features.
    Produces either:
        - predict(): class labels 0/1
        - predict_proba(): soft scores (2-class)
    """

    def __init__(self, n_clusters=2):
        super().__init__()
        self.km = KMeans(n_clusters=n_clusters, n_init=10)
        self.fit_per_trace = False  # global fit
        self._fitted = False
        self.anomaly_cluster = None

    def fit(self, X: np.ndarray):
        """
        X: (num_windows, feature_dim)
        """
        if X.shape[0] == 0:
            return

        self.km.fit(X)
        self._fitted = True

        # Determine which cluster corresponds to anomalies = higher reconstruction error
        # X[:,0] = reconstruction error
        cluster_errors = []
        labels = self.km.labels_
        for c in range(self.km.n_clusters):
            avg_err = np.mean(X[labels == c, 0])
            cluster_errors.append(avg_err)

        # anomaly cluster = cluster with highest avg. error
        self.anomaly_cluster = int(np.argmax(cluster_errors))

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Classifier not fitted yet!")

        raw_labels = self.km.predict(X)
        labels = np.zeros_like(raw_labels)
        labels[raw_labels == self.anomaly_cluster] = 1

        # return one-hot 2-class representation
        return np.eye(2)[labels]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Classifier not fitted yet!")

        raw_labels = self.km.predict(X)

        probs = np.zeros((len(raw_labels), 2))
        for i, c in enumerate(raw_labels):
            if c == self.anomaly_cluster:
                probs[i] = np.array([0.1, 0.9])  # high anomaly prob
            else:
                probs[i] = np.array([0.9, 0.1])

        return probs
