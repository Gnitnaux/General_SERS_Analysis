from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from .estimators import EstimatorNode


class KMeansClusterer(EstimatorNode):
    output_kind = "cluster_labels"

    def fit(self, X, y=None):
        a = self._remember(X); p = dict(self.parameters or {}); p.setdefault("random_state", 0); p.setdefault("n_init", 10)
        self.model_ = KMeans(**p).fit(a); self.labels_ = self.model_.labels_.copy(); self.is_fitted_ = True
        return self

    def predict(self, X):
        a = self._input(X); return self.model_.predict(a)

    def transform(self, X):
        a = self._input(X); return self.model_.transform(a)  # distance to each cluster center


class TSNEEmbedding(EstimatorNode):
    """Training-set visualization only; sklearn t-SNE has no out-of-sample transform."""
    output_kind = "embedding"

    def fit(self, X, y=None):
        a = self._remember(X); p = dict(self.parameters or {}); p.setdefault("random_state", 0)
        self.model_ = TSNE(**p); self.embedding_ = self.model_.fit_transform(a); self.is_fitted_ = True
        return self

    def fit_transform(self, X, y=None):
        return self.fit(X, y).embedding_.copy()

    def transform(self, X):
        raise NotImplementedError("t-SNE does not support transforming new samples. Use fit_transform for visualization only.")

    def predict(self, X):
        raise NotImplementedError("t-SNE is an embedding method, not a classifier or cluster-label predictor.")
