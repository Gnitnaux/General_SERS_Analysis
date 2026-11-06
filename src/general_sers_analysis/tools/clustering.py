from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from .estimators import EstimatorNode


class KMeansClusterer(EstimatorNode):
    """K-means assignments and distances; cluster IDs do not imply chemical identity.

    Attributes
    ----------
    is_fitted_: Whether fitting completed successfully; typically created or set by fit.
    labels_: Training K-means cluster assignments; not chemical ground truth.
    model_: Fitted third-party estimator, projection or scaler; usually absent before fit.
    output_kind: Declared output category.

    Inherited interface
    -------------------
    See EstimatorNode for methods and attributes not overridden here.
    """
    output_kind = "cluster_labels"

    def fit(self, X, y=None):
        """Fit KMeans on training data and store training assignments in labels_.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.
        y: Ground-truth targets aligned with rows, usually (n_rows, n_targets); never predictions.

        Returns and state
        -----------------
        self. Learned attributes are stored on the instance; see the class attribute descriptions.
        """
        a = self._remember(X); p = dict(self.parameters or {}); p.setdefault("random_state", 0); p.setdefault("n_init", 10)
        self.model_ = KMeans(**p).fit(a); self.labels_ = self.model_.labels_.copy(); self.is_fitted_ = True
        return self

    def predict(self, X):
        """Return nearest fitted cluster IDs, without chemical-label interpretation.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Return nearest fitted cluster IDs, without chemical-label interpretation.
        """
        a = self._input(X); return self.model_.predict(a)

    def transform(self, X):
        """Return distances from each input sample to each fitted cluster center.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Return distances from each input sample to each fitted cluster center.
        """
        a = self._input(X); return self.model_.transform(a)  # distance to each cluster center


class TSNEEmbedding(EstimatorNode):
    """Training-only t-SNE visualization; no new-sample transform or label prediction.

    Attributes
    ----------
    embedding_: Training embedding with shape (n_training_samples, n_embedding_dimensions).
    is_fitted_: Whether fitting completed successfully; typically created or set by fit.
    model_: Fitted third-party estimator, projection or scaler; usually absent before fit.
    output_kind: Declared output category.

    Inherited interface
    -------------------
    See EstimatorNode for methods and attributes not overridden here.
    """
    output_kind = "embedding"

    def fit(self, X, y=None):
        """Fit t-SNE on the training matrix and store embedding_; no new-sample mapping is learned.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.
        y: Ground-truth targets aligned with rows, usually (n_rows, n_targets); never predictions.

        Returns and state
        -----------------
        self. Learned attributes are stored on the instance; see the class attribute descriptions.
        """
        a = self._remember(X); p = dict(self.parameters or {}); p.setdefault("random_state", 0)
        self.model_ = TSNE(**p); self.embedding_ = self.model_.fit_transform(a); self.is_fitted_ = True
        return self

    def fit_transform(self, X, y=None):
        """Fit t-SNE and return a copy of the training embedding.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.
        y: Ground-truth targets aligned with rows, usually (n_rows, n_targets); never predictions.

        Returns and state
        -----------------
        Fit t-SNE and return a copy of the training embedding.
        """
        return self.fit(X, y).embedding_.copy()

    def transform(self, X):
        """Always raise NotImplementedError; t-SNE has no supported out-of-sample transform.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Always raise NotImplementedError; t-SNE has no supported out-of-sample transform.

        Raises
        ------
        NotImplementedError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        raise NotImplementedError("t-SNE does not support transforming new samples. Use fit_transform for visualization only.")

    def predict(self, X):
        """Always raise NotImplementedError; t-SNE does not predict class or cluster labels.

        Parameters
        ----------
        X: Current input data, usually rows=samples and columns=features. Workflow additionally accepts DataBatch.

        Returns and state
        -----------------
        Always raise NotImplementedError; t-SNE does not predict class or cluster labels.

        Raises
        ------
        NotImplementedError when the input, fitted-state or supported-interface conditions checked below are violated.
        """
        raise NotImplementedError("t-SNE is an embedding method, not a classifier or cluster-label predictor.")
