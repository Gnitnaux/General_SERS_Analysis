import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from general_sers_analysis import Spectra, Edge, Condition, Node
from general_sers_analysis.tools.preprocessing import (
    SGFilter, AirPLS, Interpolate, SpectralCut, PeakNormalize, MinMaxNormalize, Standardize)
from general_sers_analysis.tools.classification import (
    RFClassifier, SVMClassifier, XGBoostClassifier, LogisticClassifier, PLSDAClassifier)
from general_sers_analysis.tools.regression import (
    PLSRRegressor, RFRegressor, SVRRegressor, XGBoostRegressor, LinearRegressor, DigitalSERSRegressor)
from general_sers_analysis.tools.clustering import KMeansClusterer, TSNEEmbedding
from general_sers_analysis.tools.unmixing import CLSUnmixer, NNLSUnmixer
from general_sers_analysis.tools.aggregation import (
    PCAFeatures, FactorAnalysisFeatures, PeakExtractor, PeakDetector, MeanAggregator,
    MedianAggregator, MaxAggregator, MinAggregator, SpectralDifference)


def test_spectral_axis_and_fixed_preprocessing():
    x = np.arange(21.0); a = Spectra(np.vstack([x*x, x*x+4]), x, ("a", "b"))
    np.testing.assert_allclose(SGFilter(5, 2).transform(a).intensity, a.intensity, atol=1e-10)
    cropped = SpectralCut(4, 10).transform(a)
    assert cropped.sample_ids == ("a", "b") and cropped.intensity.shape == (2, 7)
    q = Interpolate(np.arange(4, 10, .5)).transform(cropped)
    assert q.intensity.shape == (2, 12)
    with pytest.raises(ValueError): Interpolate([0, 20]).transform(cropped)
    with pytest.raises(ValueError): Spectra([[1, 2]], [2, 1])
    with pytest.raises(ValueError): SGFilter(4).transform(a)


def test_training_statistics_are_not_refitted_on_test_data():
    scaler = Standardize()
    with pytest.raises(NotFittedError): scaler.transform([[1, 2]])
    scaler.fit([[0, 10], [2, 12]])
    out = scaler.transform([[101, 111]])
    np.testing.assert_allclose(out, [[100, 100]])
    np.testing.assert_allclose(scaler.model_.mean_, [1, 11])
    normalized = MinMaxNormalize().fit_transform([[2, 4, 6], [10, 20, 30]])
    np.testing.assert_allclose(normalized, [[0, .5, 1], [0, .5, 1]])
    np.testing.assert_array_equal(PeakNormalize().transform([[0, 0], [-2, 1]]), [[0, 0], [-1, .5]])


def test_airpls_retains_peak_and_reduces_smooth_background():
    x = np.arange(201.0); background = 10 + .03*x
    peak = 25*np.exp(-.5*((x-100)/5)**2)
    result = AirPLS(lam=1e5).transform(Spectra([background+peak], x)).intensity[0]
    assert abs(np.median(result[:30])) < 1 and result[100] > 20


@pytest.mark.parametrize("tool", [RFClassifier, SVMClassifier, XGBoostClassifier, LogisticClassifier, PLSDAClassifier])
def test_classification_label_mapping_and_probabilities(tool):
    rng = np.random.default_rng(17); X = rng.normal(size=(40, 6)); y = np.where(X[:, 0] > 0, "DA", "NE")
    params = {"n_components": 2} if tool is PLSDAClassifier else {}
    if tool in [RFClassifier, XGBoostClassifier]: params.update(n_estimators=8, max_depth=2, n_jobs=1)
    model = tool(parameters=params).fit(X, y)
    assert isinstance(model, Node) and model.predict(X[:4]).shape == (4,)
    assert set(model.predict(X[:4])).issubset({"DA", "NE"})
    if tool is PLSDAClassifier:
        with pytest.raises(NotImplementedError): model.predict_proba(X[:4])
    else:
        probabilities = model.predict_proba(X[:4])
        assert probabilities.shape == (4, 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)


def test_multilabel_is_not_mutually_exclusive():
    X = np.random.default_rng(8).normal(size=(40, 5)); y = (X[:, :2] > 0).astype(int)
    model = LogisticClassifier(multilabel=True).fit(X, y)
    assert model.predict(X[:5]).shape == model.predict_proba(X[:5]).shape == (5, 2)


@pytest.mark.parametrize("tool", [PLSRRegressor, RFRegressor, SVRRegressor, XGBoostRegressor, LinearRegressor])
def test_regression_supports_multiple_targets(tool):
    X = np.random.default_rng(9).normal(size=(35, 5)); y = np.column_stack([2*X[:, 0], -3*X[:, 1]])
    params = {"n_components": 2} if tool is PLSRRegressor else {}
    if tool in [RFRegressor, XGBoostRegressor]: params.update(n_estimators=8, max_depth=2, n_jobs=1)
    model = tool(parameters=params).fit(X, y)
    assert model.predict(X[:4]).shape == (4, 2)
    if tool is LinearRegressor: np.testing.assert_allclose(model.predict(X[:4]), y[:4], atol=1e-10)


@pytest.mark.parametrize("tool", [CLSUnmixer, NNLSUnmixer])
def test_unmixing_recovers_known_mixture(tool):
    references = Spectra([[1, 0, 1, 0], [0, 2, 0, 1]], [400, 401, 402, 403])
    coefficients = np.array([[2, 3], [4, .5]])
    mixture = Spectra(coefficients@references.intensity, references.axis, ("s1", "s2"))
    result = tool(("A", "B")).fit(references).transform(mixture)
    np.testing.assert_allclose(result.coefficients, coefficients, atol=1e-10)
    np.testing.assert_allclose(result.residual_norm, 0, atol=1e-10)
    assert result.component_names == ("A", "B")
    with pytest.raises(ValueError): tool().fit([[1, 2], [2, 4]])


def test_aggregation_preserves_group_identity_and_difference_alignment():
    X = Spectra([[0, 2, 4], [2, 4, 6], [8, 10, 12]], [400, 401, 402], ("a", "b", "c"))
    for tool in [MeanAggregator, MedianAggregator, MaxAggregator, MinAggregator]:
        grouped = tool().transform(X, groups=["s1", "s1", "s2"])
        assert grouped.sample_ids == ("s1", "s2") and grouped.intensity.shape == (2, 3)
        assert tool().transform(X).shape == (3, 1)
    np.testing.assert_allclose(MeanAggregator().transform(X, groups=["s1", "s1", "s2"]).intensity[0], [1, 3, 5])
    background = Spectra([[1, 1, 1]], X.axis)
    np.testing.assert_allclose(SpectralDifference().transform(X, reference=background).intensity, X.intensity-1)
    with pytest.raises(ValueError): SpectralDifference().transform(X, reference=Spectra([[0, 0, 0]], [399, 400, 401]))


def test_embeddings_peaks_and_clusters():
    X = np.random.default_rng(7).normal(size=(20, 5))
    for tool in [PCAFeatures, FactorAnalysisFeatures]:
        assert tool({"n_components": 2}).fit_transform(X).shape == (20, 2)
    cluster = KMeansClusterer({"n_clusters": 2}).fit(X)
    assert cluster.predict(X).shape == (20,) and cluster.transform(X).shape == (20, 2)
    tsne = TSNEEmbedding({"perplexity": 3, "max_iter": 250, "method": "exact"})
    assert tsne.fit_transform(X).shape == (20, 2)
    with pytest.raises(NotImplementedError): tsne.transform(X[:2])
    s = Spectra([[0, 1, 4, 1, 0]], [400, 401, 402, 403, 404])
    np.testing.assert_allclose(PeakExtractor([(401, 403)], "height").transform(s), [[4]])
    np.testing.assert_allclose(PeakExtractor([(401, 403)], "position").transform(s), [[402]])
    assert PeakDetector().transform(s)[0]["positions"].tolist() == [402]


def test_edges_and_digital_counting():
    edge = Edge("classifier", "regressor", condition=Condition("p", "ge", .5))
    np.testing.assert_array_equal(edge.condition.evaluate({"p": [.2, .8]}), [False, True])
    with pytest.raises(ValueError): Edge("same", "same")
    X = np.array([[0], [0], [0], [0], [0], [2], [0], [2]])
    groups = ["blank"]*4 + ["standard"]*4
    model = DigitalSERSRegressor(threshold=1).fit(X, [0]*4+[1]*4, groups=groups)
    np.testing.assert_allclose(model.predict(X, groups=groups), [[0], [1]])
    assert model.last_group_ids_ == ("blank", "standard")
    with pytest.raises(ValueError): model.predict([[2], [2]], groups=["saturated"]*2)
