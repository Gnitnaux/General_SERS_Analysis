from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from .estimators import RegressionNode


class PLSRRegressor(RegressionNode):
    """Partial least-squares regression. Configuration, fit/predict and learned state follow the inherited estimator adapter.

    Attributes
    ----------
    estimator_type: Underlying estimator class instantiated during fit.

    Inherited interface
    -------------------
    See RegressionNode for methods and attributes not overridden here.
    """
    estimator_type = PLSRegression


class RFRegressor(RegressionNode):
    """Random forest regression. Configuration, fit/predict and learned state follow the inherited estimator adapter.

    Attributes
    ----------
    estimator_type: Underlying estimator class instantiated during fit.

    Inherited interface
    -------------------
    See RegressionNode for methods and attributes not overridden here.
    """
    estimator_type = RandomForestRegressor


class SVRRegressor(RegressionNode):
    """Support-vector regression. Configuration, fit/predict and learned state follow the inherited estimator adapter.

    Attributes
    ----------
    estimator_type: Underlying estimator class instantiated during fit.

    Inherited interface
    -------------------
    See RegressionNode for methods and attributes not overridden here.
    """
    estimator_type = SVR


class XGBoostRegressor(RegressionNode):
    """XGBoost regression. Configuration, fit/predict and learned state follow the inherited estimator adapter.

    Attributes
    ----------
    estimator_type: Underlying estimator class instantiated during fit.

    Inherited interface
    -------------------
    See RegressionNode for methods and attributes not overridden here.
    """
    estimator_type = XGBRegressor


class LinearRegressor(RegressionNode):
    """Ordinary linear regression. Configuration, fit/predict and learned state follow the inherited estimator adapter.

    Attributes
    ----------
    estimator_type: Underlying estimator class instantiated during fit.

    Inherited interface
    -------------------
    See RegressionNode for methods and attributes not overridden here.
    """
    estimator_type = LinearRegression


from .digital_sers import DigitalSERSRegressor
