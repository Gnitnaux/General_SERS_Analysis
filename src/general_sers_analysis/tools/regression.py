from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from .estimators import RegressionNode


class PLSRRegressor(RegressionNode):
    estimator_type = PLSRegression


class RFRegressor(RegressionNode):
    estimator_type = RandomForestRegressor


class SVRRegressor(RegressionNode):
    estimator_type = SVR


class XGBoostRegressor(RegressionNode):
    estimator_type = XGBRegressor


class LinearRegressor(RegressionNode):
    estimator_type = LinearRegression


from .digital_sers import DigitalSERSRegressor
