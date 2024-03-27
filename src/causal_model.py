import lightgbm as lgb
from econml.dml import CausalForestDML
from sklearn.model_selection import KFold

class PriceElasticityModel:
    def __init__(self, config: dict):
        self.config = config
        
        # Base learners for Double Machine Learning
        self.model_y = lgb.LGBMRegressor(
            objective='tweedie',
            tweedie_variance_power=config['model']['tweedie_variance_power'],
            n_estimators=config['model']['n_estimators'],
            random_state=config['model']['random_state']
        )
        self.model_t = lgb.LGBMRegressor(
            objective='regression',
            n_estimators=config['model']['n_estimators'],
            random_state=config['model']['random_state']
        )
        
        self.dml_forest = CausalForestDML(
            model_y=self.model_y,
            model_t=self.model_t,
            cv=KFold(n_splits=config['model']['cv_splits'], shuffle=True, random_state=config['model']['random_state']),
            random_state=config['model']['random_state']
        )

    def fit(self, Y, T, X, W):
        self.dml_forest.fit(Y=Y, T=T, X=X, W=W)

    def estimate_effect(self, X):
        return self.dml_forest.effect(X)