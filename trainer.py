from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class ModelTrainer(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 criterion='gini',
                 bootstrap=True,
                 model='RandomForest'):
        """
        Initializes the model with either RandomForest or XGBoost based on the provided parameter.
        """
        # Initialize the RandomForestClassifier or XGBClassifier with the provided parameters
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            bootstrap=bootstrap
        ) if model == 'RandomForest' else XGBClassifier()

    def fit(self, X, y):
        """
        Fits the model to the provided training data.
        """
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Makes predictions on the provided data.
        """
        return self.model.predict(X)
