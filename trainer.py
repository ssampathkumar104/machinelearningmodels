from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

class ModelTrainer(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 criterion='gini',
                 bootstrap=True):
        """
        Initializes the RandomForest model with the provided parameters.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.bootstrap = bootstrap

        # Initialize the RandomForestClassifier with provided parameters
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            bootstrap=self.bootstrap
        )

    def fit(self, X, y):
        """
        Fits the RandomForest model to the provided training data.
        """
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """
        Makes predictions using the fitted RandomForest model.
        """
        return self.model.predict(X)
