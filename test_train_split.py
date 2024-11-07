from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

# Splitter stage as a transformer
class DataSplitter(BaseEstimator, TransformerMixin):
    def __init__(self, target_feature, test_size=0.25, random_state=1):
        self.target_feature = target_feature
        self.test_size = test_size
        self.random_state = random_state
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        y = X[self.target_feature]
        X = X.drop(columns=[self.target_feature])
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
