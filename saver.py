import pickle as pkl
from sklearn.base import BaseEstimator, TransformerMixin

# Model saving transformer
class ModelSaver(BaseEstimator, TransformerMixin):
    def __init__(self, filename):
        """
        Initializes the ModelSaver with the filename where the model will be saved.
        """
        self.filename = filename
    
    def fit(self, X, y=None):
        """
        The fit method, which doesn't alter data but allows compatibility with sklearn pipelines.
        """
        return self
    
    def transform(self, X, y=None):
        """
        Saves the provided model (or other object) in `y` to the specified file.
        """
        with open(self.filename, "wb") as file:
            pkl.dump(y, file)  # Save the model (or other data) to file
        return X
