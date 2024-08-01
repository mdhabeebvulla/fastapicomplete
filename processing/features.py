# processing/features.py
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Example transformation: add a feature
        X['petal_ratio'] = X['petal_length'] / X['petal_width']
        return X
