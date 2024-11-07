from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Custom transformer for label encoding
# Custom transformer for label encoding
class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        """
        Fit the transformer, identifying columns with object (categorical) data type.
        """
        print("=== Initiating label encoding ===")
        self.le = LabelEncoder()
        self.obj_col_list = X.select_dtypes(include=['object']).columns.tolist()
        # Fit label encoder for each categorical column
        self.le_dict = {}
        for col in self.obj_col_list:
            self.le_dict[col] = LabelEncoder()
            self.le_dict[col].fit(X[col])
        return self
    
    def transform(self, X):
        """
        Apply label encoding to identified categorical columns.
        """
        X_encoded = X.copy()
        for col in self.obj_col_list:
            X_encoded[col] = self.le_dict[col].transform(X[col])
        print("=== Label encoding is completed ===")
        return X_encoded

    def inverse_transform(self, X):
        """
        Reverse the label encoding, transforming the encoded labels back to their original form.
        """
        X_decoded = X.copy()
        for col in self.obj_col_list:
            if col in X_decoded.columns:  # Ensure the column exists in the input DataFrame
                X_decoded[col] = self.le_dict[col].inverse_transform(X_decoded[col])
            else:
                print(f"Warning: Column {col} not found in the data during inverse transformation.")
        print("=== Label decoding is completed ===")
        return X_decoded

    

# Custom transformer for standard scaling
class StandardScalerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        """
        Fit the transformer, identifying columns with numeric data types and applying standard scaling.
        """
        print("=== Initiating standard scaling ===")
        self.scaler = StandardScaler()
        self.num_col_list = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.scaler.fit(X[self.num_col_list])
        print("=== Standard scaling is completed ===")
        return self
    
    def transform(self, X):
        """
        Apply standard scaling to identified numeric columns.
        """
        X_scaled = X.copy()
        X_scaled[self.num_col_list] = self.scaler.transform(X[self.num_col_list])
        return X_scaled
