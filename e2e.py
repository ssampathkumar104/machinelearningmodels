from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Import custom transformers and utility classes
from transformations import LabelEncoderTransformer, StandardScalerTransformer
from trainer import ModelTrainer
from saver import ModelSaver
from test_train_split import DataSplitter
from dataUtils import load_data  # Load data function from dataUtils.py
from dataAnalysis import data_overview  # Import data overview function

if __name__ == "__main__":

    # Load data configuration and dataset
    input_details = load_data()  # Load input details from configuration
    file_id = input_details['file_id']
    download_url = f"https://drive.google.com/uc?id={file_id}"
    df = pd.read_csv(download_url)  # Read data from the Google Drive URL

    # Step 1: Perform data overview and preprocessing
    df = data_overview(df)  # Apply data overview for basic EDA

    # Step 2: Create preprocessing pipeline (Label Encoding + Scaling)
    preprocessing_pipeline = Pipeline([
        ('label_encoder', LabelEncoderTransformer()),  # Add label encoding as a stage
        ('scaling', StandardScalerTransformer())  # Add standard scaling as a stage
    ])

    # Transform data using preprocessing pipeline
    df_transformed = preprocessing_pipeline.fit_transform(df)

    # Split the dataset into training and testing sets
    splitter = DataSplitter(target_feature='fertilizer_name')
    X_train, X_test, y_train, y_test = splitter.fit_transform(df_transformed)

    print('y_train data type:', y_train.dtype)
    y_train = pd.Series(y_train).astype('category')  # Convert to category type
    print('y_train after conversion to category:', y_train.dtype)
    print("Unique values in y_train:", y_train.unique())
    print("Value counts in y_train:", y_train.value_counts())
    print('Shape of y_train:', y_train.shape)
    if len(y_train.shape) > 1:
        y_train = y_train.flatten()
    print("Data type check for unique labels:", y_train.apply(type).unique())
    y_train = pd.Series(y_train).astype(int)  # Ensure it is treated as an integer
    print("y_train after conversion:", y_train.head())

    # Step 3: Initialize and train the model
    model_trainer = ModelTrainer(n_estimators=50, max_depth=10, min_samples_split=2, min_samples_leaf=1, criterion='gini', bootstrap=True)
    model_trainer.fit(X_train, y_train)

    # Step 4: Save the model after training
    model_saver = ModelSaver(filename='final_rf_model.pkl')
    model_saver.transform(X_test, model_trainer.model)

    # Step 5: Make predictions using the test data
    y_pred = model_trainer.predict(X_test)
    print("Predictions:", y_pred)

    # Reverse label encoding for the predictions (from transformed to original labels)
    y_pred_df = pd.DataFrame(y_pred, columns=['fertilizer_name'])
    y_pred_original = preprocessing_pipeline.named_steps['label_encoder'].inverse_transform(y_pred_df)
    print("Predicted Fertilizer Names:", y_pred_original)

    # Step 6: Evaluate model performance
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    
