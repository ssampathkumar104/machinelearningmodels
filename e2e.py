import pandas as pd
import pickle as pkl
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

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

    # Preparing the pipeline
    pipeline = Pipeline([
        ('label_encoder', LabelEncoderTransformer()),  # Add label encoding as a stage
        ('scaling', StandardScalerTransformer()),  # Add standard scaling as a stage
        ('splitter', DataSplitter(target_feature='fertilizer_name')),
        ('model_trainer', ModelTrainer(n_estimators=50, max_depth=10, 
                                       min_samples_split=2, min_samples_leaf=1, 
                                       criterion='gini', bootstrap=True)),
        ('model_saver', ModelSaver(filename='final_rf_model.pkl'))
    ])

    # Step 2: Transform the data through encoding and scaling
    df_transformed = pipeline.named_steps['label_encoder'].fit_transform(df)
    df_transformed = pipeline.named_steps['scaling'].fit_transform(df_transformed)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = pipeline.named_steps['splitter'].fit_transform(df_transformed)

    # Step 3: Fit the model
    pipeline.named_steps['model_trainer'].fit(X_train, y_train)

    # Step 4: Save the model after training
    pipeline.named_steps['model_saver'].transform(X_test, pipeline.named_steps['model_trainer'].model)

    # Step 5: Make predictions using the test data
    y_pred = pipeline.named_steps['model_trainer'].predict(X_test)
    print(y_pred)

    # Step 6: Evaluate model performance
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
