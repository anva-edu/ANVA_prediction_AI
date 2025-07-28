def main():
    pass

import pandas as pd
import joblib
import os

def predict_student_performance(input_data: pd.DataFrame):
    """
    Predict G3 scores for preprocessed input_data using the trained model.
    Assumes input_data is preprocessed (scaled, encoded like training data).
    """
    model_path = 'models/best_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    predictions = model.predict(input_data)
    return predictions

if __name__ == "__main__":
    # Load a sample student record from X_test
    X_test = pd.read_csv('data/processed/X_test.csv')
    # Predict G3 scores
    preds = predict_student_performance(X_test)
    print("Predicted G3 scores for X_test:")
    print(preds)
