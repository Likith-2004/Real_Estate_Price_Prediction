# model_creation.py
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.ensemble import RandomForestRegressor

def create_model_pipeline(df):
    """
    This function creates a pipeline to train a RandomForest model.
    It preprocesses data (encoding categorical variables, scaling numerical features),
    tunes hyperparameters, and evaluates the model.
    """

    features = ['TOTAL_FLOOR', 'CITY', 'AGE', 'PROPERTY_TYPE']
    target = 'AVG_PRICE'

    # Prepare the data (features and target)
    X = df[features]
    y = df[target]

    # Check if there is enough data
    if len(df) < 10:
        print("Error: Not enough data to train the model.")
        return None

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Identify categorical and numerical features
    categorical_features = ['CITY', 'PROPERTY_TYPE']
    numerical_features = ['TOTAL_FLOOR', 'AGE']

    # Use TargetEncoder for encoding categorical features
    encoder = TargetEncoder(cols=categorical_features)
    X_train = encoder.fit_transform(X_train, y_train)
    X_test = encoder.transform(X_test)

    # Scale the numerical features
    scaler = StandardScaler()
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    # Hyperparameter tuning for RandomForestRegressor
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform grid search for hyperparameter tuning
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # Print the best parameters
    print("Best Model Parameters:", grid_search.best_params_)

    # Evaluate the model
    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)

    print(f"Train R² Score: {train_score:.2f}")
    print(f"Test R² Score: {test_score:.2f}")

    # Return the best model, encoder, and scaler
    return best_model, encoder, scaler

def save_model(model, encoder, scaler, save_dir):
    """
    Save the trained model, encoder, and scaler to disk.
    """
    joblib.dump(model, os.path.join(save_dir, 'price_predictor.joblib'))
    joblib.dump(encoder, os.path.join(save_dir, 'encoder.joblib'))
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.joblib'))
    print("Model, encoder, and scaler saved.")