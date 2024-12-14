# **Mumbai Real Estate Price Predictor**

![Screenshot 2024-12-14 073935](https://github.com/user-attachments/assets/f1391984-fe34-4545-811d-50da5a5f5435)

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Technology Stack](#technology-stack)
4. [Installation Guide](#installation-guide)
5. [Usage](#usage)
6. [Workflow](#workflow)
7. [Model Training](#model-training)
8. [Contributing](#contributing)
9. [License](#license)

---

## **Project Overview**
The **Mumbai Real Estate Price Predictor** is a web application that predicts real estate property prices in Mumbai based on user inputs such as the total number of floors, city, property age, and property type. It leverages a trained machine learning model to generate accurate predictions.

---

## **Key Features**
- **User Input Form**: Users can input data like total floors, city, property age, and property type.
- **Machine Learning Model**: Price prediction is powered by a trained `RandomForestRegressor` model.
- **Dropdown Selections**: Dynamic city and property type dropdowns.
- **Real-Time Prediction**: Displays the predicted price immediately after form submission.
- **Error Handling**: Provides error messages if invalid input is provided.

---

## **Technology Stack**
- **Backend**: Flask (Python) for server-side processing.
- **Frontend**: HTML, CSS, and Jinja2 templating for building the UI.
- **Machine Learning**: `RandomForestRegressor` for price prediction, `TargetEncoder` and `StandardScaler` for data preprocessing.
- **File Formats**: `.joblib` used to save the model, encoder, and scaler.

---

## **Installation Guide**
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repository/mumbai-real-estate-predictor.git
   cd mumbai-real-estate-predictor
2.  **Install required dependencies**:
    
    ```bash
    pip install -r requirements.txt
    ```

3.  **Ensure that you have a trained model**: If you haven't trained the model yet, follow the `model_training.py` script to train and save the model, encoder, and scaler.

---

## **Usage**

1.  **Start the Flask server**:
    
    ```bash
    python app.py
    ```

2.  **Open the app**: Go to `http://127.0.0.1:5000` in your browser.
    
3.  **Submit the form**: Fill out the form with details like total floors, city, property age, and type. Click **Predict Price** to see the predicted price.

---

## **Workflow**

1.  **Data Collection**: A dataset containing features like the number of floors, city, age of the property, property type, and price.
2.  **Model Training**: Train a model using `RandomForestRegressor`, and save the encoder and scaler.
3.  **Prediction**:
    *   User submits property details.
    *   The backend processes the data using the encoder and scaler, and the model predicts the price.
    *   The result is displayed on the same page.

---

## **Model Training**

1.  **Train the model**: Use the provided dataset and the `model_training.py` script to train the machine learning model. This will generate the following files:
    
    *   `price_predictor.joblib`: Trained model
    *   `encoder.joblib`: Encoder for categorical data
    *   `scaler.joblib`: Scaler for numerical data
2.  **Save and load model**:
    
    *   The model, encoder, and scaler are saved and loaded using `joblib`.

---

## **Contributing**

Contributions are welcome! If you'd like to contribute, please fork the repository, make changes, and create a pull request.

---

## **License**

This project is licensed under the MIT License - see the LICENSE file for details.
e for details.
