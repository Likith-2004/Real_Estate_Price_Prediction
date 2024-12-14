# **Mumbai Real Estate Price Predictor**

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
