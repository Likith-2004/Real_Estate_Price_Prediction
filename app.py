from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load the trained model, encoder, and scaler
model = joblib.load('price_predictor.joblib')
encoder = joblib.load('encoder.joblib')
scaler = joblib.load('scaler.joblib')

# Mock dropdown options
cities = ["Mumbai", "Pune", "Delhi"]
property_types = ["Apartment", "Villa", "Row House"]

@app.route('/')
def home():
    # Render the home page with dropdown options
    return render_template('index.html', cities=cities, property_types=property_types)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        total_floors = int(request.form['total_floors'])
        city = request.form['city']
        age = int(request.form['age'])
        property_type = request.form['property_type']

        # Preprocess input features
        input_features = [[total_floors, city, age, property_type]]

        # Encode categorical features using the saved encoder
        encoded_features = encoder.transform(input_features)

        # Scale numerical features
        numerical_features_indices = [0, 2]  # Indices of 'TOTAL_FLOOR' and 'AGE'
        encoded_features[:, numerical_features_indices] = scaler.transform(
            encoded_features[:, numerical_features_indices]
        )

        # Predict price using the trained model
        predicted_price = model.predict(encoded_features)

        # Pass input values and prediction back to the template
        return render_template(
            'index.html',
            cities=cities,
            property_types=property_types,
            prediction=f'Predicted Price: ₹{predicted_price[0]:,.2f}',
            total_floors=total_floors,
            city=city,
            age=age,
            property_type=property_type
        )
    except Exception as e:
        # Handle errors gracefully
        return render_template(
            'index.html',
            cities=cities,
            property_types=property_types,
            error=f"Error: {str(e)}",
            total_floors=request.form.get('total_floors'),
            city=request.form.get('city'),
            age=request.form.get('age'),
            property_type=request.form.get('property_type')
        )

if __name__ == '__main__':
    app.run(debug=True)
