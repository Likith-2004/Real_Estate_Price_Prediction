# main.py (or a separate script to train the model)
from subfolder.model_creation import create_model_pipeline, save_model
import pandas as pd

# Load the data (use your own CSV file path)
file_path = "C:\Users\madho\Desktop\REPP\mumbai.csv"
df = pd.read_csv(file_path)

# Train the model
model, encoder, scaler = create_model_pipeline(df)

# Save the model, encoder, and scaler
if model is not None:
    save_model(model, encoder, scaler, "C:\Users\madho\Desktop\REPP")
