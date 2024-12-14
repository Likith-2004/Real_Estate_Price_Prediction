import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path):
    # Load data
    df = pd.read_csv('C:\\Users\\madho\\Desktop\\REPP\\mumbai.csv')
    print("Initial Column Types:")
    print(df.dtypes)

    def clean_price(price_col):
        price_str = price_col.astype(str)
        cleaned = price_str.str.replace('₹', '', regex=False).str.replace(',', '', regex=False)
        return pd.to_numeric(cleaned, errors='coerce')

    df['MIN_PRICE'] = clean_price(df['MIN_PRICE'])
    df['MAX_PRICE'] = clean_price(df['MAX_PRICE'])
    df['AVG_PRICE'] = (df['MIN_PRICE'] + df['MAX_PRICE']) / 2
    df.dropna(subset=['AVG_PRICE'], inplace=True)

    numeric_columns = ['TOTAL_FLOOR', 'AGE']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numeric_columns + ['AVG_PRICE'], inplace=True)

    print(f"Number of rows after preprocessing: {len(df)}")

    # Remove outliers
    for col in ['AVG_PRICE', 'TOTAL_FLOOR', 'AGE']:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        df = df[z_scores < 3]  # Retain rows within 3 standard deviations

    print(f"Number of rows after outlier removal: {len(df)}")

    return df