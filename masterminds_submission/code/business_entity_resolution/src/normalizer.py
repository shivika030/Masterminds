import pandas as pd

def preprocess_text(df):
    df['business_name'] = df['business_name'].fillna('').astype(str).str.lower()
    df['business_address'] = df['business_address'].fillna('').astype(str).str.lower()
    df['combined_text'] = df['business_name'] + " " + df['business_address']
    return df