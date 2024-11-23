import pandas as pd
import os 

def clean_data(file_path):
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)  # Remove missing values
    return data
