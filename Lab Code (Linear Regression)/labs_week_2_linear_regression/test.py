import pandas as pd
import numpy as np

def create_test_csv(file_path):
    """
    This function creates a CSV file with some missing values for testing purposes.
    
    Parameters:
    file_path (str): The path where the CSV file will be saved.
    """
    # Create a DataFrame with some missing values
    data = {
        'Size of Plot (sq. meters)': [600, 800, np.nan, 1000, 1200],
        'Distance from Airport (km)': [5, 10, 15, np.nan, 25],
        'Proximity to Main Road (km)': [1, 2, 3, 4, np.nan],
        'Proximity to City Center (km)': [10, 20, 30, 40, 50],
        'Land Price (GHS)': [10000, 20000, 30000, 40000, np.nan]
    }
    
    df = pd.DataFrame(data)
    
    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False)
    print(f"CSV file created at {file_path}")

# Example usage
create_test_csv('test_data.csv')