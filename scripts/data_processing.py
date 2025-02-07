# src/data_processing.py

import pandas as pd
import numpy as np
import logging

# Configure logging (do this once)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def handle_missing_values(fraud_data, strategy='impute', columns=None):  # Changed df to fraud_data
    """Handles missing values."""
    if columns is None:
        columns = fraud_data.columns  # Use fraud_data here

    logger.info(f"Handling missing values using strategy: {strategy}")

    if strategy == 'impute':
        for col in columns:
            if fraud_data[col].isnull().any():  # Use fraud_data here
                missing_count = fraud_data[col].isnull().sum()  # Use fraud_data here
                logger.info(f"Column '{col}' has {missing_count} missing values. Imputing...")
                if pd.api.types.is_numeric_dtype(fraud_data[col]):  # Use fraud_data here
                    fraud_data[col].fillna(fraud_data[col].median(), inplace=True)  # Use fraud_data here
                elif pd.api.types.is_object_dtype(fraud_data[col]):  # Use fraud_data here
                    fraud_data[col].fillna(fraud_data[col].mode()[0], inplace=True)  # Use fraud_data here
                else:
                    fraud_data[col].fillna(method='ffill', inplace=True)  # Use fraud_data here
            else:
                logger.info(f"Column '{col}' has no missing values.")
    elif strategy == 'drop':
        rows_dropped = fraud_data.dropna(subset=columns, inplace=True)  # Use fraud_data here
        logger.info(f"Dropped {rows_dropped} rows due to missing values.")
    else:
        logger.error("Invalid strategy. Choose 'impute' or 'drop'.")
        raise ValueError("Invalid strategy. Choose 'impute' or 'drop'.")

    return fraud_data  # Return the modified fraud_data


def clean_data(fraud_data):  # Changed df to fraud_data
    """Removes duplicates and corrects data types."""
    original_rows = len(fraud_data)  # Use fraud_data here
    fraud_data.drop_duplicates(inplace=True)  # Use fraud_data here
    duplicates_removed = original_rows - len(fraud_data)
    logger.info(f"Removed {duplicates_removed} duplicate rows.")

    time_cols = ['signup_time', 'purchase_time']
    for col in time_cols:
      if col in fraud_data.columns: # Check if the column exists
        try:
           fraud_data[col] = pd.to_datetime(fraud_data[col])  # Use fraud_data here
           logger.info(f"Converted column '{col}' to datetime.")
        except (ValueError, TypeError):
          logger.warning(f"Could not convert {col} to datetime. Check data format.")

    return fraud_data  # Return the modified fraud_data



def merge_ip_country(fraud_data, ip_df): # Changed df to fraud_data
    """Merges with IP to country mapping."""
    if ip_df is None:
        logger.info("No IP to country mapping provided. Skipping merge.")
        return fraud_data

    fraud_data['ip_int'] = fraud_data['ip_address'].apply(convert_ip_to_int) # Use fraud_data here
    merged_df = pd.merge_asof(fraud_data.sort_values('ip_int'), # Use fraud_data here
                              ip_df.sort_values('lower_bound_ip_address'),
                              left_on='ip_int', right_on='lower_bound_ip_address',
                              direction='forward')

    merged_df.drop('ip_int', axis=1, inplace=True)
    logger.info("Merged transaction data with IP to country mapping.")
    return merged_df


def encode_categorical_features(fraud_data, columns=None):  # Changed df to fraud_data
    """Encodes categorical features."""
    if columns is None:
        cat_cols = fraud_data.select_dtypes(include=['object']).columns  # Use fraud_data here
    else:
        cat_cols = columns

    logger.info(f"Encoding categorical features: {cat_cols}")

    for col in cat_cols:
      if col in fraud_data.columns:  # Use fraud_data here
        fraud_data = pd.get_dummies(fraud_data, columns=[col], prefix=col, dummy_na=False)  # Use fraud_data here
    return fraud_data  # Return the modified fraud_data



def normalize_numerical_features(fraud_data, columns=None):  # Changed df to fraud_data
    """Normalizes numerical features."""
    from sklearn.preprocessing import MinMaxScaler

    if columns is None:
      num_cols = fraud_data.select_dtypes(include=np.number).columns  # Use fraud_data here
    else:
      num_cols = columns

    logger.info(f"Normalizing numerical features: {num_cols}")

    for col in num_cols:
      if col in fraud_data.columns: # Check if the column exists
        scaler = MinMaxScaler()
        fraud_data[col] = scaler.fit_transform(fraud_data[[col]])  # Use fraud_data here
    return fraud_data  # Return the modified fraud_data



# Example usage (in your notebook or main script):
if __name__ == '__main__':
    fraud_data_path = 'data/Fraud_Data.csv'
    ip_to_country_path = 'data/IpAddress_to_Country.csv'

    try:
        fraud_data, ip_df = load_data(fraud_data_path, ip_to_country_path)  # Load into fraud_data

        if fraud_data is not None:
            fraud_data = clean_data(fraud_data)  # Pass fraud_data
            fraud_data = handle_missing_values(fraud_data)  # Pass fraud_data
            if ip_df is not None:
              fraud_data = merge_ip_country(fraud_data, ip_df) # Pass fraud_data
            fraud_data = encode_categorical_features(fraud_data)  # Pass fraud_data
            fraud_data = normalize_numerical_features(fraud_data)  # Pass fraud_data
            logger.info("Data preprocessing completed.")
            print(fraud_data.head()) # Or save: fraud_data.to_csv("processed_data.csv", index=False)

    except Exception as e:
        logger.exception(f"An error occurred: {e}")