import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    dupe_count = data.duplicated().sum()
    if dupe_count > 0:
        logging.info(f"Found {dupe_count} duplicate rows. Dropping them.")
        data = data.drop_duplicates().reset_index(drop=True)
    else:
        logging.info("No duplicate rows found. Good to go!")
    return data

def fix_dates_and_ids(data: pd.DataFrame) -> pd.DataFrame:
    logging.info("Fixing up date columns and checking Order IDs...")
    
    data['Order Date'] = pd.to_datetime(data['Order Date'], format='%d/%m/%y', errors='coerce')
    data['Ship Date'] = pd.to_datetime(data['Ship Date'], errors='coerce')

    initial_rows = len(data)
    data.dropna(subset=['Order Date'], inplace=True)
    if len(data) < initial_rows:
        logging.warning(f"Dropped {initial_rows - len(data)} rows with bad Order Dates.")

    id_year = data['Order ID'].str.split('-').str[1]
    order_date_year = data['Order Date'].dt.year.astype(str)
    
    mismatched_ids = id_year != order_date_year
    if mismatched_ids.any():
        logging.info(f"Found {mismatched_ids.sum()} Order IDs with the wrong year. Correcting them now.")
        data.loc[mismatched_ids, 'Order ID'] = \
            'CA-' + data.loc[mismatched_ids, 'Order Date'].dt.year.astype(str) + \
            '-' + data.loc[mismatched_ids, 'Order ID'].str.split('-').str[2]
            
    return data

def clean_shipping_info(data: pd.DataFrame) -> pd.DataFrame:
    logging.info("Calculating 'Days to Ship' and cleaning up shipping data.")
    data['Days to Ship'] = (data['Ship Date'] - data['Order Date']).dt.days
    
    bad_shipping_dates = (data['Days to Ship'] < 0) | (data['Days to Ship'] > 90)
    if bad_shipping_dates.any():
        logging.warning(f"Found {bad_shipping_dates.sum()} impossible shipping dates. Setting them to NaN.")
        data.loc[bad_shipping_dates, ['Ship Date', 'Days to Ship']] = np.nan
        
    return data

def fill_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    logging.info("Imputing missing values for Ship Mode and Quantity...")
    
    data.loc[(data['Ship Mode'].isna()) & (data['Days to Ship'] == 0), 'Ship Mode'] = 'Same Day'
    data.loc[(data['Ship Mode'].isna()) & (data['Days to Ship'] == 7), 'Ship Mode'] = 'Standard Class'

    if data['Quantity'].isnull().any():
        median_qty = data['Quantity'].median()
        logging.info(f"Filling {data['Quantity'].isnull().sum()} missing Quantities with the median value ({median_qty}).")
        data['Quantity'].fillna(median_qty, inplace=True)
        data['Quantity'] = data['Quantity'].astype(int)
        
    return data

def standardize_text_fields(data: pd.DataFrame) -> pd.DataFrame:
    logging.info("Standardizing state names and postal codes.")
    
    data['Postal Code'] = data['Postal Code'].astype(str).str.zfill(5)

    state_map = {'CA': 'California', 'NY': 'New York', 'TX': 'Texas', 'NJ': 'New Jersey', 'WA\\': 'Washington'}
    data['State'] = data['State'].str.strip().replace(state_map)
    
    return data
    
def create_financial_metrics(data: pd.DataFrame) -> pd.DataFrame:
    logging.info("Building out financial metrics like Total Sales and Profit.")
    
    data['Sales Price'] = data['Sales Price'].abs()
    
    data['Original Price'] = data['Sales Price'] / (1 - data['Discount'])
    data['Total Sales'] = data['Sales Price'] * data['Quantity']
    data['Total Profit'] = data['Profit'] * data['Quantity']
    data['Total Discount'] = (data['Original Price'] - data['Sales Price']) * data['Quantity']
    
    return data

def remove_outliers(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    logging.info(f"Scanning for extreme outliers in {columns}...")
    initial_rows = len(data)
    
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        clean_data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        
        outliers_removed = len(data) - len(clean_data)
        if outliers_removed > 0:
            logging.info(f"Removed {outliers_removed} outliers from '{col}'.")
        data = clean_data
            
    logging.info(f"Total rows removed due to outliers: {initial_rows - len(data)}")
    return data.reset_index(drop=True)

def run_cleaning_pipeline(raw_csv_path: str, output_parquet_path: str) -> pd.DataFrame:
    logging.info(f"--- 🚀 Kicking off the data prep pipeline for {raw_csv_path} ---")
    
    df = pd.read_csv(raw_csv_path)
    logging.info(f"Successfully loaded raw data. Initial shape: {df.shape}")
    
    df = remove_duplicates(df)
    df = fix_dates_and_ids(df)
    df = clean_shipping_info(df)
    df = fill_missing_values(df)
    df = standardize_text_fields(df)
    df = create_financial_metrics(df)
    
    df.drop(columns=['Customer Name'], inplace=True)
    
    df = remove_outliers(df, columns=['Sales Price', 'Profit'])
    
    df.to_parquet(output_parquet_path, index=False)
    logging.info(f"Pipeline complete! Final data shape: {df.shape}")
    logging.info(f"✅ Cleaned data saved to '{output_parquet_path}'")
    
    return df

if __name__ == '__main__':
    RAW_DATA_PATH = 'data/raw/SuperStore_Dataset.csv'
    PROCESSED_DATA_PATH = 'data/processed/cleaned_superstore_data.parquet'
    
    os.makedirs('data/processed', exist_ok=True)
    
    if not os.path.exists(RAW_DATA_PATH):
        logging.error(f"Houston, we have a problem. Raw data file not found at '{RAW_DATA_PATH}'.")
        logging.error("Please place the SuperStore_Dataset.csv there and try again.")
    else:
        run_cleaning_pipeline(
            raw_csv_path=RAW_DATA_PATH,
            output_parquet_path=PROCESSED_DATA_PATH
        )