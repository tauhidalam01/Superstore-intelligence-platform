import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_combine_data(real_path, synth_path):
    try:
        real_data = pd.read_parquet(real_path)
        synthetic_data = pd.read_parquet(synth_path)
        
        if 'Customer ID' not in synthetic_data.columns:
            synthetic_data['Customer ID'] = [f'SYNTH_{i}' for i in range(len(synthetic_data))]
        if 'Product Name' not in synthetic_data.columns:
            synthetic_data['Product Name'] = 'Synthetic Product'
        
        augmented_data = pd.concat([real_data, synthetic_data], ignore_index=True, sort=False)
        return augmented_data
    except FileNotFoundError as e:
        logging.error(f"Data file not found: {e}.")
        return None

def create_master_feature_set(df):
    logging.info("Engineering the V3 master feature set...")
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df.dropna(subset=['Order Date'], inplace=True)

    if 'Profit_Margin' not in df.columns:
        df['Profit_Margin'] = np.where(df['Total Sales'] > 0, df['Total Profit'] / df['Total Sales'], 0)

    df['month_of_year'] = df['Order Date'].dt.month
    df['first_order_date'] = df.groupby('Customer ID')['Order Date'].transform('min')
    df['days_since_first_order'] = (df['Order Date'] - df['first_order_date']).dt.days

    df['postal_code_profitability'] = df.groupby('Postal Code')['Profit_Margin'].transform('mean')
    df['state_sales_volume'] = df.groupby('State')['Total Sales'].transform('sum')
    df['subcategory_avg_margin'] = df.groupby('Sub-Category')['Profit_Margin'].transform('mean')
    df['category_avg_discount'] = df.groupby('Category')['Discount'].transform('mean')
    df['customer_avg_order_size'] = df.groupby('Customer ID')['Total Sales'].transform('mean')
    
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

def train_final_classifier(df, output_path):
    logging.info("--- Training Final Profitability Classifier (V3) ---")
    df['is_profitable'] = (df['Total Profit'] > 0).astype(int)
    features = [
        'Segment', 'Region', 'State', 'Category', 'Sub-Category', 'Quantity', 
        'Sales Price', 'Discount', 'month_of_year', 'days_since_first_order',
        'postal_code_profitability', 'state_sales_volume', 'subcategory_avg_margin',
        'category_avg_discount', 'customer_avg_order_size'
    ]
    target = 'is_profitable'
    
    X, y = df[features], df[target]
    
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_features), ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1))])
    
    param_grid = {'classifier__n_estimators': [200], 'classifier__max_depth': [7], 'classifier__learning_rate': [0.1]}
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    
    logging.info("Tuning and training final classifier...")
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    logging.info(f"Tuned Classifier Best CV Score: {grid_search.best_score_:.4f}")

    model_path = output_path / "final_v3_profitability_classifier.joblib"
    joblib.dump(best_model, model_path)
    logging.info(f"✅ Final V3 Classifier saved to '{model_path}'")

def train_final_forecaster(df, output_path):
    logging.info("--- Training Final Profit Forecaster (V3) ---")
    df_profitable = df[df['Total Profit'] > 1].copy()
    features = [
        'Segment', 'Region', 'State', 'Category', 'Sub-Category', 'Quantity', 
        'Sales Price', 'Discount', 'month_of_year', 'days_since_first_order',
        'postal_code_profitability', 'state_sales_volume', 'subcategory_avg_margin',
        'category_avg_discount', 'customer_avg_order_size'
    ]
    target = 'Total Profit'
    
    X, y_log = df_profitable[features], np.log1p(df_profitable[target])
    
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_features), ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', XGBRegressor(random_state=42, n_jobs=-1))])
    
    param_grid = {'regressor__n_estimators': [300], 'regressor__max_depth': [10], 'regressor__learning_rate': [0.1]}
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1)

    logging.info("Tuning and training final forecaster...")
    grid_search.fit(X, y_log)
    best_model = grid_search.best_estimator_
    logging.info(f"Tuned Forecaster Best CV Score: {grid_search.best_score_:.4f}")

    model_path = output_path / "final_v3_profit_forecaster.joblib"
    joblib.dump(best_model, model_path)
    logging.info(f"✅ Final V3 Forecaster saved to '{model_path}'")

def main():
    parser = argparse.ArgumentParser(description="Train final V3 models on augmented and feature-enriched data.")
    parser.add_argument("--real-path", default='data/processed/cleaned_superstore_data.parquet')
    parser.add_argument("--synth-path", default='data/processed/synthetic_base_transactions.parquet')
    parser.add_argument("--output-dir", default='models')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    augmented_df = load_and_combine_data(args.real_path, args.synth_path)
    if augmented_df is not None:
        master_df = create_master_feature_set(augmented_df)
        train_final_classifier(master_df, output_dir)
        train_final_forecaster(master_df, output_dir)
        logging.info("--- Final V3 Model Training Pipeline Complete ---")

if __name__ == '__main__':
    main()