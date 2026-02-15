import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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

def generate_and_save_shap_plots(data_path, model_path, output_dir):
    logging.info("--- Starting SHAP Plot Generation ---")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_parquet(data_path)
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Profit_Margin'] = np.where(df['Total Sales'] > 0, df['Total Profit'] / df['Total Sales'], 0)
        
        model = joblib.load(model_path)
    except FileNotFoundError as e:
        logging.error(f"Error loading files: {e}. Halting.")
        return

    logging.info("Preparing data for SHAP analysis...")
    df_sample = df.sample(n=min(2000, len(df)), random_state=42)
    master_df_sample = create_master_feature_set(df_sample)
    
    features = [
        'Segment', 'Region', 'State', 'Category', 'Sub-Category', 'Quantity', 
        'Sales Price', 'Discount', 'month_of_year', 'days_since_first_order',
        'postal_code_profitability', 'state_sales_volume', 'subcategory_avg_margin',
        'category_avg_discount', 'customer_avg_order_size'
    ]
    X_for_shap = master_df_sample[features]
    
    preprocessor = model.named_steps['preprocessor']
    explainer_model = model.named_steps['regressor']
    
    X_transformed = preprocessor.transform(X_for_shap)
    
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(X_for_shap.select_dtypes(include=['object', 'category']).columns)
    num_features = X_for_shap.select_dtypes(include=np.number).columns.tolist()
    final_feature_names = num_features + list(cat_features)
    
    X_transformed_df = pd.DataFrame(X_transformed.toarray(), columns=final_feature_names)

    logging.info("Calculating SHAP values...")
    explainer = shap.TreeExplainer(explainer_model)
    shap_values = explainer.shap_values(X_transformed_df)
    
    plt.figure()
    shap.summary_plot(shap_values, X_transformed_df, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Bar Plot)", fontsize=16)
    plt.tight_layout()
    bar_plot_path = output_path / "shap_summary_bar.png"
    plt.savefig(bar_plot_path)
    plt.close()
    logging.info(f"✅ Bar plot saved to '{bar_plot_path}'")
    
    plt.figure()
    shap.summary_plot(shap_values, X_transformed_df, show=False)
    plt.title("SHAP Feature Impact (Beeswarm Plot)", fontsize=16)
    plt.tight_layout()
    beeswarm_plot_path = output_path / "shap_summary_beeswarm.png"
    plt.savefig(beeswarm_plot_path)
    plt.close()
    logging.info(f"✅ Beeswarm plot saved to '{beeswarm_plot_path}'")
    
    logging.info("--- SHAP Plot Generation Complete ---")


if __name__ == '__main__':
    DATA_PATH = 'data/processed/cleaned_superstore_data.parquet'
    MODEL_PATH = 'models/final_v3_profit_forecaster.joblib'
    OUTPUT_DIR = 'reports/final_shap_plots2'
    
    generate_and_save_shap_plots(DATA_PATH, MODEL_PATH, OUTPUT_DIR)