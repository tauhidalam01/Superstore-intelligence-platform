import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging
from itertools import combinations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_environment(output_dir):
    sns.set_theme(style="whitegrid")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "1_general_health").mkdir(exist_ok=True)
    (output_path / "2_specialized_analytics").mkdir(exist_ok=True)
    logging.info(f"Report artifacts will be saved to: {output_path}")
    return output_path

def load_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        logging.info(f"Data loaded from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"File not found at {file_path}.")
        return None
def engineer_all_features(df):
    logging.info("Engineering full suite of temporary and advanced features...")
    temp_df = df.copy()

    if 'Quantity' not in temp_df.columns: temp_df['Quantity'] = 1
    if 'Sales Price' in temp_df.columns:
        if 'Total Sales' not in temp_df.columns: temp_df['Total Sales'] = temp_df['Sales Price'] * temp_df['Quantity']
    if 'Profit' in temp_df.columns:
        if 'Total Profit' not in temp_df.columns: temp_df['Total Profit'] = temp_df['Profit'] * temp_df['Quantity']
    
    # Engineer Profit_Margin early so it can be used by other features
    if 'Total Sales' in temp_df.columns and 'Total Profit' in temp_df.columns:
        if 'Profit_Margin' not in temp_df.columns: 
            temp_df['Profit_Margin'] = np.where(temp_df['Total Sales'] > 0, temp_df['Total Profit'] / temp_df['Total Sales'], 0)
    else:
        # If we can't calculate profit margin, we can't do the advanced features
        logging.warning("Cannot calculate Profit_Margin, skipping advanced feature engineering.")
        return temp_df # Return the partially engineered frame

    required_cols_for_advanced = {'Order Date', 'Customer ID', 'Postal Code', 'State', 'Sub-Category', 'Category'}
    if required_cols_for_advanced.issubset(temp_df.columns):
        temp_df['Order Date'] = pd.to_datetime(temp_df['Order Date'], errors='coerce')
        temp_df.dropna(subset=['Order Date'], inplace=True)
        
        temp_df['month_of_year'] = temp_df['Order Date'].dt.month
        temp_df['first_order_date'] = temp_df.groupby('Customer ID')['Order Date'].transform('min')
        temp_df['days_since_first_order'] = (temp_df['Order Date'] - temp_df['first_order_date']).dt.days

        temp_df['postal_code_profitability'] = temp_df.groupby('Postal Code')['Profit_Margin'].transform('mean')
        temp_df['state_sales_volume'] = temp_df.groupby('State')['Total Sales'].transform('sum')
        temp_df['subcategory_avg_margin'] = temp_df.groupby('Sub-Category')['Profit_Margin'].transform('mean')
        temp_df['category_avg_discount'] = temp_df.groupby('Category')['Discount'].transform('mean')
        temp_df['customer_avg_order_size'] = temp_df.groupby('Customer ID')['Total Sales'].transform('mean')

        # Fill NaNs that might result from the transform/map operations
        temp_df.fillna(temp_df.median(numeric_only=True), inplace=True)
        logging.info("Advanced features created successfully.")
    else:
        logging.warning("Missing columns required for advanced features. Skipping.")
        
    return temp_df


def generate_general_health_report(df, output_path):
    sub_path = output_path / "1_general_health"
    logging.info(f"Generating general health report in '{sub_path}'...")
    with open(sub_path / "basic_profile.txt", "w", encoding='utf-8') as f:
        f.write(f"Shape: {df.shape}\n\nData Types:\n{df.dtypes}\n\n")
        missing = df.isnull().sum(); f.write(f"Missing Values:\n{missing[missing > 0]}\n\n")
        f.write(f"Duplicate Rows: {df.duplicated().sum()}\n")
    numerical_cols = df.select_dtypes(include=np.number).columns
    if not numerical_cols.empty:
        df[numerical_cols].describe().round(2).to_csv(sub_path / "numerical_summary.csv")
        for col in numerical_cols:
            plt.figure(figsize=(10, 6)); sns.histplot(df[col], kde=True, bins=50); plt.title(f'Distribution of {col}'); plt.savefig(sub_path / f"dist_{col}.png"); plt.close()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.empty:
        with open(sub_path / "categorical_summary.txt", "w", encoding='utf-8') as f:
            for col in categorical_cols:
                f.write(f"--- Summary for {col} ---\nUnique Values: {df[col].nunique()}\nTop 5:\n{df[col].value_counts().nlargest(5)}\n\n")
                if df[col].nunique() < 50:
                    plt.figure(figsize=(12, 7)); sns.countplot(y=df[col], order=df[col].value_counts().index[:25]); plt.title(f'Top 25 Counts for {col}'); plt.tight_layout(); plt.savefig(sub_path / f"count_{col}.png"); plt.close()
    logging.info("General health report complete.")

def generate_specialized_analytics_report(df, output_path):
    sub_path = output_path / "2_specialized_analytics"
    logging.info(f"Generating specialized analytics report in '{sub_path}'...")
    
    if {'Sales Price', 'Profit', 'Region', 'State', 'Segment', 'Category', 'Discount'}.issubset(df.columns):
        plt.figure(figsize=(12, 8)); sns.regplot(x='Sales Price', y='Profit', data=df, scatter_kws={'alpha':0.3}); plt.title('Sales vs. Profit'); plt.savefig(sub_path / "A_sales_vs_profit.png"); plt.close()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7)); df.groupby('Region')['Total Sales'].sum().sort_values().plot(kind='barh', ax=ax1, title='Total Sales by Region'); df.groupby('Region')['Total Profit'].sum().sort_values().plot(kind='barh', ax=ax2, title='Total Profit by Region'); plt.tight_layout(); plt.savefig(sub_path / "B_regional_performance.png"); plt.close()
        state_profit = df.groupby('State')['Total Profit'].sum().sort_values(ascending=False); fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8)); state_profit.head(10).plot(kind='barh', ax=ax1, title='Top 10 Profitable States'); state_profit.tail(10).plot(kind='barh', ax=ax2, title='Bottom 10 Profitable States'); plt.tight_layout(); plt.savefig(sub_path / "C_state_profitability.png"); plt.close()
        df.groupby(['Segment', 'Category'])['Total Sales'].sum().unstack().plot(kind='bar', stacked=True, figsize=(14,8), title='Sales by Segment and Category'); plt.ylabel('Total Sales'); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(sub_path / "D_segment_sales_by_category.png"); plt.close()
        plt.figure(figsize=(12, 8)); sns.regplot(x='Discount', y='Profit', data=df, scatter_kws={'alpha':0.2}, line_kws={'color': 'red'}); plt.axhline(0, color='grey', linestyle='--'); plt.title('Discount Impact on Profit'); plt.savefig(sub_path / "E_discount_impact.png"); plt.close()
        logging.info("Generated foundational business plots.")

    if 'Profit_Margin' in df.columns:
        plt.figure(figsize=(10, 6)); sns.histplot(df['Profit_Margin'], kde=True, bins=50); plt.title('Profit Margin Distribution'); plt.axvline(0, color='red', linestyle='--'); plt.savefig(sub_path / "F_profit_margin_dist.png"); plt.close()
        if 'Category' in df.columns:
            plt.figure(figsize=(12, 7)); sns.boxplot(y='Category', x='Profit_Margin', data=df); plt.title('Profit Margin by Category'); plt.axvline(0, color='red', linestyle='--'); plt.tight_layout(); plt.savefig(sub_path / "G_profit_margin_by_category.png"); plt.close()
        logging.info("Generated profit margin analysis.")
            
    if {'Customer ID', 'Total Sales', 'Total Profit', 'Order ID'}.issubset(df.columns):
        customer_summary = df.groupby('Customer ID').agg(total_sales=('Total Sales', 'sum'), total_profit=('Total Profit', 'sum'), order_count=('Order ID', 'nunique')).sort_values(by='total_profit', ascending=False)
        customer_summary.head(20).to_csv(sub_path / "H_top_20_customers_by_profit.csv")
        plt.figure(figsize=(12, 8)); sns.scatterplot(x='total_sales', y='total_profit', data=customer_summary, alpha=0.5); plt.title('Customer LTV (Sales vs. Profit)'); plt.axhline(0, color='red', linestyle='--'); plt.savefig(sub_path / "I_customer_ltv_scatter.png"); plt.close()
        logging.info("Generated customer-level analytics.")

    if {'Order Date', 'Total Sales', 'Total Profit'}.issubset(df.columns):
        logging.info("Generating Temporal Analysis...")
        df_temp = df.set_index('Order Date')
        monthly_trends = df_temp[['Total Sales', 'Total Profit']].resample('M').sum()
        monthly_trends.plot(kind='line', subplots=True, figsize=(14, 8), title='Monthly Sales and Profit Trends')
        plt.savefig(sub_path / "J_monthly_trends.png")
        plt.close()

    if {'Order ID', 'Sub-Category'}.issubset(df.columns):
        logging.info("Generating Market Basket Analysis...")
        transactions = df[df.duplicated('Order ID', keep=False)].groupby('Order ID')['Sub-Category'].apply(list)
        pairs = transactions.apply(lambda x: list(combinations(sorted(list(set(x))), 2)))
        pair_counts = pairs.explode().dropna().value_counts()
        pair_counts.head(20).to_csv(sub_path / "L_top_20_product_pairs.csv")
        
    if {'Product Name', 'Quantity', 'Total Profit'}.issubset(df.columns):
        logging.info("Generating Unit-Based Product Portfolio Analysis...")
        product_summary = df.groupby('Product Name').agg(Total_Quantity_Sold=('Quantity', 'sum'), Avg_Profit_Per_Unit=('Total Profit', 'mean')).dropna()
        product_summary = product_summary[product_summary['Total_Quantity_Sold'] > 0]
        if not product_summary.empty:
            volume_median = product_summary['Total_Quantity_Sold'].median()
            efficiency_median = product_summary['Avg_Profit_Per_Unit'].median()
            
            def assign_quadrant(row):
                is_high_volume = row['Total_Quantity_Sold'] >= volume_median
                is_high_profit = row['Avg_Profit_Per_Unit'] >= efficiency_median
                if is_high_volume and is_high_profit:
                    return 'Stars'
                elif is_high_volume and not is_high_profit:
                    return 'Cash Cows'
                elif not is_high_volume and is_high_profit:
                    return 'Question Marks'
                else:
                    return 'Dogs'
            
            product_summary['Quadrant'] = product_summary.apply(assign_quadrant, axis=1)
            
            plt.figure(figsize=(16, 10))
            sns.scatterplot(x='Total_Quantity_Sold', y='Avg_Profit_Per_Unit', hue='Quadrant', size='Total_Quantity_Sold', sizes=(20, 400), alpha=0.7, data=product_summary)
            plt.xscale('log')
            plt.axhline(efficiency_median, color='grey', linestyle='--')
            plt.axvline(volume_median, color='grey', linestyle='--')
            plt.title('Product Portfolio Analysis')
            plt.savefig(sub_path / "M_product_portfolio_matrix.png")
            plt.close()
            
            product_summary.groupby('Quadrant').size().to_csv(sub_path / "M_product_portfolio_summary.csv")

    logging.info("Specialized analytics report complete.")
def generate_advanced_feature_analysis(df, output_path):
    sub_path = output_path / "3_advanced_feature_eda"
    sub_path.mkdir(exist_ok=True)
    logging.info(f"Generating advanced feature analysis in '{sub_path}'...")
    
    advanced_features = [
        'month_of_year', 'days_since_first_order', 'postal_code_profitability', 
        'state_sales_volume', 'subcategory_avg_margin', 'category_avg_discount', 
        'customer_avg_order_size'
    ]
    
    existing_features = [f for f in advanced_features if f in df.columns]
    if not existing_features:
        logging.warning("No advanced features found to analyze.")
        return
        
    # Create correlation heatmap of new features vs. Profit Margin
    plt.figure(figsize=(12, 10))
    corr = df[existing_features + ['Profit_Margin']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation of Advanced Features with Profit Margin')
    plt.tight_layout()
    plt.savefig(sub_path / "advanced_features_correlation.png")
    plt.close()

    # Visualize key relationships
    if 'customer_avg_order_size' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='customer_avg_order_size', y='Profit_Margin', data=df.sample(n=2000, random_state=1), alpha=0.3)
        plt.title('Customer Avg Order Size vs. Transaction Profit Margin')
        plt.xscale('log')
        plt.savefig(sub_path / "customer_avg_order_size_vs_margin.png")
        plt.close()

    if 'subcategory_avg_margin' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='subcategory_avg_margin', y='Profit_Margin', data=df.sample(n=2000, random_state=1), alpha=0.3)
        plt.title('Sub-Category Avg Margin vs. Transaction Profit Margin')
        plt.axhline(0, color='grey', linestyle='--'); plt.axvline(0, color='grey', linestyle='--')
        plt.savefig(sub_path / "subcategory_margin_vs_margin.png")
        plt.close()

    logging.info("Advanced feature analysis complete.")


def main():
    parser = argparse.ArgumentParser(description="Generate a Comprehensive EDA Report.")
    parser.add_argument("--input-path", type=str, default='data/raw/SuperStore_Dataset.csv')
    parser.add_argument("--output-dir", type=str, default='reports/comprehensive_eda_report')
    args = parser.parse_args()
    
    output_path = setup_environment(args.output_dir)
    df = load_data(args.input_path)
    

    
    if df is not None:
        logging.info("--- Starting Comprehensive EDA Report Generation ---")
        df_engineered = engineer_all_features(df)
        generate_general_health_report(df, output_path)
        generate_specialized_analytics_report(df_engineered, output_path)
        generate_advanced_feature_analysis(df_engineered, output_path)
        logging.info("--- Report Generation Complete ---")

if __name__ == '__main__':
    main()