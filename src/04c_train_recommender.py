import pandas as pd
import numpy as np
from itertools import combinations
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_product_features(df):
    product_features = df.groupby('Product Name').agg(
        Category=('Category', 'first'),
        SubCategory=('Sub-Category', 'first'),
        AvgSalesPrice=('Sales Price', 'mean')
    ).reset_index()
    return product_features

def generate_training_pairs(df, product_features):
    transactions = df[df.duplicated('Order ID', keep=False)].groupby('Order ID')['Product Name'].apply(list)
    positive_pairs = transactions.apply(lambda x: list(combinations(sorted(list(set(x))), 2))).explode().dropna()
    
    if positive_pairs.empty:
        return None
        
    positive_pairs_df = pd.DataFrame(positive_pairs.tolist(), columns=['Product_A', 'Product_B'])
    positive_pairs_df['Purchased_Together'] = 1
    
    num_neg_samples = len(positive_pairs_df) * 2
    all_products = product_features['Product Name'].unique()
    neg_pairs_A = np.random.choice(all_products, size=num_neg_samples)
    neg_pairs_B = np.random.choice(all_products, size=num_neg_samples)
    
    negative_pairs_df = pd.DataFrame({'Product_A': neg_pairs_A, 'Product_B': neg_pairs_B})
    negative_pairs_df = negative_pairs_df[negative_pairs_df['Product_A'] != negative_pairs_df['Product_B']]
    negative_pairs_df['Purchased_Together'] = 0
    
    merged = pd.merge(negative_pairs_df, positive_pairs_df.drop('Purchased_Together', axis=1), on=['Product_A', 'Product_B'], how='left', indicator=True)
    negative_pairs_df = merged[merged['_merge'] == 'left_only'].drop('_merge', axis=1)
    
    training_df = pd.concat([positive_pairs_df, negative_pairs_df], ignore_index=True)
    return training_df

def build_final_dataset(pairs_df, product_features):
    df = pairs_df.merge(product_features, left_on='Product_A', right_on='Product Name', how='left')
    df.rename(columns={'Category': 'A_Category', 'SubCategory': 'A_SubCategory', 'AvgSalesPrice': 'A_AvgSalesPrice'}, inplace=True)
    df.drop('Product Name', axis=1, inplace=True)
    
    df = df.merge(product_features, left_on='Product_B', right_on='Product Name', how='left')
    df.rename(columns={'Category': 'B_Category', 'SubCategory': 'B_SubCategory', 'AvgSalesPrice': 'B_AvgSalesPrice'}, inplace=True)
    df.drop('Product Name', axis=1, inplace=True)
    
    df.dropna(inplace=True)
    return df

def train_mba_classifier(df, output_path):
    logging.info("--- Training Predictive Market Basket Classifier ---")
    features = ['A_Category', 'A_SubCategory', 'A_AvgSalesPrice', 'B_Category', 'B_SubCategory', 'B_AvgSalesPrice']
    target = 'Purchased_Together'
    
    X, y = df[features], df[target]
    
    numerical_features = ['A_AvgSalesPrice', 'B_AvgSalesPrice']
    categorical_features = ['A_Category', 'A_SubCategory', 'B_Category', 'B_SubCategory']
    
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numerical_features), ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1))])
    
    param_grid = {'classifier__n_estimators': [200], 'classifier__max_depth': [10], 'classifier__learning_rate': [0.1]}
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1)
    
    logging.info("Tuning and training final MBA model...")
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    logging.info(f"Tuned MBA Classifier Best CV Score: {grid_search.best_score_:.4f}")

    model_path = output_path / "final_predictive_mba_classifier.joblib"
    joblib.dump(best_model, model_path)
    logging.info(f"✅ Final Predictive MBA Classifier saved to '{model_path}'")

def main():
    parser = argparse.ArgumentParser(description="Train the Predictive Market Basket Analysis model.")
    parser.add_argument("--input-path", default='data/processed/cleaned_superstore_data.parquet')
    parser.add_argument("--output-dir", default='models')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    main_df = pd.read_parquet(args.input_path)
    product_features = create_product_features(main_df)
    training_pairs = generate_training_pairs(main_df, product_features)
    
    if training_pairs is not None:
        final_dataset = build_final_dataset(training_pairs, product_features)
        train_mba_classifier(final_dataset, output_dir)
    else:
        logging.error("Could not proceed with MBA training as no positive pairs were generated.")

if __name__ == '__main__':
    main()