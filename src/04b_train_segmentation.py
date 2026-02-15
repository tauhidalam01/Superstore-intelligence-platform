import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_rfm_dataframe(df):
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df.dropna(subset=['Order Date'], inplace=True)
    snapshot_date = df['Order Date'].max() + pd.Timedelta(days=1)
    
    rfm_df = df.groupby('Customer ID').agg(
        Recency=('Order Date', lambda x: (snapshot_date - x.max()).days),
        Frequency=('Order ID', 'nunique'),
        MonetaryValue=('Total Sales', 'sum')
    )
    return rfm_df

def train_autoencoder(rfm_df, output_path):
    logging.info("--- Training Advanced Segmentation Autoencoder ---")
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df)
    
    input_dim = rfm_scaled.shape[1]
    encoding_dim = 2

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(16, activation='relu')(input_layer)
    encoder = Dense(8, activation='relu')(encoder)
    encoder = Dense(encoding_dim, activation='relu')(encoder)
    decoder = Dense(8, activation='relu')(encoder)
    decoder = Dense(16, activation='relu')(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    encoder_model = Model(inputs=input_layer, outputs=encoder)
    
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    logging.info("Training the autoencoder model...")
    autoencoder.fit(rfm_scaled, rfm_scaled, epochs=50, batch_size=16, shuffle=True, verbose=1)

    encoder_path = output_path / "customer_encoder_model.keras"
    scaler_path = output_path / "rfm_scaler.joblib"
    
    encoder_model.save(encoder_path)
    joblib.dump(scaler, scaler_path)
    
    logging.info(f"✅ Encoder model saved to '{encoder_path}'")
    logging.info(f"✅ RFM scaler saved to '{scaler_path}'")

def main():
    parser = argparse.ArgumentParser(description="Train the advanced segmentation autoencoder.")
    parser.add_argument("--input-path", default='data/processed/cleaned_superstore_data.parquet')
    parser.add_argument("--output-dir", default='models')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    df = pd.read_parquet(args.input_path)
    rfm_df = create_rfm_dataframe(df)
    train_autoencoder(rfm_df, output_dir)
    
if __name__ == '__main__':
    main()