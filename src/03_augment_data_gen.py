import pandas as pd
import logging
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_synthesis_pipeline(real_data_path, synthetic_output_path, synthesizer_save_path, num_rows, epochs):
    logging.info("--- Starting Full Synthetic Data Pipeline ---")

    try:
        real_data = pd.read_parquet(real_data_path)
    except FileNotFoundError:
        logging.error(f"Input data not found at '{real_data_path}'.")
        return

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_data)
    python_dict = metadata.to_dict()
    for col, details in python_dict['columns'].items():
        if 'datetime' in details['sdtype']:
            details['datetime_format'] = '%Y-%m-%d %H:%M:%S'
    metadata.load_from_dict(python_dict)

    synthesizer = CTGANSynthesizer(metadata, epochs=epochs, verbose=True)

    logging.info("Step 1: Training the CTGAN synthesizer...")
    synthesizer.fit(real_data)

    logging.info(f"Step 2: Generating {num_rows} synthetic data rows...")
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    synthetic_data.to_parquet(synthetic_output_path, index=False)
    
    logging.info("Step 3: Performing RVS Validation...")
    quality_report = evaluate_quality(real_data, synthetic_data, metadata)
    overall_score = quality_report.get_score()
    logging.info(f"--- Overall Data Quality Score: {overall_score:.1%} ---")
    
    if overall_score >= 0.75:
        logging.info("Quality score is acceptable. Saving the synthesizer model.")
        synthesizer.save(filepath=synthesizer_save_path)
    else:
        logging.warning("Quality score is below threshold. Model not saved.")
    
    logging.info("--- Synthetic Data Pipeline Complete ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic data using CTGAN.")
    parser.add_argument("--input-path", default='data/processed/cleaned_superstore_data.parquet')
    parser.add_argument("--output-path", default='data/processed/synthetic_base_transactions.parquet')
    parser.add_argument("--model-save-path", default='models/ctgan_synthesizer.pkl')
    parser.add_argument("--num-rows", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    run_synthesis_pipeline(args.input_path, args.output_path, args.model_save_path, args.num_rows, args.epochs)