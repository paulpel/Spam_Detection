import argparse
from data_process_and_train import DataPreprocessor, ModelTrainer, ResultsVisualizer
from bert_features import process_and_extract_features
from create_concept_driftprepare_mix_dataset,  prepare_mix_dataset

def main(original_data_path, output_folder):
    # Step 1: Train classifiers on original CSV data
    print("Training initial classifiers on the original data...")
    original_data_preprocessor = DataPreprocessor(original_data_path, is_bert=False)
    original_data = original_data_preprocessor.preprocess()
    original_model_trainer = ModelTrainer(original_data, is_bert=False)
    original_model_trainer.train_and_evaluate()
    original_results_visualizer = ResultsVisualizer(original_model_trainer.get_results())
    original_results_visualizer.display_results()

    # Step 2: Generate concept drift data (Placeholder)
    print("Generating concept drift data...")
    drift_data_path = prepare_mix_dataset(output_folder)

    # Step 3: Retrain classifiers on the drifted data
    print("Retraining classifiers on the drifted data...")
    drift_data_preprocessor = DataPreprocessor(drift_data_path, is_bert=False)
    drift_data = drift_data_preprocessor.preprocess()
    drift_model_trainer = ModelTrainer(drift_data, is_bert=False)
    drift_model_trainer.train_and_evaluate()
    drift_results_visualizer = ResultsVisualizer(drift_model_trainer.get_results())
    drift_results_visualizer.display_results()

    # Step 4: Extract BERT features from both original and drifted data
    print("Extracting BERT features...")
    bert_features_original_path = f"{output_folder}/bert_encoded_features_original.csv"
    process_and_extract_features(original_data_path, bert_features_original_path)
    bert_features_drift_path = f"{output_folder}/bert_encoded_features_drift.csv"
    process_and_extract_features(drift_data_path, bert_features_drift_path)

    # Step 5: Train classifiers on BERT features and visualize results
    print("Training classifiers on BERT features and visualizing results...")
    bert_original_data_preprocessor = DataPreprocessor(bert_features_original_path, is_bert=True)
    bert_original_data = bert_original_data_preprocessor.preprocess()
    bert_original_model_trainer = ModelTrainer(bert_original_data, is_bert=True)
    bert_original_model_trainer.train_and_evaluate()
    bert_original_results_visualizer = ResultsVisualizer(bert_original_model_trainer.get_results())
    bert_original_results_visualizer.display_results()

    bert_drift_data_preprocessor = DataPreprocessor(bert_features_drift_path, is_bert=True)
    bert_drift_data = bert_drift_data_preprocessor.preprocess()
    bert_drift_model_trainer = ModelTrainer(bert_drift_data, is_bert=True)
    bert_drift_model_trainer.train_and_evaluate()
    bert_drift_results_visualizer = ResultsVisualizer(bert_drift_model_trainer.get_results())
    bert_drift_results_visualizer.display_results()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main training and evaluation loop for research on concept drift.")
    parser.add_argument("original_data_path", type=str, help="Path to the original dataset CSV file.")
    parser.add_argument("output_folder", type=str, help="Folder where output files will be stored.")
    args = parser.parse_args()
    main(args.original_data_path, args.output_folder)

