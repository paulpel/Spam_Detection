import subprocess
import argparse

def run_bert_feature_extraction(data_path, output_path):
    """
    Runs the BERT feature extraction script.
    """
    subprocess.run(['python', 'bert_feature_s.py', '--input', data_path, '--output', output_path])

def run_model_training(data_path, is_bert):
    """
    Runs the model training script.
    """
    bert_flag = '--bert' if is_bert else ''
    subprocess.run(['python', 'train.py', data_path, bert_flag])

def main():
    parser = argparse.ArgumentParser(description='Run BERT feature extraction and model training')
    parser.add_argument('data_path', type=str, help='Path to the original dataset')
    parser.add_argument('--use_bert', action='store_true', help='Flag to use BERT for feature extraction')
    args = parser.parse_args()

    if args.use_bert:
        # Specify the output path for BERT features
        output_path = args.data_path.replace('.csv', '_bert_features.csv')
        run_bert_feature_extraction(args.data_path, output_path)
        # Use the generated BERT features for model training
        run_model_training(output_path, is_bert=True)
    else:
        # Directly use the original data for model training
        run_model_training(args.data_path, is_bert=False)

if __name__ == "__main__":
    main()
