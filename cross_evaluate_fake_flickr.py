import os
import sys
import pickle
import subprocess
import warnings
import csv
import argparse
from pathlib import Path

# Add external paths
sys.path.append("/home/marek/FakeFlickr/detection/scripts")
sys.path.append("/home/marek/projects/fake_flickr_sota/UniversalFakeDetect")

from resnet50_wandb_pipeline.config import PipelineConfig
from resnet50_wandb_pipeline.data import prepare_experiments

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Cross Evaluate UFD on FakeFlickr datasets")
    parser.add_argument("--checkpoints_dir", default="checkpoints/fake_flickr_train", type=str, help="Directory containing trained models")
    parser.add_argument("--filter_train", default="", type=str, help="Substring filter for trained models to evaluate")
    parser.add_argument("--filter_test", default="", type=str, help="Substring filter for test datasets to evaluate on")
    args = parser.parse_args()

    # Configuration
    python_exe = "./.venv/bin/python3"
    arch = "CLIP:ViT-L/14"
    result_folder = "results/fake_flickr_cross_eval"
    
    # Path for temporary pickles to pass to validate.py
    pickle_root = os.path.join(result_folder, "data_lists")
    os.makedirs(pickle_root, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)
    
    # Load fake_flickr test datasets
    config = PipelineConfig(architecture="clip")
    print("Preparing test datasets from fake_flickr pipeline...")
    experiments = prepare_experiments(config)
    test_domains = list(experiments.keys())

    if args.filter_test:
        test_domains = [d for d in test_domains if args.filter_test in d]

    # Find trained models
    trained_models = []
    if os.path.exists(args.checkpoints_dir):
        for d in os.listdir(args.checkpoints_dir):
            ckpt_path = os.path.join(args.checkpoints_dir, d, "model_epoch_best.pth")
            if os.path.exists(ckpt_path):
                if args.filter_train == "" or args.filter_train in d:
                    trained_models.append((d, ckpt_path))

    if not trained_models:
        print("No trained models found to evaluate. Have you run the training script yet?")
        return

    print(f"Found {len(trained_models)} trained models.")
    print(f"Evaluating each against {len(test_domains)} test domains.")

    global_csv_file = os.path.join(result_folder, 'cross_eval_results.csv')
    file_exists = os.path.isfile(global_csv_file)

    for train_domain, ckpt_path in trained_models:
        print(f"\n" + "="*50)
        print(f"EVALUATING MODEL TRAINED ON: {train_domain}")
        print("="*50)
        
        for test_domain in test_domains:
            print(f"\n--- Testing on domain: {test_domain} ---")
            test_dataset = experiments[test_domain].datasets["test"]
            
            real_files = [f for f, l in zip(test_dataset.file_paths, test_dataset.labels) if l == 0]
            fake_files = [f for f, l in zip(test_dataset.file_paths, test_dataset.labels) if l == 1]
            test_count = min(len(real_files), len(fake_files))
            real_files = real_files[:test_count]
            fake_files = fake_files[:test_count]
            
            exp_test_pickle_root = os.path.join(pickle_root, test_domain)
            os.makedirs(exp_test_pickle_root, exist_ok=True)
            
            test_real_pickle = os.path.join(exp_test_pickle_root, "test_real.pickle")
            test_fake_pickle = os.path.join(exp_test_pickle_root, "test_fake.pickle")
            
            with open(test_real_pickle, 'wb') as f: pickle.dump(real_files, f)
            with open(test_fake_pickle, 'wb') as f: pickle.dump(fake_files, f)
            
            exp_result_dir = os.path.join(result_folder, "runs", train_domain, test_domain)
            
            cmd_eval = [
                python_exe, "validate.py",
                f"--arch={arch}",
                f"--ckpt={ckpt_path}",
                f"--result_folder={exp_result_dir}",
                f"--real_path={test_real_pickle}",
                f"--fake_path={test_fake_pickle}",
                "--data_mode=ours",
                f"--max_sample={test_count}"
            ]
            subprocess.run(cmd_eval)
            
            # Consolidate CSV results
            csv_src = os.path.join(exp_result_dir, 'results.csv')
            if os.path.exists(csv_src):
                with open(csv_src, 'r') as f_src:
                    reader = csv.DictReader(f_src)
                    rows = list(reader)
                    for row in rows:
                        row['train_domain'] = train_domain
                        row['test_domain'] = test_domain
                        if 'domain' in row:
                            del row['domain']
                        
                    with open(global_csv_file, 'a', newline='') as f_dst:
                        fieldnames = ['train_domain', 'test_domain'] + [f for f in reader.fieldnames if f not in ['train_domain', 'test_domain', 'domain']]
                        writer = csv.DictWriter(f_dst, fieldnames=fieldnames)
                        if not file_exists:
                            writer.writeheader()
                            file_exists = True
                        writer.writerows(rows)

    print(f"\nCross-evaluation complete.")
    print(f"Final results saved to: {global_csv_file}")

if __name__ == "__main__":
    main()
