import os
import sys
import pickle
import subprocess
import warnings
import csv
from pathlib import Path

# Add external paths
sys.path.append("/home/marek/FakeFlickr/detection/scripts")
sys.path.append("/home/marek/projects/fake_flickr_sota/UniversalFakeDetect")

from resnet50_wandb_pipeline.config import PipelineConfig
from resnet50_wandb_pipeline.data import prepare_experiments

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate UFD on FakeFlickr datasets")
    parser.add_argument("--filter", default="real_rescaled", type=str, help="Substring filter for experiment names")
    parser.add_argument("--list", action="store_true", help="List all available experiments and exit")
    parser.add_argument("--include", type=str, help="Comma-separated list of experiment names to include")
    args = parser.parse_args()

    # Configuration
    python_exe = "./.venv/bin/python3"
    arch = "CLIP:ViT-L/14"
    ckpt_path = "pretrained_weights/fc_weights.pth"
    result_folder = "results/fake_flickr_validation"
    
    # Path for temporary pickles to pass to validate.py
    pickle_root = os.path.join(result_folder, "data_lists")
    os.makedirs(pickle_root, exist_ok=True)
    
    # Load fake_flickr datasets (test splits)
    config = PipelineConfig(architecture="clip")
    print("Preparing test datasets from fake_flickr pipeline...")
    experiments = prepare_experiments(config)

    all_exp_names = list(experiments.keys())
    
    if args.list:
        print("\nAvailable experiments:")
        for i, name in enumerate(all_exp_names):
            print(f"{i+1}. {name}")
        return

    # Filtering logic
    selected_exp_names = all_exp_names
    if args.include:
        include_list = [s.strip() for s in args.include.split(",")]
        selected_exp_names = [name for name in selected_exp_names if name in include_list]
    
    if args.filter:
        selected_exp_names = [name for name in selected_exp_names if args.filter in name]

    if not selected_exp_names:
        print(f"No experiments matched the criteria.")
        return

    print(f"Selected {len(selected_exp_names)} experiments for evaluation.")

    # Global CSV result file
    global_csv_file = os.path.join(result_folder, 'final_results.csv')
    
    # If we are running ALL experiments, it's safe to clear the file.
    # If we are running a subset, we probably want to append or keep existing.
    # However, to avoid duplicates if re-running, it's tricky.
    # For now, let's only remove if we are running everything and no filter is applied.
    if not args.filter and not args.include and os.path.exists(global_csv_file):
        os.remove(global_csv_file)

    for exp_name in selected_exp_names:
        exp = experiments[exp_name]
        print(f"\n" + "="*50)
        print(f"Evaluating: {exp_name}")
        print("="*50)
        
        test_dataset = exp.datasets["test"]
        
        # Split into real and fake lists for validate.py
        real_files = [f for f, l in zip(test_dataset.file_paths, test_dataset.labels) if l == 0]
        fake_files = [f for f, l in zip(test_dataset.file_paths, test_dataset.labels) if l == 1]
        
        # validate.py asserts len(real) == len(fake), so we balance them
        count = min(len(real_files), len(fake_files))
        real_files = real_files[:count]
        fake_files = fake_files[:count]
        
        # Save as pickles
        real_pickle = os.path.join(pickle_root, f"{exp_name}_real.pickle")
        fake_pickle = os.path.join(pickle_root, f"{exp_name}_fake.pickle")
        
        with open(real_pickle, 'wb') as f: pickle.dump(real_files, f)
        with open(fake_pickle, 'wb') as f: pickle.dump(fake_files, f)
        
        # Each call gets its own subfolder to avoid folder wiping
        exp_result_dir = os.path.join(result_folder, "runs", exp_name)
        
        cmd = [
            python_exe, "validate.py",
            f"--arch={arch}",
            f"--ckpt={ckpt_path}",
            f"--result_folder={exp_result_dir}",
            f"--real_path={real_pickle}",
            f"--fake_path={fake_pickle}",
            "--data_mode=ours",
            f"--max_sample={count}"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd)
        
        # Consolidate CSV results
        csv_src = os.path.join(exp_result_dir, 'results.csv')
        if os.path.exists(csv_src):
            with open(csv_src, 'r') as f_src:
                reader = csv.DictReader(f_src)
                rows = list(reader)
                for row in rows:
                    row['domain'] = exp_name
                    
                file_exists = os.path.isfile(global_csv_file)
                with open(global_csv_file, 'a', newline='') as f_dst:
                    writer = csv.DictWriter(f_dst, fieldnames=reader.fieldnames)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerows(rows)

    print(f"\nEvaluation complete.")
    print(f"Final results saved to: {global_csv_file}")

if __name__ == "__main__":
    main()
