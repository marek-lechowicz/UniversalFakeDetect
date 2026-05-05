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
    parser = argparse.ArgumentParser(description="Train UFD on FakeFlickr datasets")
    parser.add_argument("--filter", default="", type=str, help="Substring filter for experiment names")
    parser.add_argument("--list", action="store_true", help="List all available experiments and exit")
    parser.add_argument("--include", type=str, help="Comma-separated list of experiment names to include")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation on the test set after training")
    args = parser.parse_args()

    # Configuration
    python_exe = "./.venv/bin/python3"
    arch = "CLIP:ViT-L/14"
    ckpt_path = "pretrained_weights/fc_weights.pth"
    checkpoints_dir = "checkpoints/fake_flickr_train"
    result_folder = "results/fake_flickr_train_eval"
    
    # Path for temporary pickles to pass to train.py
    pickle_root = os.path.join(checkpoints_dir, "data_lists")
    os.makedirs(pickle_root, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)
    
    # Load fake_flickr datasets (train/val/test splits)
    config = PipelineConfig(architecture="clip")
    print("Preparing datasets from fake_flickr pipeline...")
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

    print(f"Selected {len(selected_exp_names)} experiments for training.")

    global_csv_file = os.path.join(result_folder, 'final_results.csv')
    if not args.filter and not args.include and os.path.exists(global_csv_file):
        os.remove(global_csv_file)

    for exp_name in selected_exp_names:
        exp = experiments[exp_name]
        print(f"\n" + "="*50)
        print(f"Training on: {exp_name}")
        print("="*50)
        
        train_dataset = exp.datasets["train"]
        val_dataset = exp.datasets["val"]
        test_dataset = exp.datasets["test"]
        
        def save_pickles(dataset, split_name, exp_pickle_root):
            real_files = [f for f, l in zip(dataset.file_paths, dataset.labels) if l == 0]
            fake_files = [f for f, l in zip(dataset.file_paths, dataset.labels) if l == 1]
            
            # Balance
            count = min(len(real_files), len(fake_files))
            real_files = real_files[:count]
            fake_files = fake_files[:count]
            
            real_dir = os.path.join(exp_pickle_root, "real")
            fake_dir = os.path.join(exp_pickle_root, "fake")
            os.makedirs(real_dir, exist_ok=True)
            os.makedirs(fake_dir, exist_ok=True)
            
            with open(os.path.join(real_dir, f"{split_name}.pickle"), 'wb') as f: pickle.dump(real_files, f)
            with open(os.path.join(fake_dir, f"{split_name}.pickle"), 'wb') as f: pickle.dump(fake_files, f)
            
            return real_dir, fake_dir, count
        
        exp_pickle_root = os.path.join(pickle_root, exp_name)
        
        # Save train pickles
        train_real_dir, train_fake_dir, train_count = save_pickles(train_dataset, "train", exp_pickle_root)
        
        # Save val pickles
        val_real_dir, val_fake_dir, val_count = save_pickles(val_dataset, "val", exp_pickle_root)
        
        print(f"Balanced Train Set: {train_count*2} images")
        print(f"Balanced Val Set: {val_count*2} images")
        
        # We assume real_dir and fake_dir are the same for train and val because save_pickles uses exp_pickle_root/real and exp_pickle_root/fake
        # and just saves train.pickle and val.pickle in them. This perfectly matches UFD's `ours` mode expectations.
        real_list_path = train_real_dir
        fake_list_path = train_fake_dir
        
        cmd_train = [
            python_exe, "train.py",
            f"--name={exp_name}",
            f"--checkpoints_dir={checkpoints_dir}",
            f"--arch={arch}",
            f"--ckpt={ckpt_path}",
            "--fix_backbone",
            f"--data_mode=ours",
            f"--real_list_path={real_list_path}",
            f"--fake_list_path={fake_list_path}",
            f"--batch_size={args.batch_size}",
            f"--niter={args.epochs}",
            f"--lr={args.lr}"
        ]
        
        print(f"Running train command: {' '.join(cmd_train)}")
        subprocess.run(cmd_train)
        
        if not args.skip_eval:
            print(f"\n--- Evaluating the best model on test set for {exp_name} ---")
            best_ckpt = os.path.join(checkpoints_dir, exp_name, "model_epoch_best.pth")
            if not os.path.exists(best_ckpt):
                print(f"Warning: {best_ckpt} not found! Did training fail?")
                continue
                
            # Prepare test pickles
            exp_test_pickle_root = os.path.join(pickle_root, exp_name, "test_pickles")
            os.makedirs(exp_test_pickle_root, exist_ok=True)
            
            real_files = [f for f, l in zip(test_dataset.file_paths, test_dataset.labels) if l == 0]
            fake_files = [f for f, l in zip(test_dataset.file_paths, test_dataset.labels) if l == 1]
            test_count = min(len(real_files), len(fake_files))
            real_files = real_files[:test_count]
            fake_files = fake_files[:test_count]
            
            test_real_pickle = os.path.join(exp_test_pickle_root, "test_real.pickle")
            test_fake_pickle = os.path.join(exp_test_pickle_root, "test_fake.pickle")
            
            with open(test_real_pickle, 'wb') as f: pickle.dump(real_files, f)
            with open(test_fake_pickle, 'wb') as f: pickle.dump(fake_files, f)
            
            exp_result_dir = os.path.join(result_folder, "runs", exp_name)
            
            cmd_eval = [
                python_exe, "validate.py",
                f"--arch={arch}",
                f"--ckpt={best_ckpt}",
                f"--result_folder={exp_result_dir}",
                f"--real_path={test_real_pickle}",
                f"--fake_path={test_fake_pickle}",
                "--data_mode=ours",
                f"--max_sample={test_count}"
            ]
            print(f"Running eval command: {' '.join(cmd_eval)}")
            subprocess.run(cmd_eval)
            
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

    print(f"\nPipeline complete.")

if __name__ == "__main__":
    main()
