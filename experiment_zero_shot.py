import sys
import os
import torch
import warnings
import csv
import json
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef

# Add parent path to import from the reference pipeline
sys.path.append("/home/marek/FakeFlickr/detection/scripts")
# Add current directory to import from UniversalFakeDetect
sys.path.append("/home/marek/projects/fake_flickr_sota/UniversalFakeDetect")

from resnet50_wandb_pipeline.config import PipelineConfig
from resnet50_wandb_pipeline.data import prepare_experiments
from models import get_model

# Ignore deprecation warnings for clean output
warnings.filterwarnings("ignore")

def evaluate_on_UFD(model, test_loader, device):
    """
    Evaluates Universal Fake Detect model on given dataloader.
    Returns both aggregated metrics and per-sample results.
    """
    model.eval()
    all_labels = []
    all_preds_prob = []
    
    # We assume shuffle=False so we can correlate with dataset.file_paths
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
            inputs = inputs.to(device)
            # UFD model returns logits natively. We apply sigmoid.
            outputs = model(inputs).sigmoid()
            all_preds_prob.extend(outputs.flatten().cpu().numpy().tolist())
            all_labels.extend(labels.flatten().cpu().numpy().tolist())

    preds = [(p > 0.5) for p in all_preds_prob]
    
    total_acc = accuracy_score(all_labels, preds)
    precision = precision_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)
    mcc = matthews_corrcoef(all_labels, preds)
    
    try:
        avg_precision = average_precision_score(all_labels, all_preds_prob)
        auc_roc = roc_auc_score(all_labels, all_preds_prob)
    except ValueError:
        avg_precision = 0.0
        auc_roc = 0.5

    metrics = {
        "test_acc": total_acc,
        "test_auc": auc_roc,
        "precision": precision,
        "recall": recall,
        "average_precision": avg_precision,
        "mcc": mcc,
    }
    
    details = []
    for i in range(len(all_labels)):
        details.append({
            "label": all_labels[i],
            "pred_prob": all_preds_prob[i],
            "pred_class": int(preds[i])
        })
        
    return metrics, details


import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate UFD on FakeFlickr datasets (zero-shot)")
    parser.add_argument("--filter", type=str, help="Substring filter for experiment names")
    parser.add_argument("--list", action="store_true", help="List all available experiments and exit")
    parser.add_argument("--include", type=str, help="Comma-separated list of experiment names to include")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Universal Fake Detect Model
    arch = "CLIP:ViT-L/14"
    ckpt_path = "/home/marek/projects/fake_flickr_sota/UniversalFakeDetect/pretrained_weights/fc_weights.pth"
    
    # Configure our pipeline identically to the reference script
    config = PipelineConfig(architecture="clip")
    
    print("Preparing test datasets from resnet50_wandb_pipeline (only checking test splits)...")
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

    print(f"Loading {arch} from {ckpt_path}")
    ufd_model = get_model(arch)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    # According to validate.py, we load the fc state dict:
    ufd_model.fc.load_state_dict(state_dict)
    ufd_model.to(device)
    # Ensure gradients are off for all parameters
    for param in ufd_model.parameters():
        param.requires_grad = False

    # Setup results directories
    results_base_dir = Path("/home/marek/projects/fake_flickr_sota/UniversalFakeDetect/results")
    details_dir = results_base_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)

    summary_file = results_base_dir / "summary.csv"
    summary_results = []

    for exp_name in selected_exp_names:
        exp = experiments[exp_name]
        test_dataset = exp.datasets["test"]
        print(f"\n--- Evaluating UFD on {exp_name} test dataset ({len(test_dataset)} samples) ---")
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=config.num_workers
        )
        
        metrics, details = evaluate_on_UFD(ufd_model, test_loader, device)
        
        # Add experiment name to metrics for summary
        summary_row = {"experiment": exp_name}
        summary_row.update(metrics)
        summary_results.append(summary_row)
        
        # Save detailed results
        details_path = details_dir / f"{exp_name}.csv"
        with open(details_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "label", "pred_prob", "pred_class"])
            writer.writeheader()
            for i, detail in enumerate(details):
                row = {"image_path": test_dataset.file_paths[i]}
                row.update(detail)
                writer.writerow(row)
        
        print(f"Saved detailed results to {details_path}")
        print(
            f"Metrics: Acc: {metrics['test_acc']:.4f}, "
            f"AUC: {metrics['test_auc']:.4f}, "
            f"Prec: {metrics['precision']:.4f}, "
            f"Rec: {metrics['recall']:.4f}, "
            f"AP: {metrics['average_precision']:.4f}, "
            f"MCC: {metrics['mcc']:.4f}"
        )
        
    # Save summary results
    if summary_results:
        file_exists = summary_file.exists()
        # If we are NOT filtering and file exists, we'd normally overwrite (handled by previous logic or expected)
        # But here, if we are filtering, we definitely want to append.
        # Let's clean up if NOT filtering:
        if not args.filter and not args.include and file_exists:
            mode = "w"
            write_header = True
        else:
            mode = "a"
            write_header = not file_exists

        with open(summary_file, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_results[0].keys())
            if write_header:
                writer.writeheader()
            writer.writerows(summary_results)
    
    print(f"\nSummary saved to {summary_file}")

    print("\n\n=== Final Report ===")
    for row in summary_results:
        print(f"Dataset: {row['experiment']}")
        for k, v in row.items():
            if k != "experiment":
                print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()