import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

DEFAULT_RUN_DIRS = []

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dirs", nargs="+", default=DEFAULT_RUN_DIRS,
                         help="Cross-validation run directories (output of `main.py --task cross_val`) whose fold "
                              "curves should be aggregated together.")
    parser.add_argument("--test_output_dir", default="test_outputs",
                         help="Directory containing eval_<run_dir>_fold_<i>/results.csv from scripts/run_eval.py.")
    parser.add_argument("--n_folds", type=int, default=4)
    parser.add_argument("--output", default="training_curves.png")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load all training curves across all requested run directories
    train_losses = []
    val_losses = []
    test_losses = []

    for run_dir in args.run_dirs:
        for i in range(1, args.n_folds + 1):
            curve_file = f"{run_dir}/training_curves_fold_{i}.json"
            if os.path.exists(curve_file):
                with open(curve_file, 'r') as f:
                    data = json.load(f)
                    train_losses.append(data['train_loss'])
                    val_losses.append(data['val_loss'])

            test_file = f"{args.test_output_dir}/eval_{run_dir}_fold_{i}/results.csv"
            if os.path.exists(test_file):
                with open(test_file, 'r') as f:
                    for line in f:
                        if "val_loss" in line:
                            parts = line.split('\t') if '\t' in line else line.split()
                            test_losses.append(float(parts[-1]))

    if not train_losses:
        print("No training curves found.")
        return

    # Find minimum length
    min_len = min(len(x) for x in train_losses)
    train_losses = [x[:min_len] for x in train_losses]
    val_losses = [x[:min_len] for x in val_losses]

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    
    epochs = np.arange(1, train_losses.shape[1] + 1)
    
    train_mean = np.mean(train_losses, axis=0)
    train_std = np.std(train_losses, axis=0)
    
    val_mean = np.mean(val_losses, axis=0)
    val_std = np.std(val_losses, axis=0)
    
    final_train_mean = train_mean[-1]
    final_train_std = train_std[-1]
    
    final_val_mean = val_mean[-1]
    final_val_std = val_std[-1]
    
    if test_losses:
        test_mean = np.mean(test_losses)
        test_std = np.std(test_losses)
        test_label = f'Test: {test_mean*1000:.2f} $\\times 10^{{-3}}$' # $\pm$ {test_std*1000:.2f}
    else:
        test_label = 'Test: N/A'
    
    train_label = f'Train: {final_train_mean*1000:.2f} $\\times 10^{{-3}}$' # $\pm$ {final_train_std*1000:.2f}
    val_label = f'Validation: {final_val_mean*1000:.2f} $\\times 10^{{-3}}$' # $\pm$ {final_val_std*1000:.2f}
    
    # Set style for paper publication
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300
    })

    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Plot Train
    ax.plot(epochs, train_mean, label=train_label, color='#1f77b4', linewidth=2)
    ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, color='#1f77b4', alpha=0.2)
    
    # Plot Val
    ax.plot(epochs, val_mean, label=val_label, color='#ff7f0e', linewidth=2)
    ax.fill_between(epochs, val_mean - val_std, val_mean + val_std, color='#ff7f0e', alpha=0.2)
    
    ax.plot([], [], ' ', label=test_label)
    
    ax.set_xlabel('Epochs', fontsize=16)
    ax.set_ylabel('Loss (Wasserstein distance)', fontsize=16)
    ax.set_yscale('log')
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=14)
    
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight', dpi=300)
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()