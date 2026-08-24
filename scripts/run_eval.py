"""Runs `main.py --task test` for each fold checkpoint of one or more cross-validation
run directories and prints the resulting test loss.
"""
import os
import sys
import argparse
import subprocess

DEFAULT_RUN_DIRS = []

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dirs", nargs="+", default=DEFAULT_RUN_DIRS,
                         help="Cross-validation run directories (output of `main.py --task cross_val`) to evaluate.")
    parser.add_argument("--test_ids", default="data/test_ids.csv")
    parser.add_argument("--data_file", default="processed_data/processed_data.tar")
    parser.add_argument("--output_dir", default="test_outputs")
    parser.add_argument("--n_folds", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    for run in args.run_dirs:
        print(f"Run: {run}")
        for i in range(1, args.n_folds + 1):
            ckpt = f"{run}/checkpoint_fold_{i}_best.pth.tar"
            if os.path.exists(ckpt):
                save_path = f"{args.output_dir}/eval_{run}_fold_{i}"
                os.makedirs(save_path, exist_ok=True)
                # Use sys.executable (not a literal "python") so this works whether or
                # not the caller's shell has an activated conda/venv on PATH.
                cmd = (f"{sys.executable} pdos_gnn/main.py --task test --test_ids {args.test_ids} "
                       f"--data_file {args.data_file} --model {ckpt} --use_cdf --save_path {save_path}")

                result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                res_file = f"{save_path}/results.csv"
                if os.path.exists(res_file):
                    with open(res_file, "r") as f:
                        for line in f:
                            if "val_loss" in line:
                                # The file might be tab-separated
                                parts = line.split("\t") if "\t" in line else line.split()
                                loss = parts[-1]
                                print(f"  Fold {i} Test Loss: {loss.strip()}")
                else:
                    print(f"  Fold {i} Failed. Output: {result.stdout[:200]} Error: {result.stderr[:200]}")


if __name__ == "__main__":
    main()
