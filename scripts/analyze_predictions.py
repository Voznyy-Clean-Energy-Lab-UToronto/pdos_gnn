import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_total_dos(path, n_grid):
    """
    Load a header-less test_output_total_dos(_cdf).csv file.
    """
    df = pd.read_csv(path, header=None)
    ids = df.iloc[:, 0].astype(str).to_numpy()
    n_pred = df.shape[1] - 1 - n_grid
    pred = df.iloc[:, 1:1 + n_pred].to_numpy(dtype=float)
    target = df.iloc[:, 1 + n_pred:1 + n_pred + n_grid].to_numpy(dtype=float)
    return ids, pred, target


def wasserstein2_from_cdf(pred_cdf, target_cdf, energy):
    """Per-row squared 2-Wasserstein distance == integral of squared CDF difference."""
    return np.trapz((pred_cdf - target_cdf) ** 2, x=energy, axis=1)


def select_groups(order, n):
    """Given row indices sorted by ascending distance, pick the best-n / median-n / worst-n."""
    n_total = len(order)
    n = min(n, n_total)
    best = order[:n]
    worst = order[-n:][::-1]
    mid = n_total // 2
    half = n // 2
    start = max(min(mid - half, n_total - n), 0)
    median = order[start:start + n]
    return {"best": best, "median": median, "worst": worst}


def plot_group(name, indices, ids, distances, energy, energy_pred, pred_dos, target_dos, pred_cdf, target_cdf, output_dir):
    n = len(indices)
    if n == 0:
        return
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6), squeeze=False)
    for col, idx in enumerate(indices):
        axes[0, col].plot(energy, target_cdf[idx], "k-", label="Target")
        axes[0, col].plot(energy, pred_cdf[idx], "r--", label="Predicted")
        axes[0, col].set_title(f"{ids[idx]}\n$W_2^2$={distances[idx]:.3e}", fontsize=9)
        axes[0, col].set_xlabel("Energy (eV)")
        if col == 0:
            axes[0, col].set_ylabel("Cumulative DOS")

        axes[1, col].plot(energy, target_dos[idx], "k-", label="Target")
        axes[1, col].plot(energy_pred, pred_dos[idx], "r--", label="Predicted")
        axes[1, col].set_xlabel("Energy (eV)")
        if col == 0:
            axes[1, col].set_ylabel("DOS")

    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{name.capitalize()} test-set predictions (ranked by Wasserstein-2 distance)")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"predictions_{name}.png")
    fig.savefig(out_path, dpi=300)
    print(f"Saved {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dos_csv", required=True,
                         help="Path to test_output_total_dos.csv (raw predicted+target DOS).")
    parser.add_argument("--cdf_csv", required=True,
                         help="Path to test_output_total_dos_cdf.csv (predicted+target CDF).")
    parser.add_argument("--e_min", type=float, default=-20.0, help="Lower bound of the energy grid (eV).")
    parser.add_argument("--e_max", type=float, default=10.0, help="Upper bound of the energy grid (eV).")
    parser.add_argument("--n_grid", type=int, default=256, help="Number of energy grid points.")
    parser.add_argument("--n", type=int, default=4,
                         help="Number of materials per best/median/worst panel "
                              "4-per-row layout).")
    parser.add_argument("--output_dir", default=".", help="Where to save plots and the outlier table.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    energy = np.linspace(args.e_min, args.e_max, args.n_grid)

    ids_dos, pred_dos, target_dos = load_total_dos(args.dos_csv, args.n_grid)
    ids_cdf, pred_cdf, target_cdf = load_total_dos(args.cdf_csv, args.n_grid)
    if len(ids_dos) != len(ids_cdf) or not np.array_equal(ids_dos, ids_cdf):
        print("Warning: material id ordering differs between --dos_csv and --cdf_csv; rows are matched by "
              "position, not by id. Make sure both files came from the same `--task test --save_dos` run.")
    energy_pred = energy[-pred_dos.shape[1]:]

    distances = wasserstein2_from_cdf(pred_cdf, target_cdf, energy)
    order = np.argsort(distances)  # ascending: best (lowest distance) first

    groups = select_groups(order, args.n)
    for name, indices in groups.items():
        plot_group(name, indices, ids_dos, distances, energy, energy_pred, pred_dos, target_dos, pred_cdf, target_cdf, args.output_dir)

    # Outlier table: full ranking, worst (largest distance) first
    table = pd.DataFrame({"id": ids_dos, "wasserstein2_distance": distances})
    table = table.sort_values("wasserstein2_distance", ascending=False).reset_index(drop=True)
    table.insert(0, "rank_worst_to_best", np.arange(1, len(table) + 1))
    out_path = os.path.join(args.output_dir, "outlier_table.csv")
    table.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
