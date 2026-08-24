"""
UMAP embeddings of the materials space under four different descriptor types:
ML-PDOS fingerprints (this work's ProDosNet predictions), CrystalNN fingerprints, 
JARVIS-CFID descriptors, and Magpie descriptors. Each annotated by band gap, 
oxide/non-oxide, and Li-containing/non-Li, with 
photovoltaic/solid-electrolyte/battery-cathode/catalyst materials highlighted.

By default the script loads pre-computed UMAP coordinates (parquet files with
'x'/'y' columns) from --coord_dir; pass --recompute_coords to instead compute
them from the raw feature files (slower, requires the `umap-learn` package).
"""
import os
import re
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1 import make_axes_locatable
from adjustText import adjust_text
import matplotlib.patheffects as PathEffects
from tqdm import tqdm

FEATURE_NAMES = ["spd_pdos_800", "CrystalNNfp", "JarvisCFID", "magpie_fp"]
DESCRIPTOR_NAMES = [
    "ML-PDOS Fingerprints",
    "CrystalNN Fingerprints",
    "CFIDs",
    "Magpie Descriptors",
]

# Materials highlighted in Fig. 4 / S13, grouped by application area.
HIGHLIGHT_GROUPS = [
    {"name": "Photovoltaic Materials", "ids": ["mp-2534", "mp-20351", "mp-2490", "mp-1069538", "mp-600089", "mp-5238", "mp-406", "mp-149"]},
    {"name": "Solid Electrolytes", "ids": ["mp-696128", "mp-5614", "mp-720509", "mp-985583", "mp-2251", "mp-985591", "mp-950995", "mp-985592", "mp-38684", "mp-38008", "mp-1222315", "mp-942733"]},
    {"name": "Battery Cathodes", "ids": ["mp-19017", "mp-18997", "mp-18968", "mp-25411", "mp-22526", "mp-9615"]},
    {"name": "Catalysts", "ids": ["mp-2723", "mp-825", "mp-19395", "mp-1272749", "mp-17554", "mp-20194"]},
]


def get_umap_coord(matrix, seed=3):
    import umap
    reducer = umap.UMAP(random_state=seed)
    return reducer.fit_transform(matrix)


def get_pca_coord(matrix):
    return PCA(n_components=2).fit_transform(matrix)


def load_data(feat_file_path, prop_path, feat_name):
    """Load a feature matrix (csv/csv.gz or parquet) and merge with id/property metadata."""
    if feat_file_path.endswith(".csv") or feat_file_path.endswith(".csv.gz"):
        df = pd.read_csv(feat_file_path)
        id_prop_df = pd.read_csv(prop_path)
        id_prop_df.columns = ["id", "pretty_formula", "band_gap", "e_above_hull"]
        df = id_prop_df.merge(df, on="id", how="inner")
        matrix = df.drop(["Unnamed: 0", "id", "pretty_formula", "band_gap", "e_above_hull"], axis=1).to_numpy()
        df = df[["id", "pretty_formula", "band_gap", "e_above_hull"]]
    elif feat_file_path.endswith(".parquet"):
        df = pd.read_parquet(feat_file_path)
        df = df[df[feat_name].apply(lambda x: not np.all(np.isnan(x)))]
        id_prop_df = pd.read_csv(prop_path)
        id_prop_df.columns = ["id", "pretty_formula", "band_gap", "e_above_hull"]
        df = id_prop_df.merge(df, on="id", how="inner")
        matrix = np.vstack(df[feat_name].to_numpy())
        df = df[["id", "pretty_formula", "band_gap", "e_above_hull"]]
    else:
        raise ValueError(f"Unsupported feature file type: {feat_file_path}")
    return df, matrix


def get_oxide(row):
    if row["pretty_formula"] is None or row["pretty_formula"] is np.nan:
        return False
    return "O" in row["pretty_formula"]


def get_li(row):
    if row["pretty_formula"] is None or row["pretty_formula"] is np.nan:
        return False
    return "Li" in row["pretty_formula"]


def get_subplots_sns(dfs, highlight_groups, descriptor_names, output_path):
    colnames = ["(A) Photovoltaics", "(B) Electrolytes and Cathodes", "(C) Catalysts"]
    fig, axes = plt.subplots(len(dfs), 3, figsize=(13, len(dfs) * 4), gridspec_kw={"width_ratios": [1, 1, 1]})

    legend_loc = "lower right"
    for i, df in enumerate(dfs):
        ax = axes[i]
        descriptor_name = descriptor_names[i]
        if i == 3:
            legend_loc = "upper left"
        if i == 2:
            legend_loc = "lower left"

        if i == 0:
            for j, colname in enumerate(colnames):
                ax[j].set_title(colname, fontsize=14, fontweight="bold")

        norm = mcolors.Normalize(vmin=0, vmax=3)

        # --- Band gap panel ---
        bg_idx = 0
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sns.scatterplot(data=df, x="x", y="y", ax=ax[bg_idx], hue="band_gap", hue_norm=norm, alpha=1, palette="viridis", s=5)

        group = highlight_groups[0]
        df_pm = df[df["id"].isin(group["ids"])]
        sns.scatterplot(data=df_pm, x="x", y="y", ax=ax[bg_idx], color="red", marker=(4, 1, 45), alpha=1, s=150)

        texts = []
        for _, row in df_pm.iterrows():
            label = re.sub(r"(\d+)", r"$_{\1}$", row["pretty_formula"])
            txt = ax[bg_idx].text(row["x"], row["y"], label, fontsize=10, color="white")
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="black")])
            texts.append(txt)
        adjust_text(texts, ax=ax[bg_idx], arrowprops=dict(arrowstyle="->", color="black"))

        sm.set_array([])
        ax[bg_idx].get_legend().remove()
        ax[bg_idx].set(xlabel=None, ylabel=None)
        ax[bg_idx].set(xticklabels=[], yticklabels=[])
        ax[bg_idx].tick_params(bottom=False, left=False)

        point = Line2D([0], [0], label="Photovoltaic Materials", marker=(4, 1, 45), markersize=12,
                       markeredgecolor="w", markerfacecolor="red", linestyle="")
        ax[bg_idx].legend(["Photovoltaic Materials"], handles=[point], loc=legend_loc)

        divider = make_axes_locatable(ax[bg_idx])
        cax = divider.append_axes("left", size="5%", pad=0.1)
        cbar = plt.colorbar(sm, cax=cax, label="Band gap, eV",
                             format=mticker.FixedFormatter(["0.0", "0.5", "1.0", "1.5", "2.0", "2.5", "> 3.0"]), extend="max")
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.set_ylabel("Band gap, eV")
        cbar.ax.yaxis.set_label_position("left")

        ax[bg_idx].set_ylabel(descriptor_name, fontsize=16, labelpad=70, fontweight="bold")

        # --- Oxide panel ---
        ox_idx = 2
        sns.scatterplot(data=df[df["oxide"] == 1], x="x", y="y", ax=ax[ox_idx], color="red", alpha=0.3, legend=False, s=5)
        sns.scatterplot(data=df[df["oxide"] == 0], x="x", y="y", ax=ax[ox_idx], color="black", alpha=0.3, legend=False, s=5)

        group = highlight_groups[3]
        df_c = df[df["id"].isin(group["ids"])]
        sns.scatterplot(data=df_c, x="x", y="y", ax=ax[ox_idx], color="blue", marker="v", alpha=1, legend=False, s=150)

        texts = []
        for _, row in df_c.iterrows():
            label = re.sub(r"(\d+)", r"$_{\1}$", row["pretty_formula"])
            txt = ax[ox_idx].text(row["x"], row["y"], label, fontsize=10, color="white")
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="black")])
            texts.append(txt)
        adjust_text(texts, ax=ax[ox_idx], arrowprops=dict(arrowstyle="->", color="black"))

        ax[ox_idx].legend(["Oxide", "Non-Oxide", "Catalysts"], loc=legend_loc)
        leg = ax[ox_idx].get_legend()
        leg.legendHandles[0].set_alpha(1)
        leg.legendHandles[1].set_alpha(1)
        leg.legendHandles[0]._sizes = [30]
        leg.legendHandles[1]._sizes = [30]

        ax[ox_idx].set(xlabel=None, ylabel=None)
        ax[ox_idx].set(xticklabels=[], yticklabels=[])
        ax[ox_idx].tick_params(bottom=False, left=False)

        # --- Lithium panel ---
        li_idx = 1
        sns.scatterplot(data=df[df["li"] == 0], x="x", y="y", ax=ax[li_idx], color="darkblue", alpha=0.3, legend=False, s=5)
        sns.scatterplot(data=df[df["li"] == 1], x="x", y="y", ax=ax[li_idx], color="gold", alpha=0.3, legend=False, s=5)

        group = highlight_groups[1]
        df_se = df[df["id"].isin(group["ids"])]
        sns.scatterplot(data=df_se, x="x", y="y", ax=ax[li_idx], color="orangered", marker="p", alpha=1, s=150)
        group = highlight_groups[2]
        df_bc = df[df["id"].isin(group["ids"])]
        sns.scatterplot(data=df_bc, x="x", y="y", ax=ax[li_idx], color="dodgerblue", marker=(4, 0, 0), alpha=1, legend=False, s=150)

        ax[li_idx].legend(["No Lithium", "Contains Lithium", "Solid Electrolytes", "Battery Cathodes"], loc=legend_loc)
        leg = ax[li_idx].get_legend()
        leg.legendHandles[0].set_alpha(1)
        leg.legendHandles[1].set_alpha(1)
        leg.legendHandles[0]._sizes = [30]
        leg.legendHandles[1]._sizes = [30]
        ax[li_idx].set(xlabel=None, ylabel=None)
        ax[li_idx].set(xticklabels=[], yticklabels=[])
        ax[li_idx].tick_params(bottom=False, left=False)

        texts = []
        for _, row in df_se.iterrows():
            label = re.sub(r"(\d+)", r"$_{\1}$", row["pretty_formula"])
            txt = ax[li_idx].text(row["x"], row["y"], label, fontsize=10, color="white")
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="black")])
            texts.append(txt)
        for _, row in df_bc.iterrows():
            label = re.sub(r"(\d+)", r"$_{\1}$", row["pretty_formula"])
            txt = ax[li_idx].text(row["x"], row["y"], label, fontsize=10, color="white")
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground="black")])
            texts.append(txt)
        adjust_text(texts, ax=ax[li_idx], arrowprops=dict(arrowstyle="->", color="black"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id_prop_csv", default="data/materials_prop.csv",
                         help="CSV with id, pretty_formula, band_gap, e_above_hull columns.")
    parser.add_argument("--pdos_fingerprints", default="data/materials_space_pdos_fingeprints.csv.gz",
                         help="CSV(.gz) of ML-PDOS spd fingerprint predictions (main.py --task predict output).")
    parser.add_argument("--crystalnn_features", default="data/mp_crystalNNfp_features.parquet",
                         help="Parquet file of CrystalNN fingerprints.")
    parser.add_argument("--cfid_features", default="data/jarvis_mp_3d_2020_cfid.parquet",
                         help="Parquet file of JARVIS-CFID descriptors.")
    parser.add_argument("--magpie_features", default="data/mp_magpie_features.parquet",
                         help="Parquet file of Magpie descriptors.")
    parser.add_argument("--coord_dir", default="data",
                         help="Directory to read/write cached <feature_name>_<dim_reduction>_coord.parquet files.")
    parser.add_argument("--recompute_coords", action="store_true",
                         help="Recompute UMAP/PCA embeddings from the raw feature files instead of loading cached "
                              "coordinate parquet files from --coord_dir.")
    parser.add_argument("--dim_reduction", choices=["umap", "pca"], default="umap")
    parser.add_argument("--output", default="umap_comparison.png", help="Output figure path.")
    return parser.parse_args()


def main():
    args = parse_args()

    feature_files = [args.pdos_fingerprints, args.crystalnn_features, args.cfid_features, args.magpie_features]
    coord_files = [os.path.join(args.coord_dir, f"{name}_{args.dim_reduction}_coord.parquet") for name in FEATURE_NAMES]

    print(f"Using dimension reduction method: {args.dim_reduction}")

    if args.recompute_coords:
        dfs = []
        for feature_name, feature_file, coord_file in tqdm(
                list(zip(FEATURE_NAMES, feature_files, coord_files)), desc="Computing coordinates"):
            df, matrix = load_data(feature_file, args.id_prop_csv, feature_name)
            if args.dim_reduction == "pca":
                coord = get_pca_coord(matrix)
            else:
                coord = get_umap_coord(matrix)
            df = pd.concat([df.reset_index(drop=True), pd.DataFrame(coord, columns=["x", "y"])], axis=1)
            df.to_parquet(coord_file, index=False)
            dfs.append(df)
    else:
        dfs = [pd.read_parquet(f) for f in coord_files]

    for group in HIGHLIGHT_GROUPS:
        formulas = []
        for material_id in group["ids"]:
            match = dfs[0][dfs[0]["id"] == material_id]["pretty_formula"]
            formulas.append(match.values[0] if len(match) else None)
        group["formulas"] = formulas

    for group in HIGHLIGHT_GROUPS:
        print(f"{group['name']}")
        for material_id, formula in zip(group["ids"], group["formulas"]):
            print(f"{material_id}: {formula}")

    for df in dfs:
        df["oxide"] = df.apply(get_oxide, axis=1).astype(int)
        df["li"] = df.apply(get_li, axis=1).astype(int)
        df["pretty_formula"] = df["pretty_formula"].astype(str)

    get_subplots_sns(dfs, HIGHLIGHT_GROUPS, DESCRIPTOR_NAMES, args.output)


if __name__ == "__main__":
    main()
