import json
import torch
import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt


class Scaler(object):
    """ Scale a Tensor and restore it later. """
    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the min and max"""
        self.min = torch.min(tensor)
        self.max = torch.max(tensor)

    def norm(self, tensor):
        return (tensor - self.min) / (self.max - self.min)

    def denorm(self, scaled_tensor):
        return scaled_tensor * (self.max - self.min) + self.min

    def state_dict(self):
        return {'min': self.min, 'max': self.max}

    def load_state_dict(self, state_dict):
        self.min = state_dict['min']
        self.max = state_dict['max']


def save_model(state: dict, epoch: int, save_path: str = None, fold: int = None, best: bool = False, init: bool = False):
    """
        Saves PyTorch model state
        -------------------------
    """

    if fold is not None:
        filename = f'%s/checkpoint_fold_{fold+1}_{epoch}.pth.tar' % save_path
    else:
        filename = f'%s/checkpoint_fold_{epoch}.pth.tar' % save_path
    if best:
        torch.save(state, filename.removesuffix(
            f"_{epoch}.pth.tar")+"_best"+".pth.tar")
    elif init:
        torch.save(state, f'%s/model_init.pth.tar' % save_path)
    else:
        torch.save(state, filename)


def save_training_curves(fold, training_curves_dict, save_path):
    with open('%s/training_curves_fold_%d.json' % (save_path, fold), 'w') as tc_file:
        json.dump(training_curves_dict, tc_file)


def save_cv_results(folds, error_type_list, cv_lists, mean_list, std_list, save_path):
    fold_dict = dict(zip(error_type_list, cv_lists))
    fold_dict["Folds"] = range(1, folds+1)
    fold_df = pd.DataFrame(data=fold_dict)
    
    results_dict =  {"Error type": error_type_list, "Mean errors": mean_list, "Error standard deviation": std_list}
    result_df = pd.DataFrame(data=results_dict)

    fold_df.to_csv("%s/fold_stats.csv"%save_path, sep='\t')
    result_df.to_csv("%s/results.csv"%save_path, sep='\t')

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(" \n Total training results: ")
        print(result_df.to_string(index=False))


def print_output(epoch: int, train_loss: float, val_loss: float, train_pdos_rmse: float, val_pdos_rmse: float, train_cdf_pdos_rmse: float, val_cdf_pdos_rmse: float):
    print("Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, Train PDOS RMSE: {:.4f}, Val PDOS RMSE: {:.4f}, Train CDF PDOS RMSE: {:.4f}, Val CDF PDOS RMSE: {:.4f}".format(epoch, train_loss, val_loss, train_pdos_rmse, val_pdos_rmse, train_cdf_pdos_rmse, val_cdf_pdos_rmse))


def plot_training_curve(save_path: str, val_loss_list: list, train_loss_list: list, fold: int):
    """
        Creats training curve plots for experiment metrics
        --------------------------------------------------
        Input:
            - save_path:            Path where plots will be saved
            - val_loss_list:        List with validation loss values
            - train_loss_list:      List with training loss values
            - fold:                 Training fold number 
    """
    epochs = range(1, len(val_loss_list)+1)
    fig = plt.figure(figsize=(12,8))
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.plot(epochs, train_loss_list, label = "Training Loss", color = 'tab:olive')
    plt.plot(epochs, val_loss_list, label = "Validation Loss", color = 'tab:green')
    plt.yscale('log')
    plt.legend(loc = 'best')
    plt.title("Training Curve", loc='center')
    plt.savefig(f'%s/trainingcurve_fold_{fold}'%save_path + '.png')

def get_spd_pdos(orbital_pdos: np.ndarray, orbital_types: List[str], n_atoms: int) -> np.ndarray:
    """
        Aggregates a material's predicted per-atom, per-orbital output into the s/p/d
        fingerprint vector used for PDOS similarity search: sums within each orbital
        manifold across all atoms, then normalizes by atom count. Representation-agnostic
        -- orbital_pdos may be raw densities or cumulative densities (CDFs); the caller's
        choice of `use_cdf` when running the model determines which one this receives,
        and the output is that same representation (e.g. a "cdf" fingerprint if the model
        was called with use_cdf=True, matching the pipeline's stored
        materials_space_pdos_fingeprints.csv.gz / "pdos_all_data_cdf" convention).
        Input:
            - orbital_pdos:   [n_atoms*n_orbitals, grid] predicted orbital PDOS or CDF
            - orbital_types:  orbital name (e.g. "s", "px", "dxy") for each row above
            - n_atoms:        number of atoms in the unit cell, for normalization
        Output:
            - fingerprint:    [3*grid] concatenated [s, p, d] vector, same representation as input
    """
    p_orbitals = ("px", "py", "pz")
    d_orbitals = ("dxy", "dyz", "dz2", "dxz", "dx2")
    orbital_types = np.asarray(orbital_types)

    s_pdos = orbital_pdos[orbital_types == "s"].sum(axis=0)
    p_pdos = orbital_pdos[np.isin(orbital_types, p_orbitals)].sum(axis=0)
    d_pdos = orbital_pdos[np.isin(orbital_types, d_orbitals)].sum(axis=0)

    return np.concatenate([s_pdos, p_pdos, d_pdos]) / n_atoms


def plot_output_distribution(data, data_pred, epoch):
    data = data.detach().numpy().flatten()
    data_pred = data_pred.detach().numpy().flatten()
    import seaborn as sns
    sns.set_style("white")
    plt.hist(data, bins=100, label="Target PDOS")
    plt.hist(data_pred, bins=100, label=f"Predicted PDOS, epoch:{epoch}")
    plt.legend()
    plt.xlabel("DOS amplitude")
    plt.ylabel("Count")
    #plt.yscale("log")
    plt.show()