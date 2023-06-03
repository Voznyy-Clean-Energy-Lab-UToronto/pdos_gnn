import os
import json
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from models.crystal_model import ProDosNet
from typing import List



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


class Predictor():
    def __init__(self):
        self.model = nn.DataParallel(ProDosNet(orbital_fea_len=357, bond_fea_len=4, conv_fea_len=128, skip_embedding=True))

        self.load_model_state()

    def load_model_state(self):
        pretrained_model = torch.load('utilities/pretrained_model.pth.tar', map_location=torch.device('cpu'))
        self.model.load_state_dict(pretrained_model['state_dict'])

    def get_prediction(self, graph, include_target=False):
        self.model.eval()
        output_pdos, output_dos = self.model(graph)
        out_pdos_data = pd.DataFrame(output_pdos.detach().numpy())
        elements = np.array(graph.elements)
        sites = np.array(graph.sites)
        orbital_types = graph.orbital_types
        id = np.array([graph.material_id]*len(orbital_types))

        orbital_types = np.array(orbital_types)
        if include_target:
            target_pdos_data = pd.DataFrame(graph.target_pdos.detach().numpy())
            output_and_id = pd.concat([pd.DataFrame(id), pd.DataFrame(elements), pd.DataFrame(sites), pd.DataFrame(orbital_types), out_pdos_data, target_pdos_data], axis = 1, ignore_index=True, sort=False)
            output_and_id = output_and_id.rename({0: 'id', 1: 'element', 2: 'atom_number', 3: 'orbital_type'}, axis='columns')
        else:
            output_and_id = pd.concat([pd.DataFrame(id), pd.DataFrame(elements), pd.DataFrame(sites), pd.DataFrame(orbital_types), out_pdos_data], axis = 1, ignore_index=True, sort=False)
            output_and_id = output_and_id.rename({0: 'id', 1: 'element', 2: 'atom_number', 3: 'orbital_type'}, axis='columns')
        return output_and_id


def save_model(state: dict, epoch: int, save_path: str = None, fold: int = None, best: bool = False, init: bool = False):
    """
        Saves PyTorch model state
        -------------------------
    """

    if fold is not None:
        filename = f'test_outputs/%s/checkpoint_fold_{fold+1}_{epoch}.pth.tar' % save_path
    else:
        filename = f'test_outputs/%s/checkpoint_fold_{epoch}.pth.tar' % save_path
    if best:
        torch.save(state, filename.removesuffix(
            f"_{epoch}.pth.tar")+"_best"+".pth.tar")
    elif init:
        torch.save(state, f'test_outputs/%s/model_init.pth.tar' % save_path)
    else:
        torch.save(state, filename)


def save_training_curves(fold, training_curve_list, training_curve_name_list, save_path):
    training_curves_dict = dict(zip(training_curve_name_list, training_curve_list))
    with open('test_outputs/%s/training_curves_fold_%d.json' % (save_path, fold), 'w') as tc_file:
        json.dump(training_curves_dict, tc_file)


def save_cv_results(folds, error_type_list, cv_lists, mean_list, std_list, save_path):
    fold_dict = dict(zip(error_type_list, cv_lists))
    fold_dict["Folds"] = range(1, folds+1)
    fold_df = pd.DataFrame(data=fold_dict)
    
    results_dict =  {"Error type": error_type_list, "Mean errors": mean_list, "Error standard deviation": std_list}
    result_df = pd.DataFrame(data=results_dict)

    fold_df.to_csv("test_outputs/%s/fold_stats.csv"%save_path, sep='\t')
    result_df.to_csv("test_outputs/%s/results.csv"%save_path, sep='\t')

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(" \n Total training results: ")
        print(result_df.to_string(index=False))


def print_output(epoch: int, train_pdos_rmse: float, val_pdos_rmse: float, train_cdf_pdos_rmse: float, val_cdf_pdos_rmse: float):
    print("Epoch: {},   Training PDOS RMSE: {:.4f}, Val PDOS RMSE: {:.4f}, Trainin CDF PDOS RMSE: {:.4f}, Val CDF PDOS RMSE: {:.4f}".format(epoch, train_pdos_rmse, val_pdos_rmse, train_cdf_pdos_rmse, val_cdf_pdos_rmse))


def plot_training_curve(save_path: str, val_loss_list: List, train_loss_list: List, fold: int):
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
    plt.set_xlabel('epochs')
    plt.set_ylabel('Loss')
    plt.plot(epochs, train_loss_list, label = "Training Loss", color = 'tab:olive')
    plt.plot(epochs, val_loss_list, label = "Validation Loss", color = 'tab:green')
    plt.yscale('log')
    plt.legend(loc = 'best')
    plt.title("Training Curve", loc='center')
    plt.savefig(f'test_outputs/%s/trainingcurve_fold_{fold}'%save_path + '.png')

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