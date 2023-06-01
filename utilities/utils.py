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