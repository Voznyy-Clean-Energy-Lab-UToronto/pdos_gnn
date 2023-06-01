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
from scipy.ndimage import gaussian_filter1d
from pymatgen.core.structure import Structure
from torch_geometric.data import Data
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.core import Orbital
from pymatgen.electronic_structure.dos import CompleteDos

class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2/self.var**2)

class CrystalGraphPDOS():
    """
        Class to construct graph representation of materials with Projected Density of States as target property
    """
    def __init__(self, dos_dir, cif_dir, radius=8, max_num_nbr=12, sigma=0.2, bound_low=-20,
                 bound_high=10, grid=256, max_element=83, norm_pdos=False, total_dos_norm_limits=0.5, pdos_norm_limits=0.5):
        """
            Holds initial parametes
            -----------------------
            - dos_dir:      Path to directory with Density of States json files
            - cif_dir:      Path to directory with material cif files
            - radius:       Maximum radius for neighbors search during graph construction
            - max_num_nbr:  Maximum number of neighbours in the graph
            - sigma:        Density of States broadening parameter
            - bound_low:    Lower boundary of DOS energy window (eV)
            - bound_high:   Higher boundary of DOS energy window (eV)
            - grid:         Number of density grid points
            - max_element:  Maximum element number in the crystal
            - norm_pdos:    Normalize PDOS if True
        """
        
        self.dos_dir = dos_dir
        self.cif_dir = cif_dir
        self.radius = radius
        self.sigma = sigma
        self.bound_low = bound_low
        self.bound_high = bound_high
        self.grid = grid
        self.max_num_nbr = max_num_nbr
        self.max_element = max_element
        self.norm_pdos = norm_pdos
        self.total_dos_norm_limits = total_dos_norm_limits
        self.pdos_norm_limits = pdos_norm_limits

        self.orbital_name_list = ["s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2"]
        self.spin=Spin(1)

        with open("utilities/atom_features_one_hot.json", "r") as element_one_hot_features_file:
                    self.element_one_hot_features = json.load(element_one_hot_features_file)
        with open("utilities/orbit_radius_fea.json", "r") as element_one_hot_features_file:
                    self.orbit_radius_fea = json.load(element_one_hot_features_file)
        with open("utilities/n_electrons.json", "r") as n_electrons_file:
                    self.n_electrons_dict = json.load(n_electrons_file)
        dmin=0
        step=0.2
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)


    def get_normalization_coeficient(self, energy, density, e_min, e_max, complete_dos=None, orbital=False):
        """
            normalize_total_dos function calculates normalization coefficient for Density of States
            ---------------------------------------------------------------------------------------
            Input:
                - complete_dos:     Density of States object
                - energy:           DOS energies
                - density:          DOS densities
                - e_min:            Lower boundary of integration
                - e_max:            Higher boundary of integration
            Returns:
                - norm_coefficient: Normalization coefficient for DOS
        """

        energy_norm = energy[(energy < e_max) & (energy > e_min)]
        density_norm = density[(energy < e_max) & (energy > e_min)]
        area = np.trapz(density_norm, x=energy_norm)
        if orbital: 
            if area == 0:
                return 1
            else:
                norm_coefficient = 2/area
            return norm_coefficient
        else:
            n_electrons = 0
            for site in complete_dos.structure:
                n_electrons += int(self.n_electrons_dict[site.specie.symbol])
            norm_coefficient = n_electrons/area
            return norm_coefficient

    def preproc_dos(self, energies, densities, extend_range=10, sigma=0.3, smear=True):
        """
            preproc_dos function extends DOS energy window and applies broadening (smearing)
            --------------------------------------------------------------------------------
        """
        diff = [energies[i + 1] - energies[i]
                for i in range(len(energies) - 1)]
        avgdiff = sum(diff) / len(diff)

        min_energy = energies[0]
        max_energy = energies[-1]+avgdiff
   
        energies_low = np.linspace(min_energy-extend_range, min_energy, int(extend_range/avgdiff), endpoint=False)
        energies_high = np.linspace(max_energy, max_energy+extend_range, int(extend_range/avgdiff), endpoint=True)
        densities_low = np.zeros(len(energies_low))
        densities_high = np.zeros(len(energies_high))
        energies_extend = np.concatenate((energies_low, energies, energies_high), axis=None)
        densities_extend = np.concatenate((densities_low, densities, densities_high), axis=None)

        if smear:
            smeared_dens = gaussian_filter1d(densities_extend, sigma/avgdiff)
            return energies_extend, smeared_dens
    
        return energies_extend, densities_extend


    def get_graph(self, crystal, material_id):
        """
            Generates crystal graph from cif pymatgen structure
            Input:
                - crystal:      Pymatgen structure
                - material_id:  Materials Project id
            Output:
                - bond_index:                   Tensor of edge indices that encode conections in graph
                - bond_attr:                    Tensor of edge features that 
                - crystal_orbital_name_list:    List of orbital names (s, px, py, etc.) 
                - crystal_orbital_index_list:   List of orbital indices corresponding to orbital types
                - distances:                    Tensor of distances between atoms
        """
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        bond_index = [[],[]]
        bond_attr = []
        crystal_orbital_name_list = []
        crystal_orbital_index_list = []
        for i, nbrs in enumerate(all_nbrs):
                # Add orbital names and indices for particular site i to lists for crystal data
                crystal_orbital_name_list.append(self.orbital_name_list)
                crystal_orbital_index_list.append(list(range(0, 9)))

                if len(nbrs) < self.max_num_nbr:
                        warnings.warn('In {} found less neighbours than set maximum number of neighbors (n_max = {}). Material will be skipped.'.format(material_id, self.max_num_nbr), stacklevel=2)
                        return None
                else:
                        # Create crystal graph (atomic) bond indices and bond attributes (distances)
                        bond_index[0] += [i]*self.max_num_nbr
                        bond_index[1].extend(list(map(lambda x: x[2], nbrs[:self.max_num_nbr])))
                        bond_attr.extend(np.asarray(list(map(lambda x: x[1], nbrs[:self.max_num_nbr]))))

        bond_index, bond_attr = np.array(bond_index), np.array(bond_attr)
        distances = torch.unsqueeze(torch.Tensor(bond_attr), 1)
        bond_attr = self.gdf.expand(bond_attr)
        bond_attr = torch.Tensor(bond_attr)
        bond_index = torch.LongTensor(bond_index)
        return bond_index, bond_attr, crystal_orbital_name_list, crystal_orbital_index_list, distances


    def generate_node_features(self, material_file_name, crystal, crystal_orbital_name_list, crystal_orbital_index_list, efermi=None, complete_dos=None, total_dos_norm_coef=None, for_training=False):
        sites = []
        elements = []
        atom_fea = []
        target_pdos = []
        orbital_types = []
        orbital_max_density_list = []

        if for_training:
            target_total_density = np.zeros(self.grid)
        # Iterate over orbital energies for each site
        for k, site_orbitals in enumerate(crystal_orbital_index_list):
            if complete_dos is not None:
                site = complete_dos.structure[k]
            else:
                site = crystal.sites[k]
            site_one_hot_features = np.asarray(self.element_one_hot_features[str(crystal.sites[k].specie.number)])
            site_orbit_radius_features = np.asarray(self.orbit_radius_fea[str(crystal.sites[k].specie.number)])
            #print(site_orbit_radius_features)
            atom_fea.append(site_one_hot_features)
            for i, _ in enumerate(site_orbitals):
                orbital_types.append(crystal_orbital_name_list[k][i])
                elements.append(str(crystal.sites[k].specie))
                sites.append(str(k))

                if for_training:
                    orbital = Orbital(crystal_orbital_index_list[k][i])
                    orbital_dos = complete_dos.get_site_orbital_dos(site, orbital)
                    energies_ext, densities_ext = self.preproc_dos(orbital_dos.energies, orbital_dos.get_densities(spin=self.spin), sigma=self.sigma, extend_range=30, smear=False)
                    interp_funtion = interp1d(energies_ext, densities_ext)
                    shifted_energies = np.linspace(self.bound_low+efermi, self.bound_high+efermi, self.grid)
                    target_orbital_density = interp_funtion(shifted_energies)

                    # Remove densities with sharp peaks at high energies
                    #if np.max(target_orbital_density) > 4 and np.argmax(target_orbital_density) > 128: 
                    #        return None
                            #    with open("orbital_density_above_2.txt", "a") as file_object:
                        #        file_object.write(material_file_name + "\n")
                        #        return None
                           #print(np.argmax(target_orbital_density_norm))
        
                            #warnings.warn("\n Structure {} has maximum orbital density > 2".format(material_file_name), stacklevel=2)
                              
                            # plt.plot(shifted_energies, target_orbital_density_norm, label=material_file_name)
                            # plt.legend()
                            # plt.show()
 
                    if self.norm_pdos:
                        orbital_norm_coefficient = self.get_normalization_coeficient(energy=orbital_dos.energies, density=orbital_dos.get_densities(spin=self.spin), e_min=np.min(orbital_dos.energies), e_max=np.max(orbital_dos.energies), orbital=True)
                        target_orbital_density_norm = orbital_norm_coefficient * target_orbital_density
                        target_total_density += target_orbital_density_norm
                        target_pdos.append(target_orbital_density_norm)
                        orbital_max_density_list.append(np.max(target_orbital_density_norm))
                    else:
                        target_orbital_density = total_dos_norm_coef * target_orbital_density
                        target_total_density += target_orbital_density
                        target_pdos.append(target_orbital_density)
                        orbital_max_density_list.append(np.max(target_orbital_density))

        atom_fea = np.array(atom_fea)
        atom_fea = torch.Tensor(atom_fea)

        if for_training:
            target_pdos = np.array(target_pdos)
            target_pdos = torch.Tensor(target_pdos)
            target_dos = torch.sum(target_pdos, dim=0)
            target_dos = torch.unsqueeze(target_dos, 0)
            return atom_fea, target_dos, target_pdos, elements, sites, orbital_types, orbital_max_density_list

        return atom_fea, elements, sites, orbital_types


    def get_crystal_pdos_graph(self, material_file):
        """
            This method loads material cif and DOS files, asserts that material fits the dataset constraints, and returns a crystal graph
            Input: 
                - material_file: material cif file
            Output:
                - crystal_graph: Pytorch Geometric Data structure containing material crystal graph with node features, 
                                 edge features, and PDOS as target for training 
            
        """
        material_file_name = os.path.split(material_file)[1].split(".")[0]
        crystal = Structure.from_file(material_file)

        # Check if material contains elements with higher atomic number than specified by parameter max_element
        atomic_numbers = []
        for site in crystal:
            atomic_numbers.append(site.specie.number)
        heaviest_element = np.max(atomic_numbers)
        if heaviest_element > self.max_element:
            warnings.warn("Structure {} contains element with atomic number = {} and will be skipped. Max atomic number = {}.".format(material_file_name, heaviest_element, self.max_element), stacklevel=2)
            return None

        with open(os.path.join(self.dos_dir, material_file_name+"_dos.json"), "r") as material_dos_file:
                    material_dos_file = json.load(material_dos_file)
        # Check if material has spin non-polarized PDOS calculation
        if len(material_dos_file["densities"]) > 1: 
            warnings.warn("Structure {} contains spin-polarized PDOS calculation and will be skipped".format(material_file_name), stacklevel=2)
            return None

        # Check that material structure in cif file has same number of atoms as structure in complete_dos
        complete_dos = CompleteDos.from_dict(material_dos_file)
        if len(crystal) != len(complete_dos.structure):
            warnings.warn("Structure {} does not match structure in PDOS file and will be skipped".format(material_file_name), stacklevel=2)
            return None

        # Check that material PDOS does not contain f orbitals
        spd_dos = complete_dos.get_spd_dos()
        if len(spd_dos.keys()) > 3:
            warnings.warn("Structure {} contains f orbitals in PDOS file and will be skipped".format(material_file_name), stacklevel=2)
            return None
        
        # Check DOS normalization 
        norm_coefficient = self.get_normalization_coeficient(energy=complete_dos.energies, density=complete_dos.get_densities(spin=self.spin), e_min=np.min(complete_dos.energies), e_max=np.max(complete_dos.efermi), complete_dos=complete_dos)
     #   if norm_coefficient > 1 + self.total_dos_norm_limits or norm_coefficient < 1 - self.total_dos_norm_limits:
     #       warnings.warn("Structure {} might have incorrect DOS normalization and will be skipped".format(material_file_name), stacklevel=2)
     #       return None
        
        # Check PDOS normalization
        # for site in complete_dos.structure:
        #     for orbital_n in range(9):
        #         orbital = Orbital(orbital_n)
        #         orbital_dos = complete_dos.get_site_orbital_dos(site, orbital)  
        #         orbital_density = norm_coefficient * orbital_dos.get_densities(spin=self.spin)
        #         orbital_norm_coefficient = self.get_normalization_coeficient(energy=orbital_dos.energies, density=orbital_density, e_min=np.min(orbital_dos.energies), e_max=np.max(orbital_dos.energies), orbital=True)
        #         orbital_area = 2/orbital_norm_coefficient
        #         if orbital_area > 2 + self.pdos_norm_limits or orbital_area < 2 - self.pdos_norm_limits:
        #             warnings.warn("Structure {} might have incorrect PDOS normalization and will be skipped".format(material_file_name), stacklevel=2)
        #             return None

        graph_parameters = self.get_graph(crystal, material_file_name)
        if graph_parameters is not None:
            bond_index, bond_attr, crystal_orbital_name_list, crystal_orbital_index_list, distances = graph_parameters
        else:
            return None
        node_features = self.generate_node_features(material_file_name, crystal, crystal_orbital_name_list, crystal_orbital_index_list, efermi=material_dos_file["efermi"], complete_dos=complete_dos, total_dos_norm_coef=norm_coefficient, for_training=True)
        if node_features is not None:
            atom_fea, target_dos, target_pdos, elements, sites, orbital_types, orbital_max_density_list = node_features
        else: 
            return None
        crystal_graph = Data(x=atom_fea, edge_index=bond_index, edge_attr=bond_attr, dos=target_dos, pdos=target_pdos, material_id=material_file_name, elements=elements, sites=sites, orbital_types=orbital_types, distances=distances, orbital_max_density=np.max(orbital_max_density_list))
        return crystal_graph


    def get_crystal_pdos_graph_pred(self, material_file):
        """
            Constructs and returns crystal graph without target PDOS values
            Input:
                - material_file: cif file of material
            Output:
                - crystal_graph: Pytorch Geometric Data structure containing material crystal graph with node features and  
                                 edge features for prediction when target PDOS values are unknown
        """
        file_name = os.path.split(material_file)[1].split(".")[0]
        crystal = Structure.from_file(material_file)
        bond_index, bond_attr, crystal_orbital_name_list, crystal_orbital_index_list = self.get_crystal_graph(crystal, file_name)
        node_features = self.generate_node_features(crystal, crystal_orbital_name_list, crystal_orbital_index_list, for_training=False)
        if node_features is not None:
            atom_fea, elements, sites, orbital_types = node_features
        crystal_graph = Data(x=atom_fea, edge_index=bond_index, edge_attr=bond_attr, material_id=file_name, elements=elements, sites=sites, orbital_types=orbital_types)
        return crystal_graph
