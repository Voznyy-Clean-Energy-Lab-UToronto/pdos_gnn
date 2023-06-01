import os
import csv
import torch
import tarfile
import functools
import numpy as np
from torch_scatter import scatter
from torch.utils.data import Dataset


class MaterialData(Dataset):
    def __init__(self, root=None, id_file=None, data_file=None, n_orbitals=3, train_on_atomic_dos=False):
        self.id_file = id_file
        self.data_file = data_file
        self.n_orbitals = n_orbitals
        self.train_on_atomic_dos = train_on_atomic_dos

        ids = os.path.join(self.id_file)
        assert os.path.exists(ids),  f'{ids} does not exist!'

        assert os.path.exists(self.data_file),  f'{self.data_file} does not exist!'

        self.graphs_data = tarfile.open(self.data_file)

        with open(ids) as file:
            reader = csv.reader(file)
            self.data = [row for row in reader]

    def __len__(self):
        length = len(self.data)
        return length

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        cif_id = self.data[idx][0]
        material_graph = torch.load(self.graphs_data.extractfile(self.data_file.removesuffix('.tar') + f'/{cif_id}_crystal_graph_pdos.pt')) #torch.load(os.path.join(self.root_dir, f'{cif_id}_crystal_graph_pdos.pt'))


        target_dos = material_graph.dos
        target_pdos = material_graph.pdos

        target_pdos_cdf = torch.cumsum(material_graph.pdos, dim=1)*material_graph.e_diff
        material_graph.pdos_cdf = target_pdos_cdf
        material_graph.dos_cdf = torch.sum(target_pdos_cdf, dim=0).unsqueeze(dim=0)

        material_graph.norm_coef = torch.ones(target_pdos.shape[0], 1)
        max_density = torch.max(target_pdos)
        
        n_atoms = int(target_pdos.shape[0]/self.n_orbitals)
        atoms = []
        for i in range(n_atoms):
            atoms.extend([i]*self.n_orbitals)
        atoms_batch = torch.LongTensor(np.array(atoms))
        size = int(atoms_batch.max().item() + 1)
        atomic_dos = scatter(target_pdos, atoms_batch, dim=0, dim_size=size, reduce='add')
        atomic_dos_cdf = torch.cumsum(atomic_dos, dim=1)
        atomic_dos_cdf_inverse = torch.cumsum(torch.fliplr(atomic_dos), dim=1)
        material_graph.atomic_dos = atomic_dos
        material_graph.atomic_dos_cdf = atomic_dos_cdf
        material_graph.atomic_dos_cdf_inverse = atomic_dos_cdf_inverse
        material_graph.atoms_batch = atoms_batch

        material_graph.edge_attr = material_graph.distances

        return material_graph
