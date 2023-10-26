import torch
import itertools
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from utilities.utils import Scaler
from utilities.data import MaterialData
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from models.crystal_model import ProDosNet
from torch.nn.functional import mse_loss
from utilities.utils import save_model, save_training_curves, save_cv_results, print_output, plot_training_curve

from typing import Tuple


def run_cross_validation(config: dict, args, save_path: str):
    """
        Run k-fold cross-validation on training/validation dataset
        ----------------------------------------------------------
        Input:
            - config:             Hyperparameters for this experiment
            - args:               Set of user-defined arguments
            - save_path:          Path where outputs will be saved
        Output:
            - Saves cross-validation results in csv file
            - Saves and plots trainig curves
            - Saves models checkpoints 
    """
    if args.model_name == "crystal_model":
        from models.crystal_model import ProDosNet
    else:
        print(f"Model {args.model_name} is not available")
        return None

    fold_val_loss_list = []
    fold_train_loss_list = []
    fold_val_pdos_rmse_list = []
    fold_train_pdos_rmse_list = []
    fold_val_cdf_pdos_rmse_list = []
    fold_train_cdf_pdos_rmse_list = []

    print("------------------- Starting Cross-Validation ------------------ \n")
    print(" Running {}-fold cross-validation on {} data file \n".format(args.kfold, args.train_ids))
    dataset = MaterialData(data_file=args.data_file, id_file=args.train_ids)
    splits = KFold(n_splits=args.kfold, shuffle=True, random_state=42)

    print("------------------------ Training Model ------------------------ \n")
    for fold, (train_ids, val_ids) in enumerate(splits.split(np.arange(len(dataset)))):

        model = ProDosNet(orig_atom_fea_len=dataset[0].x.shape[1], nbr_fea_len=dataset[0].edge_attr.shape[1], n_conv=config["n_conv"], use_mlp=args.use_mlp, use_cdf=args.use_cdf)

        if args.cuda:
            device = torch.device("cuda")
            model.to(device)

        metric = nn.MSELoss()
        if args.optim == "Adam":
            print(" Using Adam optimizer")
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        elif args.optim == "AdamW":
            print(" Using AdamW optimizer")
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

        print("---------------------------- Fold {} ----------------------------".format(fold+1))
        
        wieght_sum_list = []

        val_loss_list = []
        train_loss_list = []

        val_pdos_rmse_list = []
        train_pdos_rmse_list = []

        val_cdf_pdos_rmse_list = []
        train_cdf_pdos_rmse_list = []

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        validation_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_subsampler)
        validation_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=validation_subsampler)

        # Save cross-validation id splits
        train_mp_ids = []
        training_distances = []
        for data in train_loader:
            training_distances.append(torch.flatten(data.edge_attr))
            train_mp_ids.extend(data.material_id)
        train_mp_ids_df = pd.DataFrame({"train_ids": train_mp_ids})
        train_mp_ids_df.to_csv(f'test_outputs/{save_path}/train_ids_fold_{fold+1}.csv', index=False)

        val_mp_ids = []
        for data in validation_loader:
            val_mp_ids.extend(data.material_id)
        val_mp_ids_df = pd.DataFrame({"val_ids": val_mp_ids})
        val_mp_ids_df.to_csv(f'test_outputs/{save_path}/val_ids_fold_{fold+1}.csv', index=False)

        # Create a Scaler for bond distances 
        if args.scale:
            training_distances = torch.cat(training_distances, dim=0)
            scaler = Scaler(training_distances)
        else:
            scaler = None
        
        # Train the model
        for epoch in range(1, args.epochs+1):
            train_loss, train_pdos_rmse, train_cdf_pdos_rmse = train(
                model, optimizer, metric, train_loader, train_on_dos=args.train_on_dos, use_cuda=args.cuda, use_cdf=args.use_cdf, scaler=scaler)
            val_loss, val_pdos_rmse, val_cdf_pdos_rmse = validation(
                model, metric, epoch, fold, save_path, validation_loader, train_on_dos=args.train_on_dos, use_cuda=args.cuda, use_cdf=args.use_cdf, scaler=scaler)
            print_output(epoch, train_loss, val_loss, train_pdos_rmse, val_pdos_rmse, train_cdf_pdos_rmse, val_cdf_pdos_rmse)

            n_parameters = 0
            layer_sum_list = []
            for parameters in model.parameters():
                n_parameters += torch.numel(parameters)
                parameters_np = parameters.cpu().data.numpy()
                layer_sum_list.append(np.sum(np.abs(parameters_np)))
            total_weight_sum = np.sum(layer_sum_list).item()/n_parameters
            wieght_sum_list.append(total_weight_sum)
       
            val_loss_list.append(val_loss)
            train_loss_list.append(train_loss)
            val_pdos_rmse_list.append(val_pdos_rmse)
            train_pdos_rmse_list.append(train_pdos_rmse)
            val_cdf_pdos_rmse_list.append(val_cdf_pdos_rmse)
            train_cdf_pdos_rmse_list.append(train_cdf_pdos_rmse)

            if epoch == 1:
                best_val_loss = val_loss
                best_train_loss = train_loss
                best_val_pdos_rmse = val_pdos_rmse
                best_train_pdos_rmse = train_pdos_rmse
                best_val_cdf_pdos_rmse = val_cdf_pdos_rmse
                best_train_cdf_pdos_rmse = train_cdf_pdos_rmse
         
                model_state_best = {'epoch': epoch, 'state_dict': model.state_dict(), 'best_val_loss': best_val_loss, 'best_train_loss': best_train_loss,
                                    'best_val_pdos_rmse': best_val_pdos_rmse, 'best_train_pdos_rmse': best_train_pdos_rmse, 'best_val_cdf_pdos_rmse': best_val_cdf_pdos_rmse, 'best_train_cdf_pdos_rmse': best_train_cdf_pdos_rmse, 'optimizer': optimizer.state_dict(), 'args': vars(args)}
            else:
                best = val_loss < best_val_loss
                
                if best:
                    best_val_loss = val_loss
                    best_train_loss = train_loss
                    best_val_pdos_rmse = val_pdos_rmse
                    best_train_pdos_rmse = train_pdos_rmse
                    best_val_cdf_pdos_rmse = val_cdf_pdos_rmse
                    best_train_cdf_pdos_rmse = train_cdf_pdos_rmse
                    

                    model_state_best = {'epoch': epoch, 'state_dict': model.state_dict(), 'best_val_loss': best_val_loss, 'best_train_loss': best_train_loss,
                                    'best_val_pdos_rmse': best_val_pdos_rmse, 'best_train_pdos_rmse': best_train_pdos_rmse, 'best_val_cdf_pdos_rmse': best_val_cdf_pdos_rmse, 'best_train_cdf_pdos_rmse': best_train_cdf_pdos_rmse, 'optimizer': optimizer.state_dict(), 'args': vars(args)}
            
                    if args.save_best_model:
                        save_model(model_state_best, epoch, save_path, fold=fold, best=True)

            if args.plot_training and epoch % args.plot_interval == 0:
                plot_training_curve(save_path, val_loss_list, train_loss_list, fold=fold+1)
            if epoch % args.model_save_interval==0:
                model_state = {'epoch': epoch, 'state_dict': model.state_dict(), 'best_val_loss': best_val_loss, 'best_train_loss': best_train_loss,
                                    'best_val_pdos_rmse': best_val_pdos_rmse, 'best_train_pdos_rmse': best_train_pdos_rmse, 'best_val_cdf_pdos_rmse': best_val_cdf_pdos_rmse, 'best_train_cdf_pdos_rmse': best_train_cdf_pdos_rmse, 'optimizer': optimizer.state_dict(), 'args': vars(args)}
                save_model(model_state, epoch, save_path, fold)

                training_curve_list = [val_loss_list, train_loss_list, val_pdos_rmse_list, train_pdos_rmse_list, val_cdf_pdos_rmse_list, train_cdf_pdos_rmse_list, wieght_sum_list]
                training_curve_name_list = ["Val loss", "Train loss", "Val PDOS RMSE", "Train PDOS RMSE", "Val CDF PDOS RMSE", "Train CDF PDOS RMSE", "Weight Sum"]
                save_training_curves(fold+1, training_curve_list, training_curve_name_list, save_path)

        fold_val_loss_list.append(best_val_loss)
        fold_train_loss_list.append(best_train_loss)
        fold_val_pdos_rmse_list.append(best_val_pdos_rmse)
        fold_train_pdos_rmse_list.append(best_train_pdos_rmse)
        fold_val_cdf_pdos_rmse_list.append(best_val_cdf_pdos_rmse)
        fold_train_cdf_pdos_rmse_list.append(best_train_cdf_pdos_rmse)
        
        training_curve_list = [val_loss_list, train_loss_list, val_pdos_rmse_list, train_pdos_rmse_list, val_cdf_pdos_rmse_list, train_cdf_pdos_rmse_list, wieght_sum_list]
        training_curve_name_list = ["Val loss", "Train loss", "Val PDOS RMSE", "Train PDOS RMSE", "Val CDF PDOS RMSE", "Train CDF PDOS RMSE", "Weight Sum"]
        save_training_curves(fold+1, training_curve_list, training_curve_name_list, save_path)


    print("----------------- Finished Cross-Validation -----------------")
    cv_lists = [fold_val_loss_list, fold_train_loss_list, fold_val_pdos_rmse_list, fold_train_pdos_rmse_list, fold_val_cdf_pdos_rmse_list, fold_train_cdf_pdos_rmse_list]
    mean_list = [np.mean(list) for list in cv_lists]
    std_list = [np.std(list) for list in cv_lists]
   
    save_cv_results(args.kfold, training_curve_name_list, cv_lists, mean_list, std_list, save_path)

    if args.save_dos or args.save_pdos:
        print("------------------- Saving Predicted PDOS -------------------")
        print()
        for fold, (train_ids, val_ids) in enumerate(splits.split(np.arange(len(dataset)))):
            print("--------------------------- Fold {} --------------------------".format(fold+1))
            validation_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
            validation_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=validation_subsampler)
            best_model = torch.load('test_outputs/{}/checkpoint_fold_{}_best.pth.tar'.format(save_path, fold+1), map_location=torch.device('cpu'))
            model.load_state_dict(best_model['state_dict'])
            optimizer.load_state_dict(best_model['optimizer'])
            if args.cuda:
                device = torch.device("cuda")
                model.to(device)
            val_loss, val_pdos_rmse, val_cdf_pdos_rmse = validation(
                model, metric, "best", fold, save_path, validation_loader, train_on_dos=args.train_on_dos, train_on_atomic_dos=args.train_on_atomic_dos, save_output=True, save_dos=args.save_dos, save_pdos=args.save_pdos, use_cuda=args.cuda, use_cdf=args.use_cdf, scaler=scaler)
            if args.save_only_1_fold_val_pred:
                break
        print("---------------- Finished Saving Predictions ----------------")
          


def run_test(args, save_path: str, test_loader: DataLoader, model: ProDosNet):
    metric = nn.MSELoss()
    pretrained_model = torch.load(args.model, map_location=torch.device('cpu'))
    model.load_state_dict(pretrained_model['state_dict'])
    if args.cuda:
            device = torch.device("cuda")
            model.to(device)
    test_loss, test_pdos_rmse, test_cdf_pdos_rmse = validation(
                model, metric, epoch=0, fold=None, save_path=save_path, validation_loader=test_loader, train_on_dos=args.train_on_dos, train_on_atomic_dos=args.train_on_atomic_dos, save_output=True, save_dos=args.save_dos, save_pdos=args.save_pdos, use_cuda=args.cuda, use_cdf=args.use_cdf, test=True)
    error_type_list = ["Test loss", "Test PDOS RMSE", "Test CDF PDOS RMSE"]
    errors = [test_loss, test_pdos_rmse, test_cdf_pdos_rmse]
    results_dict =  {"Error type": error_type_list, "Mean errors": errors}
    result_df = pd.DataFrame(data=results_dict)
    result_df.to_csv("test_outputs/%s/results.csv"%save_path, sep='\t')

    print("------------------------- Test Results ------------------------- \n")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(" \n Total training results: ")
        print(result_df.to_string(index=False))



def train(model: ProDosNet, 
          optimizer: torch.optim, 
          metric: nn.MSELoss, 
          train_loader: DataLoader, 
          use_cuda: bool = False, 
          use_cdf: bool = False, 
          train_on_dos: bool = False,
          scaler: Scaler = None) -> Tuple[float, float, float]:
    
    model.train()
    
    n_iter = len(train_loader)
    running_loss = 0.0
    running_pdos_rmse = 0.0
    running_cdf_pdos_rmse = 0.0

    for batch_idx, data in enumerate(train_loader):
        e_diff = torch.mean(data.e_diff)
    
        if use_cuda:
            device = torch.device('cuda')
            data = data.to(device)

        if train_on_dos:
            if use_cdf: 
                if scaler is not None:
                    edge_attr = scaler.norm(data.edge_attr)
                else: 
                    edge_attr = data.edge_attr

                target = data.dos_cdf
                output_pdos, output_atomic_dos, output_dos = model(data.x, data.edge_index, edge_attr, data.batch, data.atoms_batch)
                loss = metric(output_dos, target)
                loss_item = loss.item()
                cdf_mse = mse_loss(output_dos, target).item()
                output_dos_diff = torch.diff(output_dos, dim=1)/e_diff
                output_dos_diff[output_dos_diff<0] = 0.0
                pdos_mse = mse_loss(output_dos_diff, (torch.diff(target, dim=1)/e_diff)).item()

            else:
                if scaler is not None:
                    edge_attr = scaler.norm(data.edge_attr)
                else: 
                    edge_attr = data.edge_attr

                target = data.dos
                output_pdos, output_atomic_dos, output_dos = model(data.x, data.edge_index, edge_attr, data.batch, data.atoms_batch)
                loss = metric(output_dos, target)
                loss_item = loss.item()
                pdos_mse = loss.item()
                cdf_mse = metric(torch.cumsum(output_dos, dim=1)*e_diff, torch.cumsum(target, dim=1)*e_diff).item()

        else:
            if use_cdf: 
                if scaler is not None:
                    edge_attr = scaler.norm(data.edge_attr)
                else: 
                    edge_attr = data.edge_attr

                target = data.pdos_cdf
                output_pdos, output_atomic_dos, output_dos = model(data.x, data.edge_index, edge_attr, data.batch, data.atoms_batch)
                loss = metric(output_pdos, target)
                loss_item = loss.item()
                cdf_mse = mse_loss(output_pdos, target).item()
                output_pdos_diff = torch.diff(output_pdos, dim=1)/e_diff
                output_pdos_diff[output_pdos_diff<0] = 0.0
                pdos_mse = mse_loss(output_pdos_diff, (torch.diff(target, dim=1)/e_diff)).item()

            else:
                if scaler is not None:
                    edge_attr = scaler.norm(data.edge_attr)
                else: 
                    edge_attr = data.edge_attr

                target = data.pdos
                output_pdos, output_atomic_dos, output_dos = model(data.x, data.edge_index, edge_attr, data.batch, data.atoms_batch)
                loss = metric(output_pdos, target)
                loss_item = loss.item()
                pdos_mse = loss.item()
                cdf_mse = metric(torch.cumsum(output_pdos, dim=1)*e_diff, torch.cumsum(target, dim=1)*e_diff).item()

        running_loss += loss_item
        running_pdos_rmse += pdos_mse
        running_cdf_pdos_rmse += cdf_mse

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss/n_iter
    epoch_pdos_rmse = running_pdos_rmse/n_iter
    epoch_cdf_pdos_rmse = running_cdf_pdos_rmse/n_iter

    return epoch_loss, epoch_pdos_rmse, epoch_cdf_pdos_rmse


def validation(model: ProDosNet,
               metric: nn.MSELoss, 
               epoch: int, 
               fold: int, 
               save_path: str, 
               validation_loader: DataLoader, 
               save_output: bool = False,  
               save_dos: bool = False, 
               save_pdos: bool = False, 
               save_id_rmse: bool = False,
               save_id_rmse_interv: int = 50,
               use_cuda: bool = False, 
               use_cdf: bool = False, 
               test: bool = False, 
               train_on_dos: bool = False,
               scaler: Scaler = None) -> Tuple[float, float, float]:
    """
        Runs model predictions on validation or test set and returns loss and errors
        ----------------------------------------------------------------------------
        Input:
            - model:                   ProDosNet model
            - metric:                  MSE Loss
            - epoch                    Training epoch
            - fold                     Cross-Valifation fold
            - save_path                Path where outputs will be saved
            - validation_loader        Data loader for validation dataset
            - save_output              If True, saves validation output
            - save_dos                 If True, saves DOS validation predictions
            - save_pdos                If True, saves PDOS validation predictions
            - save_id_rmse             If True, saves validation rmse for each material separately
            - save_id_rmse_interv      Saves rmse for each material every *save_id_rmse_interv* epoch
            - use_cuda                 If True, (GPU is available), loads data to GPU 
            - use_cdf                  If True, uses RMSE Loss of CDFs
            - test                     If True, saves test set outputs
            - scaler                   scale edge features
        Output:
            - epoch_loss
            - epoch_pdos_rmse
            - epoch_cdf_pdos_rmse
        
    """
    model.eval()

    id_error_df_list = []
    if save_output:
        save_dos_list = []
        save_pdos_list = []
        save_dos_list_cdf = []
        save_pdos_list_cdf = []
        
    n_iter = len(validation_loader)

    running_loss = 0.0
    running_pdos_rmse = 0.0
    running_cdf_pdos_rmse = 0.0

    for data in tqdm(validation_loader, disable = not test):
        e_diff = torch.mean(data.e_diff)

        if save_output:
            target_dos_cpu = data.dos
            target_pdos_cpu = data.pdos
            target_dos_cpu_cdf = data.dos_cdf
            target_pdos_cpu_cdf = data.pdos_cdf
                
        if use_cuda:
            device = torch.device('cuda')
            data = data.to(device)

        if train_on_dos:
            if use_cdf: 
                if scaler is not None:
                    edge_attr = scaler.norm(data.edge_attr)
                else: 
                    edge_attr = data.edge_attr

                target = data.dos_cdf
                output_pdos, output_atomic_dos, output_dos = model(data.x, data.edge_index, edge_attr, data.batch, data.atoms_batch)
                loss = metric(output_dos, target)
                loss_item = loss.item()
                cdf_mse = mse_loss(output_dos, target).item()
                output_dos_diff = torch.diff(output_dos, dim=1)/e_diff
                output_dos_diff[output_dos_diff<0] = 0.0
                pdos_mse = mse_loss(output_dos_diff, (torch.diff(target, dim=1)/e_diff)).item()

            else:
                if scaler is not None:
                    edge_attr = scaler.norm(data.edge_attr)
                else: 
                    edge_attr = data.edge_attr

                target = data.dos
                output_pdos, output_atomic_dos, output_dos = model(data.x, data.edge_index, edge_attr, data.batch, data.atoms_batch)
                loss = metric(output_dos, target)
                loss_item = loss.item()
                pdos_mse = loss.item()
                cdf_mse = metric(torch.cumsum(output_dos, dim=1)*e_diff, torch.cumsum(target, dim=1)*e_diff).item()

        else:
            if use_cdf: 
                if scaler is not None:
                    edge_attr = scaler.norm(data.edge_attr)
                else: 
                    edge_attr = data.edge_attr
                    
                target = data.pdos_cdf
                output_pdos, output_atomic_dos, output_dos = model(data.x, data.edge_index, edge_attr, data.batch, data.atoms_batch)
                loss = metric(output_pdos, target)
                loss_item = loss.item()
                cdf_mse = mse_loss(output_pdos, target).item()
                output_pdos_diff = torch.diff(output_pdos, dim=1)/e_diff
                output_pdos_diff[output_pdos_diff<0] = 0.0
                target_pdos_diff = torch.diff(target, dim=1)/e_diff
                pdos_mse = mse_loss(output_pdos_diff, target_pdos_diff).item()

            else:
                if scaler is not None:
                    edge_attr = scaler.norm(data.edge_attr)
                else: 
                    edge_attr = data.edge_attr

                target = data.pdos
                output_pdos, output_atomic_dos, output_dos = model(data.x, data.edge_index, edge_attr, data.batch, data.atoms_batch)
                loss = metric(output_pdos, target)
                loss_item = loss.item()
                pdos_mse = loss.item()
                cdf_mse = mse_loss(torch.cumsum(output_pdos, dim=1)*e_diff, torch.cumsum(target, dim=1)*e_diff).item()

        running_loss += loss_item
        running_pdos_rmse += pdos_mse
        running_cdf_pdos_rmse += cdf_mse
        
        if save_id_rmse and epoch%save_id_rmse_interv == 0:
            ids_list = []
            for orbitals, id in zip(data.orbital_types, data.material_id):
                ids_list.append([id]*len(orbitals))

            ids_list = list(itertools.chain.from_iterable(ids_list))
            sites = list(itertools.chain.from_iterable(data.sites))
            elements = list(itertools.chain.from_iterable(data.elements))
            orbital_types = list(itertools.chain.from_iterable(data.orbital_types))

            output_pdos_orb_cdf = output_pdos.reshape(len(ids_list), 256)
            target_pdos_orb_cdf = target.reshape(len(ids_list), 256)
            rmse_orb_cdf = ((output_pdos_orb_cdf-target_pdos_orb_cdf)**2).sum(dim=1).sqrt().cpu().detach().numpy()

            output_pdos_orb = output_pdos_diff.reshape(len(ids_list), 255)
            target_pdos_orb = target_pdos_diff.reshape(len(ids_list), 255)
            rmse_orb = ((output_pdos_orb-target_pdos_orb)**2).sum(dim=1).sqrt().cpu().detach().numpy()

            id_error_df = pd.DataFrame({"id": ids_list, "element": elements, "site": sites, "orbital": orbital_types, "rmse": rmse_orb, "rmse_cdf": rmse_orb_cdf})
            id_error_df_list.append(id_error_df)
  
        if save_output:
            if use_cdf:
                dos_to_save = torch.diff(output_dos, dim=1)/e_diff
                dos_to_save_cdf = output_dos
            else:
                dos_to_save = output_dos
                dos_to_save_cdf = torch.cumsum(output_dos, dim=1)*e_diff
            
            # DOS dataframes to save
            out_dos_data = pd.DataFrame(dos_to_save.data.cpu().detach().numpy())
            out_dos_data_cdf = pd.DataFrame(dos_to_save_cdf.data.cpu().detach().numpy())
            target_dos_data = pd.DataFrame(target_dos_cpu.detach().numpy())
            target_dos_data_cdf = pd.DataFrame(target_dos_cpu_cdf.detach().numpy())
            # material id and DOS 
            output_and_id = pd.concat([pd.DataFrame(data.material_id), out_dos_data, target_dos_data], axis = 1, ignore_index=True, sort=False)
            output_and_id_cdf = pd.concat([pd.DataFrame(data.material_id), out_dos_data_cdf, target_dos_data_cdf], axis = 1, ignore_index=True, sort=False)
            save_dos_list.append(output_and_id)
            save_dos_list_cdf.append(output_and_id_cdf)

            if save_pdos:
                if use_cdf:
                    pdos_to_save = output_pdos_diff
                    pdos_to_save_cdf = output_pdos
                    target_pdos_cpu = target_pdos_diff.data.cpu()
                else:
                    pdos_to_save = output_pdos
                    pdos_to_save_cdf = torch.cumsum(output_pdos, dim=1)*e_diff
                
                ids_list = []
                for orbitals, id in zip(data.orbital_types, data.material_id):
                    ids_list.append([id]*len(orbitals))

                ids_list = list(itertools.chain.from_iterable(ids_list))
                sites = list(itertools.chain.from_iterable(data.sites))
                elements = list(itertools.chain.from_iterable(data.elements))
                orbital_types = list(itertools.chain.from_iterable(data.orbital_types))

                out_pdos_data = pd.DataFrame(pdos_to_save.data.cpu().detach().numpy())
                out_pdos_data_cdf = pd.DataFrame(pdos_to_save_cdf.data.cpu().detach().numpy())
                target_pdos_data = pd.DataFrame(target_pdos_cpu.detach().numpy())
                target_pdos_data_cdf = pd.DataFrame(target_pdos_cpu_cdf.detach().numpy())

                output_and_id = pd.concat([pd.DataFrame(ids_list), pd.DataFrame(elements), pd.DataFrame(sites), pd.DataFrame(orbital_types), out_pdos_data, target_pdos_data], axis = 1, ignore_index=True, sort=False)
                output_and_id_cdf = pd.concat([pd.DataFrame(ids_list), pd.DataFrame(elements), pd.DataFrame(sites), pd.DataFrame(orbital_types), out_pdos_data_cdf, target_pdos_data_cdf], axis = 1, ignore_index=True, sort=False)
                save_pdos_list.append(output_and_id)
                save_pdos_list_cdf.append(output_and_id_cdf)
            
    if save_dos:
        total_output_dos = pd.concat(save_dos_list)
        total_output_dos_cdf = pd.concat(save_dos_list_cdf)
        filename = f'best_model_output_total_dos_fold_{fold}.csv'
        filename_cdf = f'best_model_output_total_dos_cdf_fold_{fold}.csv'
        if test:
            filename = f'test_output_total_dos.csv'
            filename_cdf = f'test_output_total_dos_cdf.csv'
        total_output_dos.to_csv('test_outputs/%s/'%save_path + filename, header=False, index=False)
        total_output_dos_cdf.to_csv('test_outputs/%s/'%save_path + filename_cdf, header=False, index=False)

    if save_pdos:
        total_output_pdos = pd.concat(save_pdos_list)
        total_output_pdos_cdf = pd.concat(save_pdos_list_cdf)
        filename = f'best_model_output_pdos_fold_{fold}.csv'
        filename_cdf = f'best_model_output_pdos_cdf_fold_{fold}.csv'
        if test:
            filename = f'test_output_pdos.csv'
            filename_cdf = f'test_output_pdos_cdf.csv'
        total_output_pdos.to_csv('test_outputs/%s/'%save_path + filename, header=False, index=False)
        total_output_pdos_cdf.to_csv('test_outputs/%s/'%save_path + filename_cdf, header=False, index=False)

    epoch_loss = running_loss/n_iter
    epoch_pdos_rmse = running_pdos_rmse/n_iter
    epoch_cdf_pdos_rmse = running_cdf_pdos_rmse/n_iter

    if save_id_rmse and epoch%save_id_rmse_interv == 0:
        id_error_total_df = pd.concat(id_error_df_list)
        id_error_total_df.to_csv('test_outputs/%s/'%save_path + f"orbital_rmse_epoch_{epoch}_fold_{fold}.csv", index=False)
    
    return epoch_loss, epoch_pdos_rmse, epoch_cdf_pdos_rmse