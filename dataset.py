import os
import hdf5storage
import numpy as np
from utils_dataset import laplace
from scipy.spatial import distance_matrix
from utils import reduce_faces
from Local_spectr import hamiltonian, LMH, laplacian_patch, testLocality

class Dataset():

    def __init__(self, dataset_name=""):
        self.dataset_name = dataset_name
        self.path = '{0}/{0}.mat'.format(self.dataset_name)

    def create_dataset(self, **kwargs):
        raise NotImplementedError

    def create_local(self, region_ind, local_operator, dim_eigen=31, region_name="r",
                 dim_eigen_global=30, return_eigvets=False):
        raise NotImplementedError

    def split_validation(self,valid_perc=0.1):
        evals_index = np.linspace(0, self.n_sample, int(valid_perc * self.n_sample), endpoint=False, dtype=int)
        train_index = np.setdiff1d(np.arange(self.n_sample), evals_index)

        return train_index, evals_index

    def load_meshes(self, **kwargs):
        print("\n\tLoading {} meshes...".format(self.dataset_name), end='')

        if not os.path.isfile(self.path):
            data = self.create_dataset()
        else:
            data = hdf5storage.loadmat(self.path, variable_names=('v', 'f'))

        meshes = data['v']
        f = data['f'].astype(int)

        self.shape_meshes = meshes.shape
        self.n_sample = meshes.shape[0]
        print("...done!")
        return meshes,f


    def load_globalEig(self, n_autoval=30, **kwargs):
        print("\n\tLoading {} global eigenvalues...".format(self.dataset_name), end='')

        if not os.path.isfile(self.path):
            data = self.create_dataset(n_autoval=n_autoval, **kwargs)
        else:
            data = hdf5storage.loadmat(self.path, variable_names=('eigvals',))
            if data['eigvals'].shape[1] < n_autoval:
                data = self.create_dataset(n_autoval=n_autoval+1, **kwargs)

        eigs = data['eigvals'][:, :n_autoval]

        self.n_sample = eigs.shape[0]
        print("...done!")
        return eigs

    def load_localEig(self, n_autoval_local=[10], local_regions=None, local_operator=None, n_autoval=0, **kwargs):
        print("\n\tLoading {} local eigenvalues...".format(self.dataset_name), end='')

        eigs = list()
        regions_ind = list()
        i = 0
        for region in local_regions:
            local_path = '{0}_{1}_{2}.mat'.format(self.path[:-4], local_operator, region)
            if local_operator == 'LMH':
                local_path = local_path[:-4] + '_' + str(n_autoval) + '_' + str(n_autoval_local[i]) + '.mat'
            if not os.path.isfile(local_path):
                raise ValueError("Region {} for local operator {} not found".format(region, local_operator))

            else:
                data_region = hdf5storage.loadmat(local_path, variable_names=('eigvals', 'region_inds'))
            eigs.append(np.copy(data_region['eigvals'][:, :n_autoval_local[i]]))
            regions_ind.append(np.copy(data_region['region_inds'].squeeze()).astype(bool))
            i = i + 1

        self.n_sample = eigs[0].shape[0]
        print("...done!")
        return eigs, regions_ind

    def load_globalInput(self,n_autoval,diff_eigs=False):
        eigs = self.load_globalEig(n_autoval)
        if diff_eigs:
            eigs = np.diff(eigs)
        return eigs

    def load_globalLocalInput(self,n_autoval_global,n_autoval_local,local_operator,local_regions,diff_eigs=False,bc=False):
        Leigs, regions_ind = self.load_localEig(n_autoval_local=n_autoval_local, n_autoval=n_autoval_global,
                                                   local_operator=local_operator,local_regions=local_regions)
        if n_autoval_global > 1:
            eigs = self.load_globalEig(n_autoval=n_autoval_global)
            eigs = [eigs] + Leigs
        else:
            eigs = Leigs

        if diff_eigs:
            for i in range(len(eigs)):
                eigs[i] = np.diff(eigs[i])
            if bc and n_autoval_global > 1:
                eigs[1] = np.concatenate((np.expand_dims(Leigs[0][:,0],axis=1),eigs[1][:,:-1]), axis=1)
        eigs = np.concatenate(eigs, axis=-1)
        return eigs, regions_ind

class DatasetSURREAL(Dataset):
    def __init__(self,identity=True):
        self.dataset_name = "SURREAL"
        self.identity = identity
        self.path = '{0}/{0}{1}.mat'.format(self.dataset_name, "_identity" if identity else "")

    def create_dataset(self, n_autoval=31, **kwargs):
        print("\n\tCreate the {} dataset...".format(self.dataset_name), end='')
        data = hdf5storage.loadmat('../Autoencoder/SURREAL/dataset_new_10k.mat')
        dataset = dict()
        dataset['f'] = np.asarray(data['f'], dtype=int)
        n_autoval = n_autoval
        if not self.identity:
            path = "SURREAL/SURREAL.mat"
            print("..load the vertices...", end='')
            dataset['v'] = data['v'].astype('float32')
        else:
            path = "SURREAL/SURREAL_identity.mat"
            print("load identity meshes...", end='')
            dataset['v'] = data['v_neutro'].astype('float32')
            _, indices = np.unique(data['beta_params'], return_index=True, axis=0)
            dataset['v'] = dataset['v'][indices, :, :]

        dataset['v'] = dataset['v'] - np.mean(dataset['v'], axis=1, keepdims=True)  # center to the origin
        print("compute LBO and eigenvals...", end='')
        dataset["eigvals"] = np.ndarray([dataset['v'].shape[0], n_autoval], dtype=float)
        dataset["eigvets"] = np.ndarray([dataset['v'].shape[0], dataset['v'].shape[1], n_autoval], dtype=float)
        print("")
        for i in range(dataset['v'].shape[0]):
            dataset["eigvals"][i, :], dataset["eigvets"][i, :, :], dataset['v'][i, :, :] = laplace(
                dataset['v'][i, :, :],
                dataset['f'], n_autoval)
            print(' ', end='\x1b[1K\r')
            print(" {}/{} ".format(i,dataset['v'].shape[0]), end='')
        hdf5storage.savemat(path, dataset, oned_as='column', store_python_metadata=True)
        print("...done")
        return dataset

    def get_region_indexes(self):
        data = hdf5storage.loadmat("SURREAL/SURREAL_head.mat", variable_names=('region_inds',))['region_inds'].astype(bool).squeeze()
        return data


class DatasetTOY(Dataset):
    def __init__(self):
        self.dataset_name = "TOY"
        self.path = '{0}/{0}.mat'.format(self.dataset_name)

    def create_dataset(self,n_autoval=31):
        print("\n\tCreate the {} dataset...".format(self.dataset_name), end='')
        data = hdf5storage.loadmat('../toy_dataset/toy_dataset.mat')
        dataset = dict()
        dataset['f'] = np.asarray(data['faces'] - 1, dtype=int)
        print("..load the vertices...", end='')
        dataset['v'] = data['vertices'].astype('float32')
        dataset['v'] = dataset['v'] - np.mean(dataset['v'], axis=1, keepdims=True)  # center to the origin
        dataset['reg_ind'] = np.zeros([dataset['v'].shape[1]])
        dataset['reg_ind'][data['indexes'] - 1] = 1
        scales = data['scale']
        frames = data['frame']
        del data

        # check for double
        _, unique_ind = np.unique(np.reshape(dataset["v"], (dataset["v"].shape[0],-1)), axis=0, return_index=True)
        dataset['v'] = dataset['v'][unique_ind, :, :]
        scales = scales[:, unique_ind]
        frames = frames[:, unique_ind]

        print("compute LBO and eigenvals...", end='')
        dataset["eigvals"] = np.ndarray([dataset['v'].shape[0], n_autoval], dtype=float)
        dataset["eigvets"] = np.ndarray([dataset['v'].shape[0], dataset['v'].shape[1], n_autoval], dtype=float)
        # steps = range(dataset['v'].shape[0] // 10, dataset['v'].shape[0], dataset['v'].shape[0] // 10)
        steps = np.linspace(0, dataset['v'].shape[0], num=10, dtype=int)
        for i in range(dataset['v'].shape[0]):
            dataset["eigvals"][i, :], dataset["eigvets"][i, :, :], dataset['v'][i, :, :] = laplace(
                dataset['v'][i, :, :], dataset['f'], n_autoval, uniform_area=False)
            if i in steps:
                print(" {:.2f}% ".format(100 * i / dataset['v'].shape[0]), end='')

        # validation split
        a = np.unique(scales)
        dataset['evals_index'] = np.argwhere(scales==a[3]) # same scale
        a = np.unique(frames)
        dataset['evals_index'] = np.concatenate( [dataset['evals_index'], np.argwhere(frames==a[8])]) # same frame
        dataset['evals_index'] = np.concatenate( [dataset['evals_index'], np.argwhere(frames==a[3])]) # same frame
        dataset['evals_index'] = np.unique(dataset['evals_index'])
        dataset ['train_index'] = np.setdiff1d(np.arange(dataset["eigvals"].shape[0]), dataset['evals_index'])

        hdf5storage.savemat("TOY/TOY.mat", dataset, oned_as='column', store_python_metadata=True)
        print("...done")
        return dataset

    def split_validation(self, valid_perc=0.1):
        data = hdf5storage.loadmat(self.path, variable_names=('evals_index','train_index'))
        return data['train_index'].astype(int).squeeze(), data['evals_index'].astype(int).squeeze()

    def load_localEig(self, n_autoval_local=(16,), local_operator=None, n_autoval=0, **kwargs):
        return Dataset.load_localEig(self, local_regions='r', n_autoval_local=n_autoval_local, local_operator=local_operator, n_autoval=n_autoval)

    def get_region_indexes(self):
        data = hdf5storage.loadmat(self.path, variable_names=('reg_ind',))['reg_ind'].astype(bool).squeeze()
        return data

class DatasetCUBE(DatasetTOY):
    def __init__(self):
        self.dataset_name = "CUBE"
        self.path = '{0}/{0}.mat'.format(self.dataset_name)

    def create_dataset(self, n_autoval=31, save_eigvett=False):
        print("\n\tCreate the {} dataset...".format(self.dataset_name), end='')
        data = hdf5storage.loadmat('../toy_dataset/{}_dataset.mat'.format(self.dataset_name))
        dataset = dict()
        dataset['f'] = np.asarray(data['faces'] - 1, dtype=int)
        print("..load the vertices...", end='')
        dataset['v'] = data['vertices'].astype('float32')
        dataset['reg_ind'] = np.zeros([dataset['v'].shape[1]])
        dataset['reg_ind'][data['indexes'] - 1] = 1
        dataset['scales'] = data['scale']
        del data

        print("compute LBO and eigenvals...", end='')
        dataset["eigvals"] = np.ndarray([dataset['v'].shape[0], n_autoval], dtype=float)
        if save_eigvett:
            dataset["eigvets"] = np.ndarray([dataset['v'].shape[0], dataset['v'].shape[1], n_autoval], dtype=float)
        steps = np.linspace(0, dataset['v'].shape[0], num=10, dtype=int)
        for i in range(dataset['v'].shape[0]):
            out_laplace = laplace(dataset['v'][i, :, :], dataset['f'], n_autoval, uniform_area=False)
            dataset["eigvals"][i, :] = out_laplace[0]
            dataset['v'][i, :, :] = out_laplace[2]
            if save_eigvett:
                dataset["eigvets"][i, :, :] = out_laplace[1]
            print(' ', end='\x1b[1K\r')
            print(" {}/{} ".format(i + 1, dataset['v'].shape[0]), end='', flush=True)

        dataset['evals_index'] =  np.linspace(0, dataset["eigvals"].shape[0], int(dataset["eigvals"].shape[0] * 0.1), endpoint=False, dtype=int)
        dataset['train_index'] = np.setdiff1d(np.arange(dataset["eigvals"].shape[0]), dataset['evals_index'])

        hdf5storage.savemat(self.path, dataset, oned_as='column', store_python_metadata=True)
        print("...done")
        return dataset

    def create_local(self, region_ind, local_operator, dim_eigen=31, region_name="r",
                          dim_eigen_global=30, return_eigvets=False):
        print("\n\tCreate the {} local dataset...".format(self.dataset_name), end='')

        data = hdf5storage.loadmat(self.path, variable_names=('eigvals', 'v', 'f', 'frames'))
        print("..load the vertices...", end='')
        path_local = "{0}/{0}_{1}_{2}.mat".format(self.dataset_name, local_operator, region_name)

        dataset = dict()
        if local_operator == "hamiltonian":
            dataset['p_value'] = 1e3 * np.max(data['eigvals'])
        elif local_operator == "LMH":
            dataset['dim_global'] = dim_eigen_global
            dataset['muR'] = 1e3 * np.max(data['eigvals'])
            dataset['muO'] = 1e5
            path = path_local[:-4] + '_' + str(dim_eigen_global) + '_' + str(dim_eigen) + '.mat'
        verts = data['v'].astype('float32')
        f = np.asarray(data['f'], dtype=int)
        if local_operator == "patch":
            dataset["f_region"] = reduce_faces(np.argwhere(region_ind), f)
        dataset["region_inds"] = region_ind
        print("compute local spectrums...", end='')

        frames, uInd = np.unique(data['frames'], return_index=True)
        dataset["eigvals"] = np.ndarray([verts.shape[0], dim_eigen], dtype=float)
        if return_eigvets:
            dataset["eigvets"] = np.ndarray([verts.shape[0], verts.shape[1], dim_eigen], dtype=float)
        for i in range(uInd.shape[0]):
            if local_operator == "hamiltonian":
                eig_out = hamiltonian(verts[uInd[i], :, :], f, region_ind, dim_eigen,
                                      pot_value=dataset['p_value'])
            elif local_operator == "patch":
                eig_out = laplacian_patch(verts[uInd[i], :, :], dataset["f_region"], region_ind, dim_eigen)
            elif local_operator == "LMH":
                eig_out = LMH(verts[uInd[i], :, :], f, region_ind, dim_eigen,
                              dim_global=dataset['dim_global'], muR=dataset['muR'], muO=dataset['muO'])
            else:
                raise NotImplementedError

            testLocality(eig_out[1], verts[i, :, :], f, region_ind)
            # plot_with_color(verts[i, :, :],f,dataset["eigvets"][i, :, :])
            for indU in np.argwhere(data['frame'] == frames[i]).flatten():
                dataset["eigvals"][indU, :] = eig_out[0]
                if return_eigvets:
                    dataset["eigvets"][indU, :, :] = eig_out[1]
            print(' ', end='\x1b[1K\r')
            print(" {}/{} ".format(i + 1, frames.shape[0]), end='', flush=True)

        hdf5storage.savemat(path_local, dataset, oned_as='column', store_python_metadata=True)
        print("...done")
        return dataset


def datasetCreator(name="", **kwargs):
    datasetsList = { "TOY" : DatasetTOY,
                     "CUBE" : DatasetCUBE,
                     "SURREAL" : DatasetSURREAL,
                     }
    if name in datasetsList.keys():
        return datasetsList[name](**kwargs)
    return Dataset(name,**kwargs)