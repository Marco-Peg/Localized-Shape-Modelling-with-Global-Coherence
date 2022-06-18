import numpy as np
import igl
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors


def metric_distorsion(vertices, gt, faces, sources, targets=None, t=1e-1):
    D = np.zeros([vertices.shape[0], sources.shape[0]])
    for s in range(sources.shape[0]):
        D[:,s] = igl.heat_geodesic(vertices, faces, t, np.array([sources[s]])) + 1e-08
        Dgt = igl.heat_geodesic(gt, faces, t, np.array([sources[s]])) + 1e-08
        D[:,s] = np.absolute( (D[:,s] - Dgt) / Dgt )

    return D

def area_error(vertices, gt, faces):
    A = igl.massmatrix(vertices,faces).diagonal()
    Agt = igl.massmatrix(gt,faces).diagonal()

    error = np.absolute( (A-Agt) / Agt)

    return error

def alignmet_error(vertices, gt, faces, reg_inds):
    scale, R, t = igl.procrustes(vertices[reg_inds,:], gt[reg_inds,:], include_scaling=False, include_reflections=False)
    v_proc = scale * vertices @ R + t.transpose()
    error = np.mean(np.power(v_proc - gt, 2), axis=1)

    return scale, R, t, v_proc, error

def compute_centerSubsample(vertices, faces, n_samples=1000, show=True):
    Aff = igl.adjacency_matrix(faces.astype(int))
    model = AgglomerativeClustering(linkage='ward', connectivity=Aff, n_clusters=n_samples)
    ind2label =  model.fit_predict(vertices)
    labels = model.labels_

    centers = np.zeros([n_samples], dtype=int)
    for i in range(n_samples):
        indx = np.argwhere(labels == i).flatten()
        c = np.mean(vertices[indx, :], axis=0)

        nbrs = NearestNeighbors(n_neighbors=1).fit(vertices[indx, :])
        nbr_ind = nbrs.kneighbors([c], return_distance=False)[0, :]
        centers[i] = indx[nbr_ind]

    # collapse faces
    f_reduce = list()
    for i in range(faces.shape[0]):
        f = ind2label[faces[i,:]]
        if np.unique(f).size == 3:
            f_reduce.append(f)
    _, f_ind = np.unique(np.sort(f_reduce, axis=1), axis=0, return_index=True)
    f_reduce = np.array(f_reduce)[f_ind,:]

    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices[centers, :]), o3d.utility.Vector3iVector(f_reduce))
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    mesh.orient_triangles()
    f_reduce = np.asarray(mesh.triangles)

    if show:
        o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices[centers, :]))
        o3d.visualization.draw_geometries([pcd])

    return centers,f_reduce

def reduce_faces(region_ind, faces):
    inds_head = np.squeeze(region_ind)
    ind_trad = dict()
    for i in range(inds_head.shape[0]):
        ind_trad[inds_head[i]] = i
    f_region = faces[np.all(np.isin(faces, inds_head), axis=1), :]
    for i in range(f_region.shape[0]):
        for j in range(f_region.shape[1]):
            f_region[i, j] = ind_trad[f_region[i, j]]
    return f_region

if __name__ == "__main__":
    import json
    from dataset import datasetCreator
    import tensorflow as tf
    import os
    import hdf5storage
    import open3d as o3d

    dataset_name = "SURREAL"
    models = ( "PAT15+15",)
    dataset = datasetCreator(dataset_name)
    meshes, f = dataset.load_meshes()
    train_index, evals_index = dataset.split_validation()
    regions_ind = [dataset.get_region_indexes()]
    n_samples = 100
    if not os.path.exists("{}/centers_{}.mat".format(dataset_name, n_samples)):
        centers_ind, f_reduce = compute_centerSubsample(meshes[0,:,:], f, n_samples=n_samples)
        hdf5storage.savemat("{}/centers_{}.mat".format(dataset_name, n_samples), {'ind_centers':centers_ind, 'f_centers':f_reduce.astype(int)}, oned_as='column', store_python_metadata=True)
    else:
        centers_ind = hdf5storage.loadmat("{}/centers_{}.mat".format(dataset_name, n_samples))['ind_centers']

    reg_ind = np.argwhere(regions_ind[0]).flatten()
    f_region = reduce_faces(reg_ind,f)
    if not os.path.exists("{}/centersReg_{}.mat".format(dataset_name, n_samples)):
        centersReg_ind, fReg_reduce = compute_centerSubsample(meshes[0,reg_ind,:], f_region, n_samples=n_samples)
        hdf5storage.savemat("{}/centersReg_{}.mat".format(dataset_name, n_samples), {'ind_centers':centersReg_ind, 'f_centers':fReg_reduce.astype(int)}, oned_as='column', store_python_metadata=True)
    else:
        centersReg_ind = hdf5storage.loadmat("{}/centersReg_{}.mat".format(dataset_name, n_samples))['ind_centers']

    for model_name in models:
        print(model_name)
        with open("{}/{}/params.json".format(dataset_name, model_name), 'r') as file:
            params = json.load(file)
        # compute test_vertices
        if "local_operator" in params.keys() and len(params['n_autoval_local']) > 0:
            eigs, regions_ind = dataset.load_globalLocalInput(n_autoval_global=params['n_autoval'],
                                                              n_autoval_local=params['n_autoval_local'],
                                                              local_operator=params["local_operator"],
                                                              local_regions=params['regions_name'],
                                                              diff_eigs=params['diff_eigs'])
            n_autoval_local = params['n_autoval_local'][0] - 1
        else:
            eigs = dataset.load_globalEig(n_autoval=params['n_autoval'])
            if params['diff_eigs']:
                eigs = np.diff(eigs)
            n_autoval_local = 0


        evals_samples = meshes[evals_index, :, :]
        evals_eigs = eigs[evals_index, :]
        # train_samples = meshes[train_index, :, :]
        # train_eigs = eigs[train_index, :]

        model_dir = "{}/{}/test_model".format(dataset_name, model_name)
        model = tf.keras.models.load_model(model_dir)
        model.trainable = False
        n_autoval_global = params['n_autoval'] - 1

        sub_indx = np.linspace(0, evals_samples.shape[0], 20, endpoint=False, dtype=int)
        recons = np.zeros([20,meshes.shape[1],3])
        MSEErrors = np.zeros([20,meshes.shape[1]])
        alignRecons = np.zeros([20,meshes.shape[1],3])
        areaErrors = np.zeros([20,meshes.shape[1]])
        metricErrors = np.zeros([20,meshes.shape[1]])
        metricRegErrors = np.zeros([20,reg_ind.shape[0]])
        alignErrors = np.zeros([20,meshes.shape[1]])
        MSEMeanError = areaMeanError = metricMeanError = areaRMeanError = metricRMeanError = alignMeanError = 0
        isub = 0

        print("Model;Area error;Metric error;Align error;Area error region;Metric error region")
        for i_sample in range(evals_index.shape[0]):
            input_sample = np.expand_dims(evals_eigs[i_sample, :], 0)
            recon = model(input_sample, training=False).numpy()
            recon = np.squeeze(recon)

            ## MSE
            MSE_err = tf.keras.losses.MSE(meshes[i_sample,:,:], recon).numpy()
            MSEMeanError += np.mean(MSE_err)

            ## area_error
            area_err = area_error(recon, evals_samples[i_sample,:,:], f)
            areaMeanError += np.mean(area_err)
            areaRMeanError += np.mean(area_err[reg_ind])

            ## Metric distorsion
            D = metric_distorsion(recon, evals_samples[i_sample,:,:], f, centers_ind) # n_vert x n_sources
            Derr = np.mean(D,axis=1)
            metricMeanError += np.mean(Derr)

            Dreg = metric_distorsion(recon[reg_ind], evals_samples[i_sample,reg_ind,:], f_region, centersReg_ind) # n_vert x n_reg
            DRerr = np.mean(Dreg, axis=1)
            metricRMeanError += np.mean(DRerr)

            # alignment error
            scale, R, t, template_Tr, align_err = alignmet_error(recon, evals_samples[i_sample,:,:], f, reg_ind)
            alignMeanError += np.mean(align_err[reg_ind])

            print("{}: {:.2e}; {:.2e}; {:.2e}; {:.2e}; {:.2e}".format(i_sample, np.mean(area_err), np.mean(Derr), np.mean(align_err), np.mean(area_err[reg_ind]), np.mean(Derr[reg_ind])))
            if i_sample in sub_indx:
                recons[isub, :, :] = np.squeeze(recon)
                MSEErrors[isub, :] = MSE_err
                areaErrors[isub,:] = area_err
                metricErrors[isub,:] = Derr
                metricRegErrors[isub,:] = DRerr
                alignErrors[isub,:] = align_err
                alignRecons[isub,:,:] = template_Tr
                isub += 1

        areaMeanError = areaMeanError / evals_samples.shape[0]
        metricMeanError = metricMeanError / evals_samples.shape[0]
        alignMeanError = alignMeanError / evals_samples.shape[0]
        areaRMeanError = areaRMeanError / evals_samples.shape[0]
        metricRMeanError = metricRMeanError / evals_samples.shape[0]
        MSEMeanError = MSEMeanError / evals_samples.shape[0]

        savepath = "{}/{}/intrinsic_errors/".format(dataset_name,model_name)
        os.makedirs(savepath, exist_ok=True)

        with open(savepath+"meanErrors.txt", 'w+') as fw:
            fw.write("Model;Area error;Metric error;Align error;Area error region;Metric error region\n")
            fw.write("{};{:.2e};{:.2e};{:.2e};{:.2e};{:.2e}\n".format(model_name,areaMeanError,metricMeanError,alignMeanError,areaRMeanError,metricRMeanError))
        if not os.path.exists("{}/intrinsic_errors.txt".format(dataset_name)):
            with open("{}/intrinsic_errors.txt".format(dataset_name), 'w+') as fw:
                fw.write("Model;Area error;Metric error;Align error;Area error region;Metric error region\n")
        with open("{}/intrinsic_errors.txt".format(dataset_name), 'a+') as fw:
            fw.write("{};{:.2e};{:.2e};{:.2e};{:.2e};{:.2e}\n".format(model_name,areaMeanError,metricMeanError,alignMeanError,areaRMeanError,metricRMeanError))

        hdf5storage.savemat(savepath+"errors_samples.mat",
                            {'recons':recons, 'gts':evals_samples[sub_indx,:,:],
                             'f': f, 'indx':sub_indx, 'f_region': f_region,
                             "areaErrors": areaErrors, "MSEErrors": MSEErrors,
                             'metricErrors':metricErrors, 'metricRegErrors': metricRegErrors,
                             'alignErrors':alignErrors, 'alignRecons':alignRecons},
                            oned_as='column', store_python_metadata=True)