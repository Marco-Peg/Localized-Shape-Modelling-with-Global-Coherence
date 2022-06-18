import open3d as o3d
import hdf5storage
import numpy as np
from scipy.sparse import identity, diags, csr_matrix, eye
from scipy.sparse.linalg import eigsh
import igl
from utils import reduce_faces

def plot_with_color(sample, f, colors):
    # Plots
    # Our recontruction
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh2 = o3d.geometry.TriangleMesh()
    mesh2.vertices = o3d.utility.Vector3dVector( sample)
    mesh2.triangles = o3d.utility.Vector3iVector(f)
    mesh2.compute_vertex_normals()
    mesh2.vertex_colors = o3d.utility.Vector3dVector(colors[:, 0:3])
    vis.add_geometry(mesh2)
    vis.run()
    vis.capture_screen_image("region.jpg")
    vis.destroy_window()

def testLocality(evecs,v,f, region_mask, txt="", threshold=1e2):
    A = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
    norm_evecs = np.linalg.norm(evecs, axis=1)
    # region
    out_region_energy = np.sum(A @ ((1-region_mask) * norm_evecs))

    if out_region_energy > threshold:
        print("{} locality check failed: {:.2e}".format(txt,out_region_energy))


def hamiltonian(v,f,region_ind,dim=100, pot_value=1):
    # Compute Laplacian and Area Matrix
    p = np.ones((v.shape[0])) * pot_value
    p[region_ind] = 0
    P = diags(p)

    M = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
    L = -igl.cotmatrix(v, f)

    # Compute EigenDecomposition
    try:
        evals, evecs = eigsh((L+M@P), dim, M, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)
    except:
        evals, evecs = eigsh((L+M@P) + 1e-8 * identity(v.shape[0]), dim,
                             M, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)

    return evals, evecs

def laplacian_patch(v,f,region_ind,dim=100, dirichlet_bc = True):
    v_r = v[region_ind,:]
    M = igl.massmatrix(v_r, f, igl.MASSMATRIX_TYPE_VORONOI)
    L = -igl.cotmatrix(v_r, f)

    if dirichlet_bc:
        e = igl.boundary_facets(f)
        v_b = np.unique(e)
        v_in = np.setdiff1d(np.arange(v_r.shape[0]), v_b)
        L_full = L.copy()
        L = csr_matrix(L_full.shape)
        L[np.ix_(v_b, v_b)] = eye(v_b.shape[0])
        L[np.ix_(v_in, v_in)] = L_full[np.ix_(v_in, v_in)]
        del L_full
        M_full = M.copy()
        M = csr_matrix(M_full.shape)
        M[np.ix_(v_in, v_in)] = M_full[np.ix_(v_in, v_in)]
        del M_full

    # Compute EigenDecomposition
    try:
        evals, evecs_patch = eigsh(L, dim, M, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)
    except:
        evals, evecs_patch = eigsh(L + 1e-8 * identity(v_r.shape[0]), dim,
                             M, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)

    evecs = np.zeros( (region_ind.shape[0], dim))
    evecs[region_ind,:] = evecs_patch

    return evals, evecs

def LMH(vert,f,region_ind, dim=30, dim_global=20, muR=1e3, muO=1e2):
    # Compute Laplacian and Area Matrix
    A = igl.massmatrix(vert, f, igl.MASSMATRIX_TYPE_VORONOI)
    vert = vert / np.sqrt(A.sum())
    A = igl.massmatrix(vert, f, igl.MASSMATRIX_TYPE_VORONOI)
    L = -igl.cotmatrix(vert, f)

    # Compute EigenDecomposition
    try:
        _, evecs_global = eigsh(L, dim_global, A, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)
    except:
        _, evecs_global = eigsh(L + 1e-8 * identity(vert.shape[0]), dim_global,
                             A, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)


    v = 1 - region_ind.astype(float)

    ## basic
    Q = L + muR * A @ diags(v) + muO * A @ evecs_global @ np.transpose(evecs_global) @ A
    evals, evecs = eigsh(np.float32(Q), dim, A, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)

    return evals, evecs

def LMH_from_evecs(vert,f,region_ind, evecs_global, dim=30, muR=1e3, muO=1e2):
    # Compute Laplacian and Area Matrix
    A = igl.massmatrix(vert, f, igl.MASSMATRIX_TYPE_VORONOI)
    L = -igl.cotmatrix(vert, f)
    v = 1 - region_ind.astype(float)

    ## basic
    Q = L + muR * A @ diags(v) + muO * A @ evecs_global @ np.transpose(evecs_global) @ A
    evals, evecs = eigsh(Q, dim, A, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)


    return evals, evecs

operator_fun = {
        'hamiltonian': hamiltonian,
        'LMH': LMH,
    }

def create_local_dataset(dataset,region_ind,local_operator,identity=False, dim_eigen = 30, region_name="region", dim_eigen_global = 30, return_eigvets=True):
    print("\n\tCreate the {} local dataset...".format(dataset),end='')

    if not identity or dataset=='TOY':
        data = hdf5storage.loadmat('{0:}/{0:}.mat'.format(dataset), variable_names=('eigvals','v','f'))
        print("..load the vertices...", end='')
        path = "{0}/{0}_{1}_{2}.mat".format(dataset, local_operator,region_name)
    else:
        data = hdf5storage.loadmat('{0:}/{0:}_identity.mat'.format(dataset), variable_names=('eigvals','v','f'))
        print("load identity meshes...", end='')
        path = "{0}/{0}_identity_{1}_{2}.mat".format(dataset, local_operator, region_name)
    dataset = dict()
    if local_operator == "hamiltonian":
        dataset['p_value'] = 1e3 * np.max(data['eigvals'])
    elif local_operator == "LMH":
        dataset['dim_global'] = dim_eigen_global
        dataset['muR'] = 1e3 * np.max(data['eigvals'])
        dataset['muO'] = 1e5
        path = path[:-4] + '_' + str(dim_eigen_global) + '_' + str(dim_eigen) + '.mat'
    verts = data['v'].astype('float32')
    f = np.asarray(data['f'], dtype=int)
    if local_operator == "patch" or local_operator == "patchNbr":
        inds_head = np.squeeze(np.argwhere(region_ind))
        ind_trad = dict()
        for i in range(inds_head.shape[0]):
            ind_trad[inds_head[i]] = i
        f_region = f[np.all(np.isin(f, inds_head), axis=1), :]
        for i in range(f_region.shape[0]):
            for j in range(f_region.shape[1]):
                f_region[i, j] = ind_trad[f_region[i, j]]
        dataset["f_region"] = f_region
    dataset["region_inds"] = region_ind
    print("compute local spectrums...", end='')
    dataset["eigvals"] = np.ndarray([verts.shape[0], dim_eigen], dtype=float)
    if return_eigvets:
        dataset["eigvets"] = np.ndarray([verts.shape[0], verts.shape[1], dim_eigen], dtype=float)
    steps = range(verts.shape[0] // 10, verts.shape[0], verts.shape[0] // 10)
    for i in range(verts.shape[0]):
        if local_operator == "hamiltonian":
            eig_out = hamiltonian(verts[i, :, :], f, region_ind, dim_eigen,
                                    pot_value=dataset['p_value'])
        elif local_operator == "patch":
            eig_out = laplacian_patch(verts[i, :, :], f_region, region_ind, dim_eigen)
        elif local_operator == "patchNbr":
            eig_out = laplacian_patch(verts[i, :, :], f_region, region_ind, dim_eigen)
        elif local_operator == "LMH":
            eig_out = LMH(verts[i, :, :], f, region_ind, dim_eigen,
                            dim_global=dataset['dim_global'],muR=dataset['muR'], muO=dataset['muO'])
        else:
            eig_out = operator_fun[local_operator](verts[i, :, :], f, region_ind, dim_eigen, pot_value=dataset['p_value'])

        testLocality(eig_out[1], verts[i, :, :], f, region_ind)
        dataset["eigvals"][i, :] = eig_out[0]
        if return_eigvets:
            dataset["eigvets"][i, :, :] = eig_out[1]
        print(' ', end='\x1b[1K\r')
        print(" {}/{} ".format(i + 1, verts.shape[0]), end='', flush=True)
    hdf5storage.savemat(path, dataset, oned_as='column', store_python_metadata=True)
    print("...done")
    return dataset

def region_splitter(segment_indx, weights_limit=0.1, inverse_region=False):
    w_r = hdf5storage.loadmat('./weights_prior.mat')['weights_prior']

    regions = np.concatenate([w_r.getcol(i).toarray() for i in segment_indx], axis=1) # Right leg
    regions = np.max(regions, axis=1)
    region_ind = np.where(regions > weights_limit, True, False)
    if inverse_region:
        region_ind = np.logical_not(region_ind)

    return region_ind

def plot_region(verts, f, region, name=""):
    # Plots

    p = np.ones_like(verts) * 0.4
    p[region, :] = [0.4,0.4,1]

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    mesh2 = o3d.geometry.TriangleMesh()
    mesh2.vertices = o3d.utility.Vector3dVector(verts)
    mesh2.triangles = o3d.utility.Vector3iVector(f)
    mesh2.compute_vertex_normals()
    mesh2.compute_triangle_normals()
    mesh2.vertex_colors = o3d.utility.Vector3dVector(p)
    vis.add_geometry(mesh2)

    vc = vis.get_view_control()
    vc.set_zoom(0.55)

    vis.run()
    vis.capture_screen_image("region_{}.jpg".format(name))
    vis.destroy_window()

def compute_local_spectr(dataset, segment_indx = [15],
              region_name= "head",
              inverse_region= False,
              identity= True,
              local_operator = "LMH", # LMH , hamiltonian, patch
              n_eig = 15,
              n_eig_global= 15,
              return_eigvet= True):
    if dataset == 'CUBE':
        region_inds = hdf5storage.loadmat('{0:}/{0:}.mat'.format(dataset), variable_names=('reg_ind',))['reg_ind'].astype(bool)
        region_inds = region_inds.squeeze()
        region_name = "r"
        identity = False
    else:
        if inverse_region:
            region_name = "NO" + region_name
        region_inds = region_splitter(segment_indx, inverse_region=inverse_region)

        template = hdf5storage.loadmat('SURREAL/template.mat', variable_names=['template', 'f'])
        plot_region(template['template'],template['f'],region_inds, region_name)

    # Loading data
    d = create_local_dataset(dataset, region_inds,
                             local_operator=local_operator, identity=identity,
                             region_name=region_name, dim_eigen=n_eig,
                             dim_eigen_global=n_eig_global, return_eigvets=return_eigvet)
    return d

if __name__ == "__main__":
    params = {'dataset': "SURREAL",
              'segment_indx': [19],
              'region_name': "elbow",
              'inverse_region': False,
              'identity': True,
              'local_operator' : "patch", # LMH , hamiltonian, patch
              'n_eig' : 15+1,
              'n_eig_global': 15+1,
              'return_eigvet': False,
              }
    compute_local_spectr(**params)
