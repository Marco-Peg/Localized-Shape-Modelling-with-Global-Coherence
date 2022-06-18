import hdf5storage
import numpy as np
from scipy.sparse import csr_matrix, identity, eye
from scipy.sparse.linalg import eigsh
import igl
import robust_laplacian


def robust_laplace(v,dim=100, uniform_area = True, dirichlet_bc=False, v_b=None):

    if uniform_area:
        L, M = robust_laplacian.point_cloud_laplacian(v)
        v = v / np.sqrt(M.sum())

    L, M = robust_laplacian.point_cloud_laplacian(v)

    if dirichlet_bc:
        if v_b is None:
            raise ValueError("Specify border vertex")
        ## List of interior indices
        v_in = np.setdiff1d(np.arange(v.shape[0]), v_b)
        # # original code
        L_full = L.copy()
        L = csr_matrix(L_full.shape)
        L[np.ix_(v_b,v_b)] = eye(v_b.shape[0])
        L[np.ix_(v_in,v_in)] = L_full[np.ix_(v_in,v_in)]
        del L_full
        M_full = M.copy()
        M = csr_matrix(M_full.shape)
        M[np.ix_(v_in, v_in)] = M_full[np.ix_(v_in, v_in)]
        del M_full

    try:
        evals, evecs = eigsh(L, dim, M, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)
    except:
        evals, evecs = eigsh(L + 1e-8 * identity(v.shape[0]), dim,
                             M, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)
    return evals, evecs, v


def laplace(v,f,dim=100, dirichlet_bc=False, uniform_area = True):
    # Compute Laplacian and Area Matrix
    if uniform_area:
        M = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
        v = v / np.sqrt(M.sum())
    M = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
    L = -igl.cotmatrix(v, f)

    if dirichlet_bc:
        e = igl.boundary_facets(f)
        v_b = np.unique(e)
        v_in = np.setdiff1d(np.arange(v.shape[0]), v_b)
        L_full = L.copy()
        L = csr_matrix(L_full.shape)
        L[np.ix_(v_b,v_b)] = eye(v_b.shape[0])
        L[np.ix_(v_in,v_in)] = L_full[np.ix_(v_in,v_in)]
        del L_full
        M_full = M.copy()
        M = csr_matrix(M_full.shape)
        M[np.ix_(v_in, v_in)] = M_full[np.ix_(v_in, v_in)]
        del M_full


    # Compute EigenDecomposition
    try:
        evals, evecs = eigsh(L, dim, M, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)
    except:
        evals, evecs = eigsh(L + 1e-8 * identity(v.shape[0]), dim,
                             M, sigma=0.0, which='LM', maxiter=1e9, tol=1.e-15)
    return evals, evecs, v

def select_head(samples):
    w_r = hdf5storage.loadmat('./weights_prior.mat')['weights_prior']
    head_region = np.squeeze(w_r.getcol(15).toarray())
    collar = np.squeeze(np.argwhere(np.logical_and(head_region > 0.01, head_region < 0.05)))
    head_region = np.where(head_region < 0.05, False, True)
    mean_collar = np.mean(samples[:, collar, :], axis=1, keepdims=True)
    head_samples = samples[:, head_region, :] - mean_collar

    return head_samples, head_region, collar

def region_splitter(samples):
    w_r = hdf5storage.loadmat('./weights_prior.mat')['weights_prior']
    head_region = np.squeeze(w_r.getcol(15).toarray())
    collar = np.squeeze(np.argwhere(np.logical_and(head_region > 0.01, head_region < 0.05)))
    head_region = np.where(head_region < 0.05, False, True)
    mean_collar = np.mean(samples[:, collar, :], axis=1, keepdims=True)
    head_samples = samples[:, head_region, :] - mean_collar

    regions = np.concatenate([w_r.getcol(i).toarray() for i in [1, 4, 7, 10]], axis=1)
    regions = np.sum(regions, axis=1)
    limit_regions = np.squeeze(np.argwhere(np.logical_and(regions > 0.3, regions < 0.5)))
    regions = np.where(regions < 0.5, False, True)
    mean_limit = np.mean(samples[:, limit_regions, :], axis=1, keepdims=True)
    region_samples = samples[:, regions, :] - mean_limit

    return {'samples': [head_samples,region_samples],
            'bool_region': [head_region,regions],
            'mean_limit': [mean_collar,mean_limit],
            'names': ['Head', 'LegR']}