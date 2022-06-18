import igl
from scipy.sparse import find
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def plot_loss(path,train_error,test_error,show=False):
    plt.plot(train_error, label="Train")
    plt.plot(test_error, label="Test", color='red')
    plt.yscale("log")
    # plt.legend(("Train", "Test"))
    plt.legend()
    plt.savefig(path + "loss.jpg")
    if show:
        plt.show()
    plt.close()

def plot_mesh(sample, f, colors=None, show=False, save=False, name="meshPlot", rotation_xyz=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh2 = o3d.geometry.TriangleMesh()
    mesh2.vertices = o3d.utility.Vector3dVector(sample)
    mesh2.triangles = o3d.utility.Vector3iVector(f)
    mesh2.compute_vertex_normals()
    if colors is not None:
        mesh2.vertex_colors = o3d.utility.Vector3dVector(colors[:, 0:3])
    if rotation_xyz is not None:
        R = o3d.geometry.get_rotation_matrix_from_xyz(rotation_xyz)
        mesh2.rotate(R, center=mesh2.get_center())
    vis.add_geometry(mesh2)
    vis.update_renderer()
    vis.poll_events()
    if show:
        vis.run()
    if save:
        vis.capture_screen_image("{}.png".format(name))
    vis.destroy_window()

def reduce_faces(region_ind, faces):
    inds_head = np.squeeze(region_ind)
    ind_trad = dict()
    for i in range(inds_head.shape[0]):
        ind_trad[inds_head[i]] = i
    f_region = faces[np.all(np.isin(faces, inds_head), axis=1), :]  # seleziono triangoli della regione
    for i in range(f_region.shape[0]):
        for j in range(f_region.shape[1]):
            f_region[i, j] = ind_trad[f_region[i, j]]
    return f_region