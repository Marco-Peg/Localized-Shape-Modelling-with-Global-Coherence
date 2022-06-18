import tensorflow as tf
import matplotlib.cm as cm
from open3d import *
import hdf5storage
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

# from TrainingLocal import samples_mesh

def test_log(path, inputs, f, model=None, indx=None, targets=None, regions=[], rotation_xyz=None):
    if indx is None:
        indx = range(inputs.shape[0])
    os.makedirs(path, exist_ok=True)

    vis = visualization.Visualizer()
    vis.create_window()

    if targets is None:
        target = inputs[0, :]
    else:
        target = targets[0, :, :]
    err = np.zeros( (np.shape(indx)[0], target.shape[0]))
    recons = np.zeros( (np.shape(indx)[0], target.shape[0],3))
    mean = [0] * (len(regions) + 1)
    with open(path + 'report.txt', 'w') as fh:
        fh.write("Mean reconstrucion error(MSE):\n")
        print("Mean reconstrucion error(MSE):")
        i_err = -1
        for i_test in indx:
            i_err = 1 + i_err
            input_sample = np.expand_dims(inputs[i_test, :], 0)
            if targets is None:
                target = inputs[i_test,:]
            else:
                target = targets[i_test,:,:]

            recon = model(input_sample, training=False).numpy()
            recon = np.squeeze(recon)
            # Compute error

            recons[i_err,:,:] = recon
            err[i_err, :] = tf.keras.losses.mean_squared_error(target, recon).numpy()
            mean[0] = mean[0] + np.mean(err[i_err,:])
            for i in range(len(regions)):
                mean[i+1] = mean[i+1] + np.mean(err[i_err,regions[i]])
            err_str = "Model {}: {:.3e} +- {:.3e}".format(i_test, np.mean(err[i_err,:]), np.std(err[i_err,:]))
            for i in range(len(regions)):
                err_str += "; {:.3e}".format(np.mean(err[i_err,regions[i]]))
            print(err_str)
            fh.write(err_str + "\n")

        max_err = np.max(err[:,:])
        hdf5storage.savemat(path + "errors_samples.mat",
                            {'recons': recons, 'gts':targets[indx,:,:], 'errors':err, 'f':f, 'indx':indx},
                            oned_as='column', store_python_metadata=True)
        i_err = -1
        if rotation_xyz is not None:
            R = geometry.get_rotation_matrix_from_xyz(rotation_xyz)
        for i_test in indx:
            i_err = 1 + i_err
            colors = cm.Reds(err[i_err,:] / max_err) - [0.2,0.2,0.2,0]

            if targets is None:
                target = inputs[i_test,:]
            else:
                target = targets[i_test,:,:]
            # Plots
            mesh2 = geometry.TriangleMesh()
            mesh2.vertices = utility.Vector3dVector(target)
            mesh2.triangles = utility.Vector3iVector(f)
            mesh2.compute_vertex_normals()

            # Reconstruction
            mesh3 = geometry.TriangleMesh()
            mesh3.vertices = utility.Vector3dVector(recons[i_err,:,:])
            mesh3.triangles = utility.Vector3iVector(f)
            mesh3.vertex_colors = utility.Vector3dVector(colors[:, 0:3])
            mesh3.compute_vertex_normals()
            mesh3.compute_triangle_normals()

            if rotation_xyz is not None:
                mesh2.rotate(R, center=mesh2.get_center())
                mesh3.rotate(R, center=mesh3.get_center())
            dist = (np.max(np.asarray(mesh2.vertices)[:, 0]) - np.min(np.asarray(mesh2.vertices)[:, 0])) * 1.1
            mesh3.translate([dist , 0, 0])

            # Visualize Point Cloud
            vis.reset_view_point(True)
            vis.add_geometry(mesh2)
            vis.add_geometry(mesh3)

            # vc.set_zoom(0.55)
            vis.poll_events()
            vis.capture_screen_image(path + "Model_{}.png".format(i_test),True)
            vis.clear_geometries()

        mean = [m/np.shape(indx)[0] for m in mean]
        err_str = "Mean: {:.4e} ".format(mean[0])
        for i in range(len(regions)):
            err_str += "; {:.3e}".format(mean[i+1])
        print(err_str)
        fh.write(err_str + "\n")
    del model
    vis.destroy_window()

def test_NN(path, test_eig, train_eig, test_shapes, train_shapes,f, model_dir='test_model', model=None, indx=None, template=None, batch_size=32, rotation_xyz=None):
    # load the model
    os.makedirs(path, exist_ok=True)

    # NN search
    nbrs = NearestNeighbors(n_neighbors=1).fit(train_eig)
    _, indices = nbrs.kneighbors(test_eig)
    nn_shapes = train_shapes[np.squeeze(indices)]
    nn_errors = np.mean(tf.keras.losses.MSE(test_shapes, nn_shapes).numpy(), axis=-1)

    # model reconstruction
    model_errors = None
    dataset = tf.data.Dataset.from_tensor_slices((test_eig,test_shapes)).batch(batch_size)
    if indx is not None:
        indx = np.sort(indx)
        nn_shapes = nn_shapes[indx,:,:]
        model_shapes = None
        i_indx = 0
        i_batch = 0
    for samples_mesh in dataset:
        generated_mesh = model(samples_mesh[0], training=False)
        err = np.mean(tf.keras.losses.mean_squared_error(samples_mesh[1], generated_mesh).numpy(), axis=-1)
        if model_errors is None:
            model_errors = err
        else:
            model_errors = np.concatenate((model_errors,err))
        # retrieve shape for visualization
        if indx is not None:
            i_batch += 1
            while i_indx < indx.shape[0] and indx[i_indx] < i_batch * batch_size :
                shape = np.expand_dims(generated_mesh[indx[i_indx] - (i_batch-1) * batch_size ,:,:].numpy(), axis=0)
                if model_shapes is None:
                    model_shapes = shape
                else:
                    model_shapes = np.concatenate((model_shapes, shape), axis=0)
                i_indx += 1

    # compare errors
    perc_model = np.sum(model_errors < nn_errors) / model_errors.shape[0]
    perc_nn = np.sum(model_errors > nn_errors) / model_errors.shape[0]
    print("Model: {:.2%}; NN: {:.2%}".format(perc_model,perc_nn))
    model_errors = np.concatenate((model_errors, [perc_model]))
    nn_errors = np.concatenate((nn_errors, [perc_nn]))

    # save errors
    pd.DataFrame(np.stack((model_errors, nn_errors, np.concatenate((np.squeeze(indices), [0]))), axis=1), columns=("Model_errors", "NN_errors", "NN_index"),
                 index=[str(i) for i in range(model_errors.shape[0]-1)]+['Total_percentage'])\
        .to_csv("{}errors.csv".format(path), float_format='%1.3e')
    if indx is None:
        return

    # plot examples
    vis = visualization.Visualizer()
    vis.create_window()
    vc = vis.get_view_control()
    if rotation_xyz is not None:
        R = geometry.get_rotation_matrix_from_xyz(rotation_xyz)
    for i in range(indx.shape[0]):
        # Original
        mesh = geometry.TriangleMesh()
        mesh.vertices = utility.Vector3dVector(test_shapes[indx[i]])
        mesh.triangles = utility.Vector3iVector(f)
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        # Models shape
        mesh2 = geometry.TriangleMesh()
        mesh2.vertices = utility.Vector3dVector(model_shapes[i] )
        mesh2.triangles = utility.Vector3iVector(f)
        mesh2.compute_vertex_normals()
        mesh2.compute_triangle_normals()
        # NN shape
        mesh3 = geometry.TriangleMesh()
        mesh3.vertices = utility.Vector3dVector(nn_shapes[i])
        mesh3.triangles = utility.Vector3iVector(f)
        mesh3.compute_vertex_normals()
        mesh3.compute_triangle_normals()

        if rotation_xyz is not None:
            mesh.rotate(R, center=mesh.get_center())
            mesh2.rotate(R, center=mesh2.get_center())
            mesh3.rotate(R, center=mesh3.get_center())
        dist = (np.max(np.asarray(mesh.vertices)[:,0]) - np.min(np.asarray(mesh.vertices)[:,0])) * 1.1
        mesh2.translate([dist * 1, 0, 0])
        mesh3.translate([dist * 2, 0, 0])

        # Visualize Point Cloud
        vis.add_geometry(mesh)
        vis.add_geometry(mesh2)
        vis.add_geometry(mesh3)
        vis.reset_view_point(True)
        vis.poll_events()
        vis.capture_screen_image(path + "Model_{}.png".format(indx[i]), True)
        vis.clear_geometries()
    vis.destroy_window()
    del model
    return

def test_vertices(path, inputs, targets, model=None, batch_size=32, template_v=None, template_f=None):
    """
    Output the mean error per vertex
    :param path:
    :param inputs:
    :param f:
    :param template:
    :param indx:
    :param targets: 3d point cloud
    :param regions:
    :return:
    """
    os.makedirs(path, exist_ok=True)
    print('test_vertices')

    # vis.run()
    vertices_err = None
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(batch_size)

    for samples_mesh in dataset:
        generated_mesh = model(samples_mesh[0], training=False)

        if vertices_err is None:
            vertices_err = np.sum(tf.keras.losses.mean_squared_error(samples_mesh[1], generated_mesh).numpy(), axis=0)
        else:
            vertices_err += np.sum(tf.keras.losses.mean_squared_error(samples_mesh[1], generated_mesh).numpy(), axis=0)
    vertices_err = vertices_err / inputs.shape[0]

    # Plot the vertices errors on the template
    if template_f and template_v:
        print("Vertices error mean: {:.2e}".format(np.mean(vertices_err)))
        colors = cm.Reds(vertices_err / np.max(vertices_err)) - [0.1, 0.1, 0.1, 0]
        mesh3 = geometry.TriangleMesh()
        mesh3.vertices = utility.Vector3dVector(template_v)
        mesh3.triangles = utility.Vector3iVector(template_f)
        mesh3.vertex_colors = utility.Vector3dVector(colors[:, 0:3])
        mesh3.compute_vertex_normals()
        mesh3.compute_triangle_normals()
        vis = visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh3)
        vc = vis.get_view_control()
        vc.set_zoom(0.55)
        vis.poll_events()
        vis.capture_screen_image(path + "vertices_errors.png", True)
        vis.destroy_window()

    np.savetxt(path + 'vertices_errors.txt', vertices_err, fmt='%1.4e', delimiter='\n')