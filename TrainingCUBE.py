import tensorflow as tf
import numpy as np
import os
import hdf5storage
import matplotlib.pyplot as plt

from Model import decoderSpectrum
from utils import plot_loss, plot_mesh
from utils_test import test_log, test_vertices, test_NN
from dataset import datasetCreator

# Setting seed for deterministic experimnts
tf.random.set_seed(1234)
# tf.config.set_visible_devices([], 'GPU') # disable gpu

# Network Parameters
params = {'n_autoval': 15 + 1,
          'dataset': 'CUBE',  #
          'dec_units': [258, 1024, 2048],
          'dropout_prob_dec': [0.1],
          'b_norm': True,
          'activation': 'selu',
          'lr': 2e-3,
          'decay_step': 1000,
          'decay_rate': 0.9,
          'batch_size': 64,
          'EPOCHS': 2000,
          'identity': True,
          'local_operator': "patch",
          'regions_name': ("r",),
          'n_autoval_local': (15 + 1,),
          'diff_eigs': True,
          'load_model' : None,
          }

BUFFER_SIZE = 6000
path = params['dataset'] + '/PAT15+15/'

import json

os.makedirs(path + 'checkpoint', exist_ok=True)
with open(path + 'params.json', 'w') as f:
    json.dump(params, f)

# Load the model
if params['load_model'] is not None:
    model = tf.keras.models.load_model(params['load_model']+'/train_model')
    p = params.copy()
    with open(params['load_model'] + '/params.json', 'w') as f:
        json.dump(params, f)
    params['lr'] = p['lr']
    params['decay_step'] = p['decay_step']
    params['decay_rate'] = p['decay_rate']
    params['batch_size'] = p['batch_size']
    params['EPOCHS'] = p['EPOCHS']
    params['load_model'] = p['load_model']
    del p
    # model.load_weights(path + '/checkpoint/ck')

dataset = datasetCreator(params['dataset'])
meshes, f = dataset.load_meshes()
if len(params['n_autoval_local']) > 0:
    eigs, regions_ind = dataset.load_globalLocalInput(n_autoval_global=params['n_autoval'],
                                                      n_autoval_local=params['n_autoval_local'],
                                                      local_operator=params["local_operator"],
                                                      local_regions=params['regions_name'],
                                                      diff_eigs=params['diff_eigs'],
                                                      bc=False)
else:
    eigs = dataset.load_globalEig(n_autoval=params['n_autoval'])
    regions_ind = [dataset.get_region_indexes()]
    if params['diff_eigs']:
        eigs = np.diff(eigs)
train_index, evals_index = dataset.split_validation()

evals_samples = meshes[evals_index, :, :]
train_samples = meshes[train_index, :, :]
evals_eigs = eigs[evals_index, :]
train_eigs = eigs[train_index, :]

if params['load_model'] is None:
    # load bias
    model = decoderSpectrum(n_autoval=evals_eigs.shape[1], output_vertices=evals_samples.shape[1],
                            dec_units=params['dec_units'], dropout_prob_dec=params['dropout_prob_dec'],
                            b_norm=params['b_norm'], activation=params['activation'])

# Defining the optimizer for the training
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    params['lr'],
    decay_steps=params['decay_step'],
    decay_rate=params['decay_rate'],
    staircase=True)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Shuffling the training data and creating the batch
train_dataset = tf.data.Dataset.from_tensor_slices((train_eigs, train_samples)).shuffle(BUFFER_SIZE).batch(
    np.int64(params['batch_size']))
evals_dataset = tf.data.Dataset.from_tensor_slices((evals_eigs, evals_samples)).batch(np.int64(params['batch_size']))

test_error = list()
train_error = list()
i_modelTest = 0
fig, ax = plt.subplots()
lineTrain, = ax.plot(train_error, label="Train")
lineTest, = ax.plot(test_error, label="Test", color='red')
lineSaved = ax.axvline(i_modelTest, color='green', linestyle='--', label="Epoch saved {}".format(i_modelTest))
ax.set_yscale("log")
ax.legend()
fig.show()

def update_plot():
    global lineTrain, lineTest
    lineTrain.remove()
    lineTest.remove()
    lineTrain, = ax.plot(train_error, label="Train", color="blue")
    lineTest, = ax.plot(test_error, label="Test", color='red')
    ax.get_legend().remove()
    ax.legend()
    ax.grid(True, color='0.9')
    fig.canvas.draw()
    fig.canvas.flush_events()
    fig.savefig(path + "loss_int.png")


min_error = None
# For each epoch
for epoch in range(params['EPOCHS']):
    print("EPOCH " + str(epoch))
    # For each batch
    batch_i = 0
    err = gerr = 0
    for samples_mesh in train_dataset:
        batch_i = batch_i + 1
        with tf.GradientTape() as gen_tape:
            # Forward pass
            generated_mesh = model(samples_mesh[0], training=True)  # X generated_mesh = decoder(evals,training=True)
            # loss computation
            gen_loss = tf.keras.losses.mean_squared_error(samples_mesh[1], generated_mesh)
            loss = tf.reduce_mean(gen_loss, axis=-1)

        gradients_of_generator = gen_tape.gradient(loss, model.trainable_variables)
        tmp = generator_optimizer.apply_gradients(zip(gradients_of_generator, model.trainable_variables))
        err += np.sum(loss)
    err /= train_samples.shape[0]
    train_error.append(err)
    print("Training Error: {:.4e}".format(err))

    err = gerr = 0
    for samples_mesh in evals_dataset:
        generated_mesh = model(samples_mesh[0], training=False)
        gen_loss = tf.keras.losses.mean_squared_error(samples_mesh[1], generated_mesh)
        loss = tf.reduce_mean(gen_loss, axis=-1)
        gerr += np.sum(loss)
    gerr /= evals_samples.shape[0]
    print("Test Error: {:.4e}".format(gerr))
    test_error.append(gerr)
    if (min_error is None) or (min_error > test_error[-1] + 0.3 * train_error[-1]):
        min_error = test_error[-1] + 0.3 * train_error[-1]
        i_modelTest = epoch
        model.save_weights(path + 'checkpoint/ck')
        lineSaved.remove()
        lineSaved = ax.axvline(i_modelTest, color='green', linestyle='--',
                               label="Epoch saved {}".format(i_modelTest))
    if epoch % 20 == 0:
        update_plot()

os.makedirs(path + 'train_model', exist_ok=True)
os.makedirs(path + 'test_model', exist_ok=True)
# Losses
with open(path + 'Log.txt', 'w') as fh:
    for terr, trainerr in zip(test_error, train_error):
        fh.write("{};{}\n".format(trainerr, terr))
plt.close(fig)
plot_loss(path + "", np.asarray(train_error), test_error)

# Report
with open(path + 'report.txt', 'w') as fh:
    fh.write("PARAMS\n")
    for k in params:
        fh.write("{}:{}\n".format(k, params[k]))
    fh.write("\n")
    fh.write("SAVED MODEL \n")
    fh.write("Epoch:{}\n".format(i_modelTest))
    fh.write("Error:{}\n".format(min_error))
    fh.write("\n")
    fh.write("DECODER\n")
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
tf.keras.utils.plot_model(model, to_file=path + 'model.png', show_shapes=True)

l= model.layers[-2]
bias = l.weights[-1]
bias = np.reshape( bias.numpy(), (-1,3))
plot_mesh(bias,f,save=True,name=path + 'template_train')
hdf5storage.savemat(path+"template_train.mat", {'v': bias, 'f': f}, oned_as='column',
                    store_python_metadata=True)
model.save(path + 'train_model')

model.load_weights(path + 'checkpoint/ck')
l= model.layers[-2]
bias = l.weights[-1]
bias = np.reshape( bias.numpy(), (-1,3))
plot_mesh(bias,f,save=True,name=path + 'template_test')
hdf5storage.savemat(path+"template_test.mat", {'v': bias, 'f': f}, oned_as='column',
                    store_python_metadata=True)
model.save(path + 'test_model')

indx = np.linspace(0, train_samples.shape[0], 20, endpoint=False, dtype=int)
test_log(path + "train_imm_{}/".format(params['dataset']), train_eigs, f,
         model=model, indx = indx,
         targets=train_samples,regions=(regions_ind[0], np.logical_not(regions_ind[0])),
         rotation_xyz=(0, -np.pi / 2, 0))
indx = np.linspace(0, evals_samples.shape[0], 20, endpoint=False, dtype=int)
test_log(path + "test_imm_{}/".format(params['dataset']), evals_eigs, f,
         model=model, indx=indx,
         targets=evals_samples, regions=(regions_ind[0], np.logical_not(regions_ind[0])),
         rotation_xyz=(0, -np.pi / 2, 0))
test_NN(path + "TestNN/" , evals_eigs, train_eigs, evals_samples,
                    train_samples, f, model=model,
                    indx=indx, batch_size=params['batch_size'], rotation_xyz=(0, -np.pi / 2, 0))

test_vertices(path + "test_imm_{}/".format(params['dataset']), evals_eigs, evals_samples,  model=model,
              batch_size=params['batch_size'])
test_vertices(path + "train_imm_{}/".format(params['dataset']), train_eigs, train_samples,
              model=model, batch_size=params['batch_size'])