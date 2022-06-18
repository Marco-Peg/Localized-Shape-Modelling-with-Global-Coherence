import tensorflow as tf

def decoderSpectrum(n_autoval=30,output_vertices=6890,
                    dec_units=[128, 258, 512], dropout_prob_dec=None,
                    b_norm=True, activation='selu'):
    # DECODER
    n_layers = len(dec_units)
    if n_layers < 1:
        raise ValueError('More than 1 layers are expected.')
    input = tf.keras.Input(shape=(n_autoval,), name="Autovals", dtype=tf.float32)
    for i in range(n_layers):
        name = 'dec_' + str(i)
        if i == 0:
            x = input
        x = tf.keras.layers.Dense(dec_units[i], name=name,
                                  kernel_regularizer=tf.keras.regularizers.l1_l2(),
                                  bias_regularizer=tf.keras.regularizers.l1_l2())(x)
        if b_norm:
            x = tf.keras.layers.BatchNormalization(name='BN' + str(i))(x)
        x = tf.keras.layers.Activation(activation, name=activation + str(i))(x)
        j=i if len(dropout_prob_dec) > i else -1
        if dropout_prob_dec is not None and dropout_prob_dec[j] > 0 and dropout_prob_dec[j]<1:
            x = tf.keras.layers.Dropout(dropout_prob_dec[j],
                                        name='Drop' + str(i))(x)
    x = tf.keras.layers.Dense(output_vertices * 3, activation='linear', name='output', use_bias=True,
                                  kernel_regularizer=tf.keras.regularizers.l1_l2(),
                                  bias_regularizer=tf.keras.regularizers.l1_l2())(x)  # output layer
    out_dec = tf.keras.layers.Reshape((output_vertices, 3), name='res_output')(x)  # reshape to be a matrix of nv x 3

    decoder = tf.keras.Model(inputs=input, outputs=out_dec, name="Decoder")
    return decoder
