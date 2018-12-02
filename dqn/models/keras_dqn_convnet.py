import numpy as np
import keras

def build_network_keras(input_dims, n_actions):

    nel = np.zeros(input_dims).size
    input_shape = input_dims
    model = keras.Sequential([
            keras.layers.Reshape(input_shape, input_shape=input_shape),
            keras.layers.Conv2D(32, (8, 8), strides=(4,4),
                                padding='same',
                                activation='relu'),
            keras.layers.Conv2D(64, (4, 4), strides=(2,2),
                                padding='same',
                                activation='relu'),
            keras.layers.Conv2D(64, (3, 3), strides=(1,1),
                                padding='same',
                                activation='relu')
    ])

    out_shape = model.predict(np.zeros((1,) + input_shape)).shape
    nel = np.prod(out_shape)
    # Adding the final 3 layers post-run does nothing
    model = keras.Sequential([
        keras.layers.Reshape(input_shape, input_shape=input_shape),
        keras.layers.Conv2D(32, (8, 8), strides=(4,4),
                            padding='same',
                            activation='relu'),
        keras.layers.Conv2D(64, (4, 4), strides=(2,2),
                            padding='same',
                            activation='relu'),
        keras.layers.Conv2D(64, (3, 3), strides=(1,1),
                            padding='same',
                            activation='relu'),
        keras.layers.Reshape((nel,)),
        keras.layers.Dense(512),
        keras.layers.Dense(n_actions)
    ])

    model.compile(
        loss='mse',
        optimizer=keras.optimizers.Adam(lr=0.001)
    )

    return model