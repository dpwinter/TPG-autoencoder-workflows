import keras
import keras.backend as K

class Reshaper(keras.models.Model):
    def __init__(self, input_shape, output_shape):
        model_in  = keras.layers.Input(shape=input_shape)
        model_out = K.concatenate((K.variable([-1], dtype='int32'), output_shape))
        shaper = keras.layers.Lambda(lambda x: K.reshape(x, model_out))(model_in)
        super(Reshaper, self).__init__(inputs=model_in, outputs=shaper)

def conv_encoder(input_shape, output_dims, name, filters=8, kernel=(3,3), strides=(2,2)):
    return keras.Sequential([
        keras.layers.Input(shape=input_shape, name=f"{name}_in"),
        keras.layers.Conv2D(filters, kernel, strides, padding='same', 
            activation='relu', name=f"{name}_conv"),
        keras.layers.Flatten(name=f"{name}_flatten"),
        keras.layers.Dense(units=output_dims, activation='linear', name=f"{name}_dense")
        ], name=name)

def conv_decoder(input_dims, output_shape, name, filters=1, kernel=(3,3), strides=(2,2)):
    import math
    in_dims = [math.floor((o-k+2)/s)+1 for o,k,s in zip(output_shape, kernel, strides)]
    return keras.Sequential([
        keras.layers.Input(shape=(input_dims), name=f"{name}_in"),
        keras.layers.Dense(units=in_dims[0]*in_dims[1], activation='relu', name=f"{name}_dense"),
        keras.layers.Reshape(target_shape=(in_dims[0],in_dims[1],1)),
        keras.layers.Conv2DTranspose(filters, kernel, strides, padding='same', 
            activation='relu', name=f"{name}_convT")
        ], name=name)

layer_dims = [256,128,64,32]

def dense_encoder(*args, **kwargs):
    return dense_cascade(*args, **kwargs, dims=layer_dims)

def dense_decoder(*args, **kwargs):
    return dense_cascade(*args, **kwargs, dims=layer_dims[::-1])

def dense_cascade(input_dims, output_dims, name, dims, activation="linear"):
    return keras.Sequential([
        keras.layers.Input(shape=(input_dims), name=f"{name}_in"),
        keras.layers.Dense(units=dims[0], activation='relu', name=f"{name}_dense1"),
        keras.layers.Dense(units=dims[1], activation='relu', name=f"{name}_dense2"),
        keras.layers.Dense(units=dims[2], activation='relu', name=f"{name}_dense3"),
        keras.layers.Dense(units=dims[3], activation='relu', name=f"{name}_dense4"),
        keras.layers.Dense(units=(output_dims), activation=activation, name=f"{name}_out")
        ], name=name)
