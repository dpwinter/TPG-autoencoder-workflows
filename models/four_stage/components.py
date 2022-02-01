import keras
import keras.backend as K
import tensorflow as tf
from collections import OrderedDict

multiplicity = [3,3*19,30]  # rocs/module, modules/triplet, triplets/layer, layers

class EncoderBase(keras.models.Model):

    def __init__(self, latent_dims, stage):
        super(EncoderBase, self).__init__()
        # self.depth = 4  # depth of autoencoding
        self.encoder_reshapers = OrderedDict({
            'ROC' : Reshaper([30,19,3,3, 4,4,1], [4,4,1]),
            'MOD' : Reshaper([latent_dims[0]], [3 *latent_dims[0]]),
            'ST1' : Reshaper([latent_dims[1]], [3*19*latent_dims[1]]),
            'ST2' : Reshaper([latent_dims[2]], [30*latent_dims[2]])
            })
        self.encoders = OrderedDict({
            'ROC' : conv_encoder(output_dims=latent_dims[0]),
            'MOD' : encoder(input_dims=latent_dims[0], output_dims=latent_dims[1], n=multiplicity[0]),
            'ST1' : encoder(input_dims=latent_dims[1], output_dims=latent_dims[2], n=multiplicity[1]),
            'ST2' : encoder(input_dims=latent_dims[2], output_dims=latent_dims[3], n=multiplicity[2])
            })
        self.encoders = self.name_layers_in_model(self.encoders, "enc")
        self.encoder_reshapers = self.name_layers_in_model(self.encoder_reshapers, "enc_resh")
        self.depth = list(self.encoders.keys()).index(stage) + 1

    def encode(self, x, training=True):
        for level, encoder in list(self.encoders.items())[:self.depth]:
            x = encoder(self.encoder_reshapers[level](x), training=training)
        return x

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x)
            loss = self.compiled_loss(y, y_pred)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def call(self, x, training=True):
        z = self.encode(x, training=training)
        return z

    def model(self):  # helper function to print the model
        x = keras.layers.Input(shape=(30,19,3,3,4,4,1))
        return keras.models.Model(inputs=[x], outputs=self.call(x))

    def name_layers_in_model(self, model_dict, model_name):
        for name, submodel in model_dict.items():
            if not submodel: continue
            submodel._name = "{}_{}".format(model_name, name)
            for i, layer in enumerate(submodel.layers):
                layer._name = "{}_{}_{}".format(model_name, name, i)
        return model_dict

    def set_trainable(self, trainable):
        trainable = trainable[0] + trainable # expand first stage to ROC+MOD
        for enc, v in zip(self.encoders.values(), trainable):
            enc.trainable = int(v)
        depth = [idx+1 for idx, item in enumerate(trainable) if int(item) != 0]
        self.depth = depth[-1] if depth else 4  # all layers untrainable
        self.compile(loss=self.loss, optimizer=self.optimizer)  # recompile model AFTER setting trainable attributes.
        return {k : enc.trainable for k,enc in self.encoders.items()}

class Reshaper(keras.models.Model):
    def __init__(self, input_shape, output_shape):
        model_in  = keras.layers.Input(shape=input_shape)
        model_out = K.concatenate((K.variable([-1], dtype='int32'), output_shape))
        shaper = keras.layers.Lambda(lambda x: K.reshape(x, model_out))(model_in)
        super(Reshaper, self).__init__(inputs=model_in, outputs=shaper)

def conv_encoder(output_dims):
    return keras.Sequential([
        keras.layers.Input(shape=(4,4,1)),
        keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=(1,1), padding='valid', activation='relu'),
        keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=(1,1), padding='valid', activation='relu'),
        keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=(1,1), padding='valid', activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dense(units=256, activation='relu'),
        keras.layers.Dense(units=output_dims, activation='linear')
        ])

def conv_decoder(input_dims):
    return keras.Sequential([
        keras.layers.Input(shape=(input_dims)),
        keras.layers.Dense(units=256, activation='relu'),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Reshape(target_shape=(1,1,128)),
        keras.layers.Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'),
        keras.layers.Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu'),
        keras.layers.Conv2D(filters=64, kernel_size=(2,2), strides=(1,1), padding='valid', activation='relu'),
        keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), padding='valid', activation='relu')
        ])

def encoder(input_dims, output_dims, n):
    return keras.Sequential([
        keras.layers.Input(shape=(n*input_dims)),
        keras.layers.Dense(units=32, activation='relu'),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dense(units=256, activation='relu'),
        keras.layers.Dense(output_dims, activation='linear')
        ])

def decoder(input_dims, output_dims, n):
    return keras.Sequential([
        keras.layers.Input(shape=input_dims),
        keras.layers.Dense(units=256, activation='relu'),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dense(units=32, activation='relu'),
        keras.layers.Dense(units=n*output_dims, activation='linear')
        ])

def regressor(input_dims, output_dims):
    return keras.Sequential([
        keras.layers.Input(shape=input_dims),
        keras.layers.Dense(units=256, activation='relu'),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dense(units=32, activation='relu'),
        keras.layers.Dense(units=output_dims, activation='linear'),
        ])

def classifier(input_dims, output_dims):
    return keras.Sequential([
        keras.layers.Input(shape=input_dims),
        keras.layers.Dense(units=256, activation='relu'),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dense(units=32, activation='relu'),
        keras.layers.Dense(units=output_dims, activation='softmax'),
        ])
