# autoencoder experiments for triplet encoding
# by Thorben Quast, 09 October 2020
# use python3 on pclcdgpu! not python2!!!

import argparse
import os, shutil
import time
import imageio
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
#qkeras v0.8 installed on pclcdgpu 20 October 2020
from qkeras import QConv2D, QDense
from qkeras.quantizers import quantized_bits, quantized_relu
import numpy as np

weights_quantized = quantized_bits(9, 6)
activation_quantized = quantized_relu(9, 6)

parser = argparse.ArgumentParser()
parser.add_argument("--dataTrain", type=str, help="Path to the training data",
                    default="/home/tquast/L1_trigger_data/converted/electrons_config1_1_to_500GeV_4Tesla.pickle", required=False)
parser.add_argument("--costFigure", type=str, help="Path to the cost function evolution",
                    default="/home/tquast/L1_trigger_data/training/cost.pdf", required=False)
parser.add_argument("--animFile", type=str, help="Animation file",
                    default="/home/tquast/L1_trigger_data/training/ROC_AE.mp4", required=False)
parser.add_argument("--modelFile", type=str, help="Model path",
                    default="/home/tquast/L1_trigger_data/training/model", required=False)
parser.add_argument("--batchSize", type=int,
                    help="Batch size for the training", default=2, required=False)
parser.add_argument("--NEpochs", type=int,
                    help="Number of epochs for the training", default=2000, required=False)
parser.add_argument("--LatentDimension", type=int,
                    help="Dimension of the latent space", default=8, required=False)
parser.add_argument("--GPU", type=int,
                    help="GPU to use", default=3, required=False)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "%i" % args.GPU
batch_size = args.batchSize
num_examples_to_generate = 16

# load the data
with open(args.dataTrain, 'rb') as h:
    data_train = pickle.load(h)
    train_sample = data_train["digi"]
    train_meta_sample = data_train["meta"]


#preprocess data
train_sample = np.expand_dims(train_sample.astype(
    "float32"), axis=len(train_sample.shape))
maxY = train_sample.max()

train_size = train_sample.shape[0]
# shuffle samples
np.random.seed(0)
perm = [x for x in range(0, train_size)]
np.random.shuffle(perm)
train_sample = train_sample[perm]
train_meta_sample = train_meta_sample[perm]

test_sample = train_sample[0:num_examples_to_generate]
test_meta_sample = train_meta_sample[0:num_examples_to_generate]
train_sample = train_sample[num_examples_to_generate:train_size]
train_meta_sample = train_meta_sample[num_examples_to_generate:train_size]
train_size = train_sample.shape[0]
test_size = test_sample.shape[0]
train_dataset = (tf.data.Dataset.from_tensor_slices(train_sample)
                .batch(batch_size))
train_meta_dataset = (tf.data.Dataset.from_tensor_slices(train_meta_sample)
                .batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_sample)
                .batch(batch_size))
test_meta_dataset = (tf.data.Dataset.from_tensor_slices(test_meta_sample)
                .batch(batch_size))

INDEXING = {
    "ROC": {
        "STAGE2": [30, 19, 3, 3],
        "STAGE1": [19, 3, 3],
        "TRIPLET": [3, 3],
        "MODULE": [3],
        "ROC": []
    },
    "MODULE": {
        "STAGE2": [30, 19, 3],
        "STAGE1": [19, 3],
        "TRIPLET": [3],
        "MODULE": []
    },
    "TRIPLET": {
        "STAGE2": [30, 19],
        "STAGE1": [19],
        "TRIPLET": []
    },
    "STAGE1": {
        "STAGE2": [30],
        "STAGE1": []
    },
    "STAGE2": {
        "STAGE2": []
    }
}

# implementation from: https://stackoverflow.com/questions/56974482/how-to-define-a-2d-convolution-on-tensors-with-rank-greater-than-4-in-keras-tens
def reshaper(input_shape, target_shape):
    input = tf.keras.layers.Input(shape=input_shape)
    new_shape = tf.keras.backend.concatenate(
        (tf.keras.backend.variable([-1], dtype='int32'), target_shape))
    # reshaping input into 4D
    reshaped = tf.keras.layers.Lambda(
        lambda x: tf.keras.backend.reshape(x, new_shape))(input)
    return tf.keras.models.Model(inputs=input, outputs=reshaped)

# define the encoder and decoder networks


class HGCL1(tf.keras.Model):
    def __init__(self):
        super(HGCL1, self).__init__()
        self.encoders = {}
        self.decoders = {}
        self.latent_dims = {
            "ROC": args.LatentDimension,
            "MODULE": args.LatentDimension,
            "TRIPLET": args.LatentDimension,
            "STAGE1": args.LatentDimension,
            "STAGE2": args.LatentDimension
        }
        self.level_hierarchy = ["ROC", "MODULE", "TRIPLET", "STAGE1", "STAGE2"]

        self._defineROCEncoder()
        self._defineModuleEncoder()
        self._defineTripletEncoder()
        self._defineStage1Encoder()
        self._defineStage2Encoder()
        self._nameLayers()

        self.dimension_reducers = {}
        self.dimension_expanders = {}
        self.dimension_decoding_reducers = {}
        self.dimension_decoding_expanders = {}
        self._define_reshapers()

        #aux information
        self.current_epoch = 0
        self.min_step = 0        
        self.minsteps_test = []
        self.costs_test = []
        self.minsteps_train = []
        self.costs_train = []

    def _define_reshapers(self):
        for _level in self.level_hierarchy:
            self.dimension_reducers[_level] = {}
            self.dimension_expanders[_level] = {}
            self.dimension_decoding_reducers[_level] = {}
            self.dimension_decoding_expanders[_level] = {}
            for _target_level in INDEXING[_level]:
                _indexes = np.array(
                    INDEXING[_level][_target_level]).astype("int32")
                _input_indexes = np.array(
                    self.encoders[_level].input.shape[1:]).astype("int32")
                _output_indexes = np.array(
                    self.encoders[_level].output.shape[1:]).astype("int32")
                self.dimension_reducers[_level][_target_level] = reshaper(
                    np.append(_indexes, _input_indexes).tolist(
                    ), _input_indexes.tolist()
                )
                self.dimension_expanders[_level][_target_level] = reshaper(
                    _output_indexes.tolist(), np.append(_indexes, _output_indexes).tolist()
                )
                self.dimension_decoding_reducers[_level][_target_level] = reshaper(
                    np.append(_indexes, _output_indexes).tolist(
                    ), _output_indexes.tolist()
                )
                self.dimension_decoding_expanders[_level][_target_level] = reshaper(
                    _input_indexes.tolist(), np.append(_indexes, _input_indexes).tolist()
                )

    def _nameLayers(self):
        _mm = {"ENC": self.encoders, "DEC": self.decoders}
        for _t in ["ENC", "DEC"]:
            for level in self.encoders:
                for layer in _mm[_t][level].layers:
                    _ln = "{:s}_{:s}_{:s}".format(_t, level, layer.name)
                    layer._name = _ln
                #print(_mm[_t][level].summary())
        self.compile()

    def print_trainable(self):
        _mm = {"ENC": self.encoders, "DEC": self.decoders}
        for _t in ["ENC", "DEC"]:
            for level in self.encoders:
                for layer in _mm[_t][level].layers:
                    print(layer.name, layer.trainable)
                print(_mm[_t][level].summary())

    def _defineROCEncoder(self):
        LEVEL = "ROC"
        self.encoders[LEVEL] = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(4, 4, 1)),
                QConv2D(
                    filters=64, kernel_size=(2, 2), strides=(1, 1), activation=activation_quantized, padding='valid', kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized),
                QConv2D(
                    filters=64, kernel_size=(2, 2), strides=(1, 1), activation=activation_quantized, padding='valid', kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized),
                QConv2D(
                    filters=64, kernel_size=(2, 2), strides=(1, 1), activation=activation_quantized, padding='valid', kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized),
                tf.keras.layers.Flatten(),
                QDense(128, activation=activation_quantized, kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized),
                QDense(256, activation=activation_quantized, kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized),
                # No activation
                QDense(self.latent_dims[LEVEL], kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized, activation=weights_quantized)
            ], name="encoder-%s" % LEVEL
        )

        self.decoders[LEVEL] = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    input_shape=(self.latent_dims[LEVEL])),
                tf.keras.layers.Dense(units=1*1*256, activation='relu'),
                tf.keras.layers.Dense(units=1*1*128, activation='relu'),
                tf.keras.layers.Reshape(target_shape=(1, 1, 128)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                    activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=(2, 2), strides=(1, 1), padding='valid', activation="relu"),
                tf.keras.layers.Conv2D(
                    filters=1, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation="relu")
            ], name="decoder-%s" % LEVEL
        )

    def _defineModuleEncoder(self):
        LEVEL = "MODULE"

        self.encoders[LEVEL] = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(3, self.latent_dims["ROC"])),
                tf.keras.layers.Flatten(),
                QDense(32, activation=activation_quantized, kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized),
                QDense(64, activation=activation_quantized, kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized),
                QDense(128, activation=activation_quantized, kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized),
                QDense(256, activation=activation_quantized, kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized),
                QDense(self.latent_dims[LEVEL], kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized, activation=weights_quantized)
            ], name="encoder-%s" % LEVEL
        )

        self.decoders[LEVEL] = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    input_shape=self.encoders[LEVEL].output.shape[1:]),
                tf.keras.layers.Dense(256),
                tf.keras.layers.Dense(128),
                tf.keras.layers.Dense(64),  
                tf.keras.layers.Dense(32),  
                tf.keras.layers.Dense(3*self.latent_dims["ROC"]),
                tf.keras.layers.Reshape(self.encoders[LEVEL].input.shape[1:])
            ], name="decoder-%s" % LEVEL
        )

    def _defineTripletEncoder(self):
        LEVEL = "TRIPLET"

        self.encoders[LEVEL] = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(3, self.latent_dims["MODULE"])),
                tf.keras.layers.Flatten(),
                QDense(32, activation=activation_quantized, kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized),
                QDense(64, activation=activation_quantized, kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized),
                QDense(128, activation=activation_quantized, kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized),
                QDense(256, activation=activation_quantized, kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized),              
                QDense(self.latent_dims[LEVEL], kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized, activation=weights_quantized)
            ], name="encoder-%s" % LEVEL
        )

        self.decoders[LEVEL] = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    input_shape=self.encoders[LEVEL].output.shape[1:]),
                tf.keras.layers.Dense(256),
                tf.keras.layers.Dense(128),
                tf.keras.layers.Dense(64),  
                tf.keras.layers.Dense(32),                     
                tf.keras.layers.Dense(3*self.latent_dims["MODULE"]),
                tf.keras.layers.Reshape(self.encoders[LEVEL].input.shape[1:])
            ], name="decoder-%s" % LEVEL
        )

    def _defineStage1Encoder(self):
        LEVEL = "STAGE1"

        self.encoders[LEVEL] = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(19, self.latent_dims["TRIPLET"])),
                tf.keras.layers.Flatten(),
                QDense(32, activation=activation_quantized, kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized),
                QDense(64, activation=activation_quantized, kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized),
                QDense(128, activation=activation_quantized, kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized),
                QDense(256, activation=activation_quantized, kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized),               
                QDense(self.latent_dims[LEVEL], kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized, activation=weights_quantized)
            ], name="encoder-%s" % LEVEL
        )

        self.decoders[LEVEL] = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    input_shape=self.encoders[LEVEL].output.shape[1:]),
                tf.keras.layers.Dense(256),
                tf.keras.layers.Dense(128),
                tf.keras.layers.Dense(64),  
                tf.keras.layers.Dense(32),                      
                tf.keras.layers.Dense(19*self.latent_dims["TRIPLET"]),
                tf.keras.layers.Reshape(self.encoders[LEVEL].input.shape[1:])
            ], name="decoder-%s" % LEVEL
        )

    def _defineStage2Encoder(self):
        LEVEL = "STAGE2"

        self.encoders[LEVEL] = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(30, self.latent_dims["STAGE1"])),
                tf.keras.layers.Flatten(),
                QDense(32, activation=activation_quantized, kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized),
                QDense(64, activation=activation_quantized, kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized),
                QDense(128, activation=activation_quantized, kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized),
                QDense(256, activation=activation_quantized, kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized),             
                QDense(self.latent_dims[LEVEL], kernel_quantizer=weights_quantized, bias_quantizer=weights_quantized, activation=weights_quantized)
            ], name="encoder-%s" % LEVEL
        )

        self.decoders[LEVEL] = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    input_shape=self.encoders[LEVEL].output.shape[1:]),
                tf.keras.layers.Dense(256),
                tf.keras.layers.Dense(128),
                tf.keras.layers.Dense(64),  
                tf.keras.layers.Dense(32),                    
                tf.keras.layers.Dense(30*self.latent_dims["STAGE1"]),
                tf.keras.layers.Reshape(self.encoders[LEVEL].input.shape[1:])
            ], name="decoder-%s" % LEVEL
        )

    def encode(self, x, level="ROC", data_format="STAGE2", recursive=True):
        if recursive:
            level_index = self.level_hierarchy.index(level)
            if level_index == 0:
                _x = x
            else:
                _x = self.encode(
                    x, level=self.level_hierarchy[level_index-1], data_format=data_format, recursive=recursive)
        else:
            _x = x

        z = self.dimension_expanders[level][data_format](
            self.encoders[level](self.dimension_reducers[level][data_format](_x)))

        return z

    def decode(self, z, level="ROC", data_format="STAGE2", recursive=True):
        x = self.dimension_decoding_expanders[level][data_format](
            self.decoders[level](self.dimension_decoding_reducers[level][data_format](z)))

        if not recursive:
            return x
        else:
            level_index = self.level_hierarchy.index(level)
            if level_index == 0:
                return x
            else:
                return self.decode(x, level=self.level_hierarchy[level_index-1], data_format=data_format, recursive=recursive)

    def set_trainable(self, d):
        if type(d) != dict:
            raise TypeError("d must be a dictionary")
        _mm = {"ENC": self.encoders, "DEC": self.decoders}
        try:
            for _t in d:
                models = _mm[_t]
                for level in d[_t]:
                    _nn = models[level]
                    _trainable = d[_t][level]
                    for layer in _nn.layers:
                        layer.trainable = _trainable
            print()
            print()
            print()
            print("Successfully redefined trainable parameters")
            self.compile()
        except:
            print(d, "setting trainable failed.")
        #self.print_trainable()

    def save_model(self, fpath, auxpath):
        for _level in self.encoders:
            _fpath = fpath + "_" + "encode" + "_" + _level
            if os.path.exists(_fpath):
                shutil.rmtree(_fpath)
            # save the trained model
            self.encoders[_level].save(_fpath)
        for _level in self.decoders:
            _fpath = fpath + "_" + "decode" + "_" + _level
            if os.path.exists(_fpath):
                shutil.rmtree(_fpath)
            # save the trained model
            self.decoders[_level].save(_fpath)    

        with open(auxpath, "wb") as f:
            pickle.dump({
                "costs_train": self.costs_train,
                "minsteps_train": self.minsteps_train, 
                "costs_test": self.costs_test,
                "minsteps_test": self.minsteps_test, 
                "current_epoch": self.current_epoch,
                "min_step": self.min_step
            }, f)

    def load_model(self, fpath, auxpath):
        for _level in self.encoders:
            _fpath = fpath + "_" + "encode" + "_" + _level
            # load the trained model
            self.encoders[_level] = tf.keras.models.load_model(_fpath)
        for _level in self.decoders:
            _fpath = fpath + "_" + "decode" + "_" + _level
            # load the trained model
            self.decoders[_level] = tf.keras.models.load_model(_fpath)

        with open(auxpath, "rb") as f:
            x_loaded = pickle.load(f)
            self.costs_train = x_loaded["costs_train"]
            self.minsteps_train = x_loaded["minsteps_train"]
            self.costs_test = x_loaded["costs_test"]
            self.minsteps_test = x_loaded["minsteps_test"]
            self.current_epoch = x_loaded["current_epoch"]
            self.min_step = x_loaded["min_step"]


@tf.function
def compute_loss(model, x, level="ROC"):
    z = model.encode(x, level=level)
    reconstruction_error = tf.reduce_mean(
        tf.square(tf.subtract(model.decode(z, level=level), x)))
    return reconstruction_error

# define the training step

@tf.function
def train_step(model, x, optimizer, level):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, level)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def evaluate_loss(model, dataset, TRAIN_STAGE):
    loss = tf.keras.metrics.Mean()
    for _x in dataset:
        loss(compute_loss(model, _x, TRAIN_STAGE))
    l2_loss = loss.result() 
    return l2_loss   

figpaths = []
def generate_and_save_images(model, _test_sample, level="ROC"):
    if "STAGE" in level:
        _level = "TRIPLET"
    else: 
        _level = level
    # manually reshape and select only test samples with non-zero entries
    test_sample = _test_sample.reshape((-1, 3, 3, 4, 4, 1))
    positives = test_sample[np.where(test_sample.max(axis=(1, 2, 3, 4, 5)) > 0)]
    zero_entries = test_sample[np.where(
        test_sample.max(axis=(1, 2, 3, 4, 5)) == 0)]

    perm = [x for x in range(positives.shape[0])]
    np.random.seed(0)
    np.random.shuffle(perm)
    positives = positives[perm]
    positives = positives[0:num_examples_to_generate]
    zero_entries = zero_entries[0:4]
    test_sample = np.append(positives, zero_entries, axis=0)

    # add zeros
    _num_examples_to_generate = num_examples_to_generate+4

    z = model.encode(test_sample, level=_level, data_format="TRIPLET")
    predictions = model.decode(z, level=_level, data_format="TRIPLET")

    # reshape for visualisation    
    test_sample = np.transpose(test_sample, [0, 1, 3, 2, 4, 5])
    predictions = tf.transpose(predictions, [0, 1, 3, 2, 4, 5])
    test_sample = test_sample.reshape(-1, 12, 12, 1)
    predictions = tf.reshape(predictions, (-1, 12, 12, 1))

    fig = plt.figure(figsize=(_num_examples_to_generate, 2))
    fig.suptitle("Latent dim.=%i, Epoch: %i/%i, Training stage: %s - Top: Input TC data, Bottom: Decoded TC data" %
                 (args.LatentDimension, model.current_epoch, args.NEpochs, level), fontsize=16)
    # print real image
    for i in range(test_sample.shape[0]):
        plt.subplot(2, _num_examples_to_generate, i + 1)
        plt.imshow(test_sample[i, :, :, 0], cmap='turbo', vmin=0, vmax=maxY)
        plt.axis('off')

    # print generated image
    for i in range(predictions.shape[0]):
        plt.subplot(2, _num_examples_to_generate,
                    i + _num_examples_to_generate + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='turbo', vmin=0, vmax=maxY)
        plt.title("Sample %i" % i, fontsize=5)
        plt.axis('off')

    fig_path = '{:s}_{:04d}.png'.format(
        args.animFile.replace(".mp4", ""), model.current_epoch)
    figpaths.append(fig_path)
    plt.savefig(fig_path)
    plt.close(fig)


# main part
optimizer = tf.keras.optimizers.Adam(1e-4)
model = HGCL1()
try:
    model.load_model(args.modelFile, args.costFigure.replace(".pdf", ".pkl"))
    print("Loaded %s"%args.modelFile)
except:
    print("No model file exists. Model is not loaded.")

TRAIN_STAGE = "ROC"
l2_loss_test = evaluate_loss(model, test_dataset, TRAIN_STAGE)
print
print("Test loss before training is:", l2_loss_test)
print
generate_and_save_images(model, test_sample, level=TRAIN_STAGE)
for epoch in range(model.current_epoch+1, args.NEpochs + 1):
    if epoch == 1:
        TRAIN_STAGE = "STAGE2"
        model.set_trainable({
            "ENC":
                {"ROC": True,
                 "MODULE": True,
                 "TRIPLET": True,
                 "STAGE1": True,
                 "STAGE2": True},
            "DEC":
                {"ROC": True,
                 "MODULE": True,
                 "TRIPLET": True,
                 "STAGE1": True,
                 "STAGE2": True}})
    elif epoch == 2:
        TRAIN_STAGE = "STAGE1"
        model.set_trainable({
            "ENC":
                {"ROC": True,
                 "MODULE": True,
                 "TRIPLET": True,
                 "STAGE1": True,
                 "STAGE2": False},
            "DEC":
                {"ROC": True,
                 "MODULE": True,
                 "TRIPLET": True,
                 "STAGE1": True,
                 "STAGE2": False}})   
    elif epoch == 3:
        TRAIN_STAGE = "TRIPLET"
        model.set_trainable({
            "ENC":
                {"ROC": True,
                 "MODULE": True,
                 "TRIPLET": True,
                 "STAGE1": False,
                 "STAGE2": False},
            "DEC":
                {"ROC": True,
                 "MODULE": True,
                 "TRIPLET": True,
                 "STAGE1": False,
                 "STAGE2": False}})  
    elif epoch == 4:
        TRAIN_STAGE = "MODULE"
        model.set_trainable({
            "ENC":
                {"ROC": True,
                 "MODULE": True,
                 "TRIPLET": False,
                 "STAGE1": False,
                 "STAGE2": False},
            "DEC":
                {"ROC": True,
                 "MODULE": True,
                 "TRIPLET": False,
                 "STAGE1": False,
                 "STAGE2": False}})                                               
    elif epoch == 5:
        TRAIN_STAGE = "ROC"
        model.set_trainable({
            "ENC":
                {"ROC": True,
                 "MODULE": False,
                 "TRIPLET": False,
                 "STAGE1": False,
                 "STAGE2": False},
            "DEC":
                {"ROC": True,
                 "MODULE": False,
                 "TRIPLET": False,
                 "STAGE1": False,
                 "STAGE2": False}})
    elif epoch == 301:
        TRAIN_STAGE = "MODULE"
        model.set_trainable({
            "ENC":
                {"ROC": False,
                 "MODULE": True,
                 "TRIPLET": False,
                 "STAGE1": False,
                 "STAGE2": False},
            "DEC":
                {"ROC": False,
                 "MODULE": True,
                 "TRIPLET": False,
                 "STAGE1": False,
                 "STAGE2": False}})
    elif epoch == 501:
        TRAIN_STAGE = "TRIPLET"
        model.set_trainable({
            "ENC":
                {"ROC": False,
                 "MODULE": False,
                 "TRIPLET": True,
                 "STAGE1": False,
                 "STAGE2": False},
            "DEC":
                {"ROC": False,
                 "MODULE": False,
                 "TRIPLET": True,
                 "STAGE1": False,
                 "STAGE2": False}})
    elif epoch == 901:
        TRAIN_STAGE = "STAGE1"
        model.set_trainable({
            "ENC":
                {"ROC": False,
                 "MODULE": False,
                 "TRIPLET": False,
                 "STAGE1": True,
                 "STAGE2": False},
            "DEC":
                {"ROC": False,
                 "MODULE": False,
                 "TRIPLET": False,
                 "STAGE1": True,
                 "STAGE2": False}})
    elif epoch == 1401:
        TRAIN_STAGE = "STAGE2"
        model.set_trainable({
            "ENC":
                {"ROC": False,
                 "MODULE": False,
                 "TRIPLET": False,
                 "STAGE1": False,
                 "STAGE2": True},
            "DEC":
                {"ROC": False,
                 "MODULE": False,
                 "TRIPLET": False,
                 "STAGE1": False,
                 "STAGE2": True}})

    start_time = time.time()
    for train_x in train_dataset:
        train_step(model, train_x, optimizer, TRAIN_STAGE)
        model.min_step += 1
        l2_loss_train = compute_loss(model, train_x, TRAIN_STAGE)
        model.costs_train.append(l2_loss_train)
        model.minsteps_train.append(model.min_step)
    end_time = time.time()


    l2_loss_test = evaluate_loss(model, test_dataset, TRAIN_STAGE)
    model.costs_test.append(l2_loss_test)
    model.minsteps_test.append(model.min_step)
    print('Phase: {}, Epoch: {}, Train set L2: {}, Test set L2: {}, time elapse for current epoch: {}s'
          .format(TRAIN_STAGE, epoch, l2_loss_train, l2_loss_test, round(end_time - start_time, 1)))
    generate_and_save_images(model, test_sample, level=TRAIN_STAGE)

    # print cost function
    fig = plt.figure(figsize=(16, 9))
    plt.plot(model.minsteps_train, model.costs_train, label="train")
    plt.plot(model.minsteps_test, model.costs_test, label="test")
    plt.xlabel("Optimisation steps", fontsize=16)
    plt.ylabel("L2 loss (test sample) [a.u.]", fontsize=16)
    plt.title("Cost function evolution: Epoch %i/%i" %
              (epoch, args.NEpochs), fontsize=16)
    plt.ylim([0, 4.])
    plt.legend()
    plt.savefig(args.costFigure)
    plt.close(fig)

    #save the model after each epoch: 
    model.current_epoch = epoch
    model.save_model(args.modelFile, args.costFigure.replace(".pdf", ".pkl"))