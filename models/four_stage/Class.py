import keras
import tensorflow as tf
from models.four_stage.components import *
from collections import OrderedDict

class Class(EncoderBase):

    def __init__(self, latent_dims, stage=None):
        # super(Class, self).__init__(latent_dims)
        EncoderBase.__init__(self, latent_dims, stage)

        self.classifiers = OrderedDict({
            'ST2' : classifier(input_dims=latent_dims[3], output_dims=4) # HOW MANY PARTICLE IDs?
            })
        self.classifiers = self.name_layers_in_model(self.classifiers, "Class")

    def classify(self, z, training=True):
        return self.classifiers["ST2"](z, training=training)

    def call(self, x, training=True):
        z = self.encode(x, training=training)
        y = self.classify(z, training=training)
        return {"Class": y}
