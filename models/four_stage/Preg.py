import keras
import tensorflow as tf
from models.four_stage.components import *
from collections import OrderedDict

class Preg(EncoderBase):

    def __init__(self, latent_dims, stage=None):
        # super(Preg, self).__init__(latent_dims)
        EncoderBase.__init__(self, latent_dims, stage="ST2")

        self.p_regressors = OrderedDict({
            'ST2' : regressor(input_dims=latent_dims[3], output_dims=2)
            })
        self.p_regressors = self.name_layers_in_model(self.p_regressors, "Preg")

    def p_regress(self, z, training=True):
        return self.p_regressors["ST2"](z, training=training)

    def call(self, x, training=True):
        z = self.encode(x, training=training)
        y = self.p_regress(z, training=training)
        return {"Preg": y}
