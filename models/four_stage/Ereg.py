import keras
import tensorflow as tf
from models.four_stage.components import *
from collections import OrderedDict

class Ereg(EncoderBase):
    def __init__(self, latent_dims, stage):
        EncoderBase.__init__(self, latent_dims=latent_dims, stage=stage)

        self.e_regressor_reshapers = OrderedDict({
            'ROC' : None,
            'MOD' : Reshaper([1], [30,19,3]),
            'ST1' : Reshaper([1], [30,1]),
            'ST2' : Reshaper([1], [1])
            })
        self.e_regressors = OrderedDict({
            'ROC' : None,
            'MOD' : regressor(input_dims=latent_dims[1], output_dims=1),
            'ST1' : regressor(input_dims=latent_dims[2], output_dims=1),
            'ST2' : regressor(input_dims=latent_dims[3], output_dims=1)
            })
        self.e_regressors = self.name_layers_in_model(self.e_regressors, "Ereg")
        self.e_regressor_reshapers = self.name_layers_in_model(self.e_regressor_reshapers, "Ereg_resh")
        # only select requested stage.
        for k in self.e_regressors.keys():
            if k != stage and k != "ROC":
                self.e_regressor_reshapers[k] = None
                self.e_regressors[k] = None

    def e_regress(self, z, training=True):
        level, regressor = list(self.e_regressors.items())[self.depth-1]
        return self.e_regressor_reshapers[level](regressor(z), training=training)

    def call(self, x, training=True):
        z = self.encode(x, training=training)
        y = self.e_regress(z, training=training)
        return {"Ereg": y}
