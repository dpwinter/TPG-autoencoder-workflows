import keras
from collections import OrderedDict
from common.models import *

STAGES = ["MOD", "ST1", "ST2"]
MULTIPLICITY = [3*19, 30, 1] # MOD/ST1, ST1/ST2, ST2/ST2
DATA_DIMS = [30, 19, 3, 8, 8, 1]

class Encoder(keras.models.Model):

    def __init__(self, latent_dims, stage):
        super().__init__()
        self.stage_idx = STAGES.index(stage)

        self.encoders = [conv_encoder(DATA_DIMS[-3:], latent_dims[0], "Enc_MOD")]
        self.encoder_reshapers = [Reshaper(DATA_DIMS, DATA_DIMS[-3:])]
        for i in range(self.stage_idx):
            in_0  = latent_dims[i]
            n_in  = int(in_0 * MULTIPLICITY[i])
            n_out = latent_dims[i+1]

            self.encoder_reshapers.append(Reshaper([in_0], [n_in]))
            self.encoders.append(dense_encoder(n_in, n_out, f"Enc_{STAGES[i+1]}"))

    def encode(self, x, training=True):
        for encoder, reshaper in zip(self.encoders, self.encoder_reshapers):
            x = encoder(reshaper(x), training=training)
        return x

    def call(self, x, training=True):
        z = self.encode(x, training=training)
        return z

    def set_trainable(self, stage_mask):
        for encoder, trainable in zip(self.encoders, stage_mask):
            encoder.trainable = int(trainable)
        self.stage_idx = stage_mask.rindex('1')
        self.compile(loss=self.loss, optimizer=self.optimizer) # recompile
        return [e.trainable for e in self.encoders]
