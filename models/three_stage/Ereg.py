from models.three_stage.Encoder import *

EREG_OUTPUT_SHAPES = [[30,19,3], [30,1], [1]]

class Ereg(Encoder):

    def __init__(self, latent_dims, stage):
        Encoder.__init__(self, latent_dims, stage)
        self.e_regressor_reshaper = Reshaper([1], EREG_OUTPUT_SHAPES[self.stage_idx])
        self.e_regressor = dense_decoder(latent_dims[self.stage_idx], 1, f"Ereg_{STAGES[self.stage_idx]}")

    def e_regress(self, z, training=True):
        return self.e_regressor_reshaper(self.e_regressor(z), training=training)

    def call(self, x, training=True):
        z = self.encode(x, training=training)
        y = self.e_regress(z, training=training)
        return {"Ereg": y}
