from models.three_stage.Encoder import *

class Preg(Encoder):

    def __init__(self, latent_dims, stage):
        Encoder.__init__(self, latent_dims, stage)
        assert self.stage_idx == 2  # Preg only for ST2.
        self.p_regressor = dense_decoder(latent_dims[self.stage_idx], 2, f"Preg_{STAGES[self.stage_idx]}")

    def p_regress(self, z, training=True):
        return self.p_regressor(z)

    def call(self, x, training=True):
        z = self.encode(x, training=training)
        y = self.p_regress(z, training=training)
        return {"Preg": y}
