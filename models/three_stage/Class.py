from models.three_stage.Encoder import *

class Class(Encoder):

    def __init__(self, latent_dims, stage):
        Encoder.__init__(self, latent_dims, stage)
        assert self.stage_idx == 2  # only for ST2
        self.classifier = dense_decoder(latent_dims[self.stage_idx], 4, f"Class_{STAGES[self.stage_idx]}")

    def classify(self, z, training=True):
        return self.classifier(z)

    def call(self, x, training=True):
        z = self.encode(x, training=training)
        y = self.classify(z, training=training)
        return {"Class": y}
