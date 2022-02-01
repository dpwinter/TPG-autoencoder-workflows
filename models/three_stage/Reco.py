from models.three_stage.Encoder import *

class Reco(Encoder):
    def __init__(self, latent_dims, stage):
        Encoder.__init__(self, latent_dims, stage)

        self.recos = [conv_decoder(latent_dims[0], DATA_DIMS[-3:], "Reco_MOD")]
        self.reco_reshapers = [Reshaper(DATA_DIMS[-3:], DATA_DIMS)]

        for i in range(self.stage_idx):
            n_out = latent_dims[i]
            out_0 = int(n_out * MULTIPLICITY[i])
            n_in = latent_dims[i+1]

            self.recos.insert(0, dense_decoder(n_in, out_0, f"Reco_{STAGES[i+1]}"))
            self.reco_reshapers.insert(0, Reshaper([out_0], [n_out]))


    def reconstruct(self, x, training=True):
        for reco, reco_reshaper in zip(self.recos, self.reco_reshapers):
            x = reco_reshaper(reco(x), training=training)
        return x

    def call(self, x, training=True):
        z = self.encode(x, training=training)
        y = self.reconstruct(z ,training=training)
        return {"Reco": y}

    def set_trainable(self, stage_mask):
        for encoder, reco, trainable in zip(self.encoders, self.recos, stage_mask):
            encoder.trainable = int(trainable)
            reco.trainable    = int(trainable)
        self.stage_idx = stage_mask.rindex('1')
        self.compile(loss=self.loss, optimizer=self.optimizer) # recompile
        return [(e.trainable, r.trainable) for e, r in zip(self.encoders, self.recos)]
