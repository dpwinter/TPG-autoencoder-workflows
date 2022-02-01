from models.three_stage.Reco import Reco
from models.three_stage.Ereg import Ereg
from models.three_stage.Preg import Preg
from models.three_stage.Class import Class

class RecoEregPregClass(Reco, Ereg, Preg, Class):

    def __init__(self, latent_dims, stage):
        Reco.__init__(self, latent_dims, stage)
        Ereg.__init__(self, latent_dims, stage)
        Preg.__init__(self, latent_dims, stage)
        Class.__init__(self, latent_dims, stage)

    def call(self, x, training=True):
        z = self.encode(x, training=training)
        y_reco = self.reconstruct(z ,training=training)
        y_ereg = self.e_regress(z, training=training)
        y_preg = self.p_regress(z, training=training)
        y_class = self.classify(z, training=training)
        return {'Reco': y_reco, 'Ereg': y_ereg, 'Preg': y_preg, 'Class': y_class}
