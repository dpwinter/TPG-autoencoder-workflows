from models.four_stage.Reco import *
from models.four_stage.Ereg import *
from models.four_stage.Preg import *
from models.four_stage.Class import *

class RecoEregPregClass(Reco, Ereg, Preg, Class):

    def __init__(self, latent_dims, stage):
        # super(RecoEregPregClass, self).__init__(latent_dims, stage)
        Reco.__init__(self, latent_dims=latent_dims, stage=stage)
        Ereg.__init__(self, latent_dims=latent_dims, stage=stage)
        Preg.__init__(self, latent_dims=latent_dims, stage=stage)
        Class.__init__(self, latent_dims=latent_dims, stage=stage)

    def call(self, x, training=True):
        z = self.encode(x, training=training)
        y_reco = self.decode(z ,training=training)
        y_ereg = self.e_regress(z, training=training)
        y_preg = self.p_regress(z, training=training)
        y_class = self.classify(z, training=training)
        return {'Reco': y_reco, 'Ereg': y_ereg, 'Preg': y_preg, 'Class': y_class}
