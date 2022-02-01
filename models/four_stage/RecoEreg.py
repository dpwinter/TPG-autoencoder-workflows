import keras
import tensorflow as tf
from components import *
from models.four_stage.Reco import Reco
from models.four_stage.Ereg import Ereg
from collections import OrderedDict

class RecoEreg(Reco, Ereg):
	def __init__(self, latent_dims, stage):
		# super(RecoEreg, self).__init__(latent_dims=latent_dims, stage=stage)
		Reco.__init__(self, latent_dims=latent_dims, stage=stage)
		Ereg.__init__(self, latent_dims=latent_dims, stage=stage)
	
	def call(self, x, training=True):
		z = self.encode(x, training=training)
		y_reco = self.decode(z ,training=training)
		y_ereg = self.e_regress(z, training=training)
		return {'Reco': y_reco, 'Ereg': y_ereg}
