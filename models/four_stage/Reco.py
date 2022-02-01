from models.four_stage.components import *
from collections import OrderedDict

class Reco(EncoderBase):
    def __init__(self, latent_dims, stage):
        # super(Reco, self).__init__(latent_dims=latent_dims)
        EncoderBase.__init__(self, latent_dims=latent_dims, stage=stage)

        self.decoder_reshapers = OrderedDict({
            'ST2' : Reshaper([30*latent_dims[2]], [latent_dims[2]]),
            'ST1' : Reshaper([3*19*latent_dims[1]], [latent_dims[1]]),
            'MOD' : Reshaper([3 *latent_dims[0]], [latent_dims[0]]),
            'ROC' : Reshaper([4,4,1], [30,19,3,3, 4,4,1])
            })
        self.decoders = OrderedDict({
            'ST2' : decoder(input_dims=latent_dims[3], output_dims=latent_dims[2], n=multiplicity[2]),
            'ST1' : decoder(input_dims=latent_dims[2], output_dims=latent_dims[1], n=multiplicity[1]),
            'MOD' : decoder(input_dims=latent_dims[1], output_dims=latent_dims[0], n=multiplicity[0]),
            'ROC' : conv_decoder(input_dims=latent_dims[0])
            })

        self.decoders = self.name_layers_in_model(self.decoders, "dec")
        self.decoder_reshapers = self.name_layers_in_model(self.decoder_reshapers, "dec_resh")

        # remove all unused stages higher up the stack
        for k in self.decoders.keys():
            if k == stage: break
            else:
                self.decoder_reshapers[k] = None
                self.decoders[k] = None


    def decode(self, x, training=True):
        for level, decoder in list(self.decoders.items())[-self.depth:]:
            x = self.decoder_reshapers[level](decoder(x), training=training)
        return x

    def call(self, x, training=True):
        z = self.encode(x, training=training)
        y = self.decode(z ,training=training)
        return {"Reco": y}

    def set_trainable(self, trainable):
        trainables = super().set_trainable(trainable)
        trainable = trainable[0] + trainable # expand first stage to ROC+MOD
        for dec, v in zip(self.decoders.values(), trainable):
            if dec: dec.trainable = int(v)
        return trainables
