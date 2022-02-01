import luigi
from luigi.util import inherits
import tensorflow as tf
import keras
import os, sys

from common import loaders as loader
from common.utility import fs, iterdo
from common.tasks import ModelTask

class TrainParams(luigi.Config):
    """Parameters for TrainTasks. Inherited to TrainModel with DataParams and ModelParams."""

    relPreloadPath = luigi.Parameter()
    nameSuffix     = luigi.Parameter()
    lossFunctions  = luigi.ListParameter()
    lossWeights    = luigi.ListParameter()
    nEpochs        = luigi.IntParameter()
    batchSize      = luigi.IntParameter()
    optimizer      = luigi.Parameter()
    learningRate   = luigi.FloatParameter()
    cbFns          = luigi.ListParameter()


@inherits(TrainParams)
class TrainModel(ModelTask):
    """Task for model training. Inherits DataParams and ModelParams from ModelTask."""

    def __init__(self, *args, **kwargs):
        """Check meta.yaml and create dir structure"""

        # load meta yaml
        import yaml
        super().__init__(*args, **kwargs)
        meta_file = fs.touch(os.path.join(self.model_cls_dir, 'meta.yaml'))
        with open(meta_file, 'r') as fd:
            meta = yaml.full_load(fd)
            meta = meta if meta else {}

        # lookup training in meta yaml
        params = self.__dict__['param_kwargs']
        key = iterdo.val_in_dict(params, meta)

        if key:
            print("Continued training (Training found in meta.yaml).", file=sys.stderr)
            meta[key]['nEpochs'] += self.nEpochs
        else:
            print("New training.", file=sys.stderr) # generate new key for meta yaml
            key = self.gen_key(len(meta), self.stageMask.rindex('1'))
            meta[key] = params

        self.name = self.name_from_key(key)
        print(f"Training model {self.name}", file=sys.stderr)
        self.model_dir = fs.mkdir(os.path.join(self.model_cls_dir, self.name))

        print("Write to meta.yaml.", file=sys.stderr)
        with open(meta_file, 'w') as fd:
            yaml.dump(meta, fd)

    def complete(self):
        """If weights exist, load those (continued training)."""

        if super().complete():
            print("Changing relPreloadPath.", file=sys.stderr)
            self.relPreloadPath = self.name
        return False

    def gen_key(self, n, idx):
        """Generate new key for this training."""

        char = chr(65 + n) # 'A' = 65
        if self.relPreloadPath: key = self.relPreloadPath.split("/")[-1].split("_")[0]
        else: key = self.stageMask
        return key[:idx] + char + key[idx+1:]

    def name_from_key(self, key):
        """Generate model name from key."""

        losses = "_".join(self.lossFunctions)
        latent_dims = ".".join(map(str, self.latentDims))
        name = "{}_{}_{}".format(key, losses, latent_dims)
        name += "_{}".format(self.nameSuffix) if self.nameSuffix else ""
        return name

    def output(self):
        return luigi.LocalTarget(os.path.join(self.model_dir, 'weights.h5'))

    def run(self):
        with tf.device(self.gpu):

            # load model and weights
            preload_dir = fs.redirect(self.model_dir, self.relPreloadPath) if self.relPreloadPath else None
            model = self.load_model(weight_dir=preload_dir)

            # load data and others
            (x_train, x_test), (y_train, y_test)  = self.load_data()
            losses = loader.loss_loader(self.heads, self.lossFunctions)
            loss_weights = loader.loss_weights_loader(self.heads, self.lossWeights)
            opt = getattr(keras.optimizers, self.optimizer)(lr=self.learningRate)

            # compile model
            model.compile(loss=losses, loss_weights=loss_weights, optimizer=opt)
            model.set_trainable(self.stageMask)

            # start tensorboard
            from datetime import datetime
            from tensorboard import program
            tb = program.TensorBoard()
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = fs.mkdir(os.path.join(self.model_dir, "logs", timestamp))
            tb.configure(argv=[None, '--logdir', log_dir])
            url = tb.launch()
            print(f"Tensorboard running on {url}.", file=sys.stderr)

            # instantiate callbacks
            from common.callbacks import BaseCallback
            from common import callbacks
            from tasks import eval_plots
            cbs = []
            cbs.append(keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
            for fn in self.cbFns:
                cb = BaseCallback(cb_fn=getattr(eval_plots, fn), log_dir=log_dir, data=[(x_train, x_test), (y_train, y_test)], heads=self.heads, nbins=15)
                cbs.append(cb)

            # start training, save weights
            model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=self.nEpochs, batch_size=self.batchSize, callbacks=cbs, verbose=False)
            model.save_weights(self.output().path)

            # Print path to stdout for consecutive training
            print(self.model_dir)
