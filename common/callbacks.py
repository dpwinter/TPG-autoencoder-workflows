import os
import keras
from tasks import eval_plots
from common.utility import fs, iterdo
import tensorflow as tf

class BaseCallback(keras.callbacks.Callback):
    """Callback class to evaluate model.
    cb_fn:   callback function
    log_dir: tensorboard log directory
    data:    train and test data
    heads:   heads to evaluate"""

    def __init__(self, cb_fn, log_dir, data, heads, *args, **kwargs):
        super().__init__()
        self.data = data
        self.fn = cb_fn
        self.args, self.kwargs = args, kwargs # save to pass to cb_fn

        self.file_writers = {}
        for head in heads:
            file_writer = tf.summary.create_file_writer(f"{log_dir}/{head}_{self.fn.__name__}")
            self.file_writers[head] = file_writer

    def on_epoch_end(self, epoch, logs={}):
        """Call function for every head and save to tf summary file."""

        for head, file_writer in self.file_writers.items():
            fig = self.fn(model=self.model, data=self.data, head=head, **self.kwargs)
            img = iterdo.plot_to_image(fig)
            with file_writer.as_default():
                tf.summary.image(self.fn.__name__, img, step=epoch)
