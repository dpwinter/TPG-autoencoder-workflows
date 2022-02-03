from common.utility import iterdo
import numpy as np
import importlib
import os, sys

heads = ['Reco', 'Ereg', 'Class', 'Preg']
head_from_model = lambda model: next(h for h in heads if h in model)
stage_sel = lambda data, model, stage: data[ "_".join([head_from_model(model), stage]) ]

def data_loader(path, heads, stage, split=True):
    """Load data, prepare for training and split."""

    data = np.load(path, allow_pickle=True)
    X = stage_sel(data, "Reco", "ST2")
    if split:
        y_train, y_test = {}, {}
        for h in heads:
            y_train[h], y_test[h] = iterdo.split( stage_sel(data, h, stage) )
        X = iterdo.split(X)
        Y = (y_train, y_test)
    else:
        Y = {h:stage_sel(data, h, stage) for h in heads}
    return X, Y

def model_loader(model_type, model_name, latent_dims, stage):
    """Load model from disk."""

    return getattr(importlib.import_module("models.{}.{}".format(model_type, model_name)), model_name)(latent_dims, stage)

def weights_loader(model, weights_dir):
    """Load weights and compare weights before and after loading."""

    w0s = [layer.get_weights() for submodel in model.layers for layer in submodel.layers]
    model.load_weights(os.path.join(weights_dir, "weights.h5"), by_name=True, skip_mismatch=True)
    w1s = [layer.get_weights() for submodel in model.layers for layer in submodel.layers]
    names = [layer.name for submodel in model.layers for layer in submodel.layers]
    for (eq, name) in zip(iterdo.compare_tensors(w0s, w1s), names):
        if not eq: print("Loaded weights for layer {}".format(name), file=sys.stderr)
    return model

def loss_loader(heads, losses):
    """Load losses from disk."""

    return {h:getattr(importlib.import_module("common.losses"), l) for h,l in zip(heads,losses)}

def loss_weights_loader(heads, loss_weights):
    """Prepare loss_weights for training."""

    return {h:w for h,w in zip(heads,loss_weights)}

