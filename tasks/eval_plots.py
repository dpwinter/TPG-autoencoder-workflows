import luigi
from common.utility import fs, plot
from common.tasks import ModelTask, DataParams, ModelParams
from luigi.util import inherits
import matplotlib.pyplot as plt
plt.style.use('common/styles/elegant.mplstyle')
import numpy as np
import os

"""Eval functions: used for callback and EvalTask."""

def residualsVsTruth(model, data, head, *args, **kwargs):
    """R(y)=y_hat-y"""

    (x_train, x_test), (y_train, y_test) = data

    plt.clf() # start clean.
    fig = plt.figure(figsize=(6, 4)) 

    for i, (x, y) in enumerate([(x_train, y_train), (x_test, y_test)]):
        y_hat = model.predict(x)[head].flatten()
        y_tru = y[head].flatten()
        res = y_hat - y_tru
        plt.scatter(y_tru, res, label=plot.labels[i], color=plot.colors[i])

    plt.legend()
    plt.axhline(y=0, color='black', linestyle='--')
    plt.ylabel("$\hat{y}$ - y")
    plt.xlabel("y")
    return fig

def relResidualsVsTruth(model, data, head, *args, **kwargs):
    """R(y)=(y_hat-y)/y"""

    (x_train, x_test), (y_train, y_test) = data

    plt.clf() # start clean.
    fig = plt.figure(figsize=(6, 4)) 

    for i, (x, y) in enumerate([(x_train, y_train), (x_test, y_test)]):

        y_hat = model.predict(x)[head].flatten()
        y_tru = y[head].flatten()
        res = (y_hat - y_tru) / y_tru
        plt.scatter(y_tru, res, label=plot.labels[i], color=plot.colors[i])

    plt.legend()
    plt.axhline(y=0, color='black', linestyle='--')
    plt.ylabel("$(\hat{y}$ - y)/y")
    plt.xlabel("y")
    return fig

def relResidualsVsTruthBoxplot(model, data, head, nbins, *args, **kwargs):
    """Plot boxplot of binned R(y)=(y_hat-y)/y."""

    (x_train, x_test), (y_train, y_test) = data

    plt.clf() # start clean.
    fig = plt.figure(figsize=(6, 4)) 

    for i, (x, y) in enumerate([(x_train, y_train), (x_test, y_test)]):
        y_hat = model.predict(x)[head].flatten()
        y_tru = y[head].flatten()
        res = (y_hat - y_tru) / y_tru

        y_min, y_max = np.min(y_tru), np.max(y_tru)
        step = (y_max - y_min) / nbins
        bins = np.arange(y_min, y_max, step)
        binned_ids = np.digitize(y_tru, bins)
        binned_res = np.array([res[np.where(binned_ids==i)] for i in range(1,nbins+1)])

        n = (-1)**i * step/5
        bp = plt.boxplot(binned_res, positions=bins+n, widths=abs(n), sym='')
        plot.set_boxplot_color_idx(bp, i)

    plot.locs_to_labels(bins)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.legend()

    plt.ylabel("$(\hat{y}$ - y)/y")
    plt.xlabel("y")

    return fig

def relRmsVsTruth(model, data, head, nbins, *args, **kwargs):
    """Plot of binned RMS[R(y)]."""

    (x_train, x_test), (y_train, y_test) = data

    plt.clf() # start clean.
    fig = plt.figure(figsize=(6, 4)) 

    for i, (x, y) in enumerate([(x_train, y_train), (x_test, y_test)]):
        y_hat = model.predict(x)[head].flatten()
        y_tru = y[head].flatten()
        res = y_hat - y_tru

        y_min, y_max = np.min(y_tru), np.max(y_tru)
        step = (y_max - y_min) / nbins
        bins = np.arange(y_min, y_max, step)
        binned_ids = np.digitize(y_tru, bins)
        binned_res = np.array([res[np.where(binned_ids==i)] for i in range(1,nbins+1)])
        rms = lambda x: np.sqrt(np.mean(np.square(x)))
        binned_rms = [rms(a) for a in binned_res]
        binned_rel_rms = binned_rms / bins
        plt.scatter(bins, binned_rel_rms, color=plot.colors[i], label=plot.labels[i])

    plot.locs_to_labels(bins)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.legend()
    plt.ylabel("$RMS[(\hat{y}$ - y)]/y")
    plt.xlabel("y")
    return fig

### Eval Base Task

class EvalTask(ModelTask):
    """Base task for evaluation tasks."""

    trainName = luigi.Parameter()
    nbins = luigi.IntParameter()

    def __init__(self, eval_fn, *args, **kwargs):
        """Set model_dir and save kwargs."""

        super().__init__(*args, **kwargs)
        self.model_dir = os.path.join(self.model_cls_dir, self.trainName)
        print(self.model_dir)
        assert os.path.exists(self.model_dir)
        self.fn = eval_fn
        self.kwargs, self.args = kwargs, args

    def output(self):
        """Return dict: one plot for each head."""

        return {head: luigi.LocalTarget(os.path.join(self.model_dir, f"{head}_{self.fn.__name__}.png")) for head in self.heads}


    def run(self):
        """Load model and data, apply function and save to png."""

        import tensorflow as tf
        with tf.device(self.gpu):
            data  = self.load_data()
            model = self.load_model(weight_dir=self.model_dir)
            for head in self.heads:
                fig = self.fn(model, data, head, **self.kwargs)
                plt.savefig(self.output()[head].path, dpi=200)

### Eval Tasks

class PlotResiduals(EvalTask):
    def __init__(self, *args, **kwargs):
        super().__init__(eval_fn=residualsVsTruth, *args, **kwargs)

class PlotRelResiduals(EvalTask):
    def __init__(self, *args, **kwargs):
        super().__init__(eval_fn=relResidualsVsTruth, *args, **kwargs)

class PlotRelResidualsBoxplot(EvalTask):
    def __init__(self, *args, **kwargs):
        super().__init__(eval_fn=relResidualsVsTruthBoxplot, *args, **kwargs)

class PlotRelRMS(EvalTask):
    def __init__(self, *args, **kwargs):
        super().__init__(eval_fn=relRmsVsTruth, *args, **kwargs)

### Eval Task Wrapper

@inherits(DataParams, ModelParams)
class EvalPlots(luigi.WrapperTask):
    """Evaluation plot workflow. Expects DataParams and ModelParams to be set."""

    trainName = luigi.Parameter()
    nbins = luigi.IntParameter()

    def requires(self):
        yield self.clone(PlotResiduals, trainName=self.trainName) # clone: pass over all set parameters
        yield self.clone(PlotRelResiduals, trainName=self.trainName)
        yield self.clone(PlotRelResidualsBoxplot, trainName=self.trainName, nbins=self.nbins)
        yield self.clone(PlotRelRMS, trainName=self.trainName, nbins=self.nbins)
