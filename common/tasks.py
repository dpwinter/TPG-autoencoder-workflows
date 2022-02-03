import luigi
from luigi.util import inherits
from common import loaders as loader
from common.utility import fs, iterdo, gpu
import os

class DataParams(luigi.Config):
    """Parameters for DataTasks. These get inherited by every task."""

    workingDir    = luigi.Parameter()
    dataset       = luigi.Parameter()
    thresholdMeV  = luigi.IntParameter()
    dataModifiers = luigi.ListParameter()

@inherits(DataParams)
class DataTask(luigi.Task):
    """Base task for DataTasks like PrepareData and DataPlots.
    Create dir structure, set source and target data paths."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        data_dir = os.path.join(self.workingDir, "data")
        self.data_source = os.path.join(data_dir, "source", "{}.pickle".format(self.dataset))
        assert os.path.exists(self.data_source)

        thr_data_src = self.dataset if self.thresholdMeV == 0 else "{}_{}+MeV".format(self.dataset, self.thresholdMeV)
        thr_data_src += "_" + "_".join(self.dataModifiers)
        self.data_target = os.path.join(data_dir, thr_data_src, "{}.npz".format(thr_data_src))


class ModelParams(luigi.Config):
    """Parameters for ModelTask. Inherited by TrainModel and EvalTask."""

    modelType     = luigi.Parameter()
    modelName     = luigi.Parameter()
    stageMask     = luigi.Parameter()
    latentDims    = luigi.ListParameter()

@inherits(DataParams, ModelParams)
class ModelTask(luigi.Task):
    """Base task for ModelTasks.
    Create dir structure until model class, set task vars."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data_name = os.path.splitext(os.path.basename(self.input().path))[0]
        self.model_cls_dir = os.path.join(self.workingDir, "eval", data_name, self.modelType, self.modelName, self.stageMask)
        fs.mkdir(self.model_cls_dir)

        self.stage = ["MOD", "ST1", "ST2"][self.stageMask.rindex('1')]
        self.heads = iterdo.camel_case_split(self.modelName)
        self.gpu = "/GPU:{}".format(gpu.mask_unused_gpus()[0])

    def load_data(self):
        """Convenience wrapper for data_loader."""

        return loader.data_loader(self.input().path, self.heads, self.stage, split=True)

    def load_model(self, weight_dir=None):
        """Load model and weights."""

        import random; random.seed(42)
        model = loader.model_loader(self.modelType, self.modelName, self.latentDims, self.stage)
        if weight_dir: model = loader.weights_loader(model, weight_dir)
        return model

    def requires(self):
        from tasks.prep_data import PrepareData
        return self.clone(PrepareData)


### not used ###

from abc import abstractmethod

class ExampleTask(luigi.Task):
    task_namespace = 'Example'
    LatentSpaceDim = luigi.IntParameter(default=28, description="Dimension of the latent space", significant=True)

    @abstractmethod
    def output(self):
        pass

    @abstractmethod
    def run(self):
        pass        

    @abstractmethod
    def command(self):
        pass    

    def requires(self, delete_mode=False):
        return []

    def getLatentSpaceDim(self):
        return "latentSpaceDim%i"%self.LatentSpaceDim

    #run a task manually, e.g. when yielded in a requirement function
    def manual_yield(self):
        _requirements = self.requires()
        if type(_requirements)==dict:
            for _req in _requirements:
                if not _req.complete():
                    _req.manual_yield()
                    _req.run()  
        elif type(_requirements)==list:
            for _req in _requirements:
                if not requirements[_req].complete():
                    requirements[_req].manual_yield()
                    requirements[_req].run()
        else:
            _req = _requirements
            if not _req.complete():
                _req.manual_yield()
                _req.run()            
        if not self.complete():
            self.run()
