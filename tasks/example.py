import os, luigi

from common.decorators import *
from common.tasks import ExampleTask

class SetupModel(ExampleTask):

    def requires(self, delete_mode=False):
        return DefineDatasets(LatentSpaceDim=self.LatentSpaceDim)

    @AddOutputDirectory(os.path.join(os.environ["OUTPUT_DIR"], "models"), ID=ExampleTask.getLatentSpaceDim)
    @DefineLocalTargets
    def output(self):
        return {
            "cost": "cost.pdf",
            "anim": "anim.mp4",
            "model_encoder": "model_encoder",
            "model_decoder": "model_decoder",
        }

    @PythonCommand(os.path.join(os.environ["WORKFLOW_DIR"], "example/setup_model.py"))
    @UnpackPythonOptions
    def command(self):
        return {
            "dataTrain": self.input()["train"].path,     
            "dataTest": self.input()["test"].path,     
            "costFigure": self.output()["cost"].path,     
            "animFile": self.output()["anim"].path,     
            "modelFileEncoder": self.output()["model_encoder"].path,     
            "modelFileDecoder": self.output()["model_decoder"].path,     
            "batchSize": 32,     
            "NEpochs": 50,     
            "LatentDimension": self.LatentSpaceDim
        }

    @Subprocess(command)
    def run(self):
        pass   


class DefineDatasets(ExampleTask):

    @AddOutputDirectory(os.path.join(os.environ["OUTPUT_DIR"], "data"))
    @DefineLocalTargets
    def output(self):
        return {
            "train": "MNIST_train.npy",
            "test": "MNIST_test.npy",
        }

    @PythonCommand(os.path.join(os.environ["WORKFLOW_DIR"], "example/define_datasets.py"))
    @UnpackPythonOptions
    def command(self):
        return {
            "dataTrain": self.output()["train"].path,
            "dataTest": self.output()["test"].path
        }

    @Subprocess(command)
    def run(self):
        pass
