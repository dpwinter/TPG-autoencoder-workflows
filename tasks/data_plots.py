import luigi
from luigi.util import inherits
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('common/styles/elegant.mplstyle')
import os, sys

from common.tasks import DataTask, DataParams
from tasks.prep_data import PrepareData

from common.utility import iterdo, fs
from common import loaders as loader

axis = {"ST2": (1,2,3,4,5,6), "ST1": (2,3,4,5,6), "MOD": (4,5,6), "SAMP": (0)}
stage_sum = lambda data, stage: np.sum(data, axis=axis[stage])
stage_units = {"ST2":"GeV", "ST1":"MeV", "MOD":"MeV"}

### Data plot base task

class DataPlotTask(DataTask):
    """Base task for DataPlotTask. Creates data/plots dir."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_plot_dir = fs.redirect(self.data_target, "plots")
        fs.mkdir(self.data_plot_dir)

    def requires(self):
        return self.clone(PrepareData)

### Data plot tasks

class PlotDigiVsTruth(DataPlotTask):
    """Plot encoded digis vs energy truth (MeV/GeV)
    for both training and test dataset."""

    stage = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(self.data_plot_dir, 'DigiVsTruth_{}.png'.format(self.stage)))

    def run(self):
        head = "Ereg"
        X, Y = loader.data_loader(self.input().path, [head], self.stage, split=False)
        x_train, x_test = iterdo.split(stage_sum(X, self.stage))
        y_train, y_test = iterdo.split(Y[head])

        fig = plt.figure(figsize=(10, 6)) 
        plt.scatter(y_train, x_train)
        plt.scatter(y_test, x_test)
        plt.ylabel('Digi energy sum ({}) [a.u.]'.format(self.stage))
        plt.xlabel('Truth label ({}) [{}]'.format(self.stage, stage_units[self.stage]))
        plt.title('Sum of (encoded) TC digis vs. {} energy truth'.format(self.stage))
        plt.legend(['train', 'test'], loc='upper right')
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.savefig(self.output().path, dpi=200)

class PlotDecodedDigiVsTruth(DataPlotTask):
    """Plot encoded-decoded digis vs energy truth (MeV/GeV)
    for both training and test dataset."""

    stage = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(self.data_plot_dir, 'DecodedDigiVsTruth_{}.png'.format(self.stage)))
    
    def run(self):
        from digi.ROC import E_keV_per_MIP, ADC_per_MIP, TCval_from_TCcode
        E_MeV_per_ADC = (E_keV_per_MIP * 1e-3) / ADC_per_MIP
        head = "Ereg"

        X, Y = loader.data_loader(self.input().path, [head], self.stage, split=False)
        decoded_digis = TCval_from_TCcode(X) * E_MeV_per_ADC
        x_train, x_test = iterdo.split(stage_sum(decoded_digis, self.stage))
        y_train, y_test = iterdo.split(Y[head])

        fig = plt.figure(figsize=(10, 6)) 
        plt.scatter(y_train, x_train)
        plt.scatter(y_test, x_test)
        plt.ylabel('Digi energy sum ({}) [MeV]'.format(self.stage))
        plt.xlabel('Truth label ({}) [{}]'.format(self.stage, stage_units[self.stage]))
        plt.title('Sum of decoded TC digis vs. {} energy truth'.format(self.stage))
        plt.legend(['train', 'test'], loc='upper right')
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.savefig(self.output().path, dpi=200)


class PlotTruthEnergyHist(DataPlotTask):
    """Plot histogram of energy truths in stage."""

    stage = luigi.Parameter()
    nbins = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(self.data_plot_dir, 'TruthEnergyHist_{}_{}bins.png'.format(self.stage, self.nbins)))

    def run(self):
        head = "Ereg"
        _, Y = loader.data_loader(self.input().path, [head], self.stage, split=False)

        fig = plt.figure(figsize=(10, 6)) 

        plt.hist(Y[head].flatten(), density=False, bins=self.nbins)
        plt.yscale('log')
        plt.ylabel('Occurence')
        plt.xlabel('Energy [{}]'.format(stage_units[self.stage]))
        plt.title('Log occurence of truth energies in {}'.format(self.stage))
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.savefig(self.output().path, dpi=200)


class PlotEnergyVsLayerByIndex(DataPlotTask):
    """Plot each layer's energy truths for one example."""

    exampleIndex = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(self.data_plot_dir, 'EnergyVsLayer_Sample{}.png'.format(self.exampleIndex)))

    def run(self):
        head = "Ereg"
        X, Y = loader.data_loader(self.input().path, [head], "ST1", split=False)
        X = stage_sum(X, "ST1")[self.exampleIndex].reshape(30, -1)
        Y = Y[head][self.exampleIndex].reshape(30, -1)

        fig, ax = plt.subplots(figsize=(10,6))
        ax.set_xlabel('Layer')
        ax.set_ylabel('Layer (digi) energy sum [a.u.]', color='blue')
        ax.plot(X)

        ax2 = ax.twinx()

        ax2.plot(Y, color='orange')
        ax2.set_ylabel('Layer (truth) energy label [MeV]', color='orange')
        plt.title('(Encoded) layer TC digi module sums and layer energy truth vs. layers for sample #{}'.format(self.exampleIndex))
        plt.grid(b=True, axis='y', which='major', color='#666666', linestyle='--')
        plt.savefig(self.output().path, dpi=200)


class PlotEnergyDistribVsLayer(DataPlotTask):
    """Plot distribution of module energy truths per layer."""

    def output(self):
        return luigi.LocalTarget(os.path.join(self.data_plot_dir, 'EnergyDistribVsLayer.png'))

    def run(self):
        head = "Ereg"
        X, Y = loader.data_loader(self.input().path, [head], "ST1", split=False)
        X = stage_sum(X, "ST1").reshape(-1, 30)
        Y = Y[head].reshape(-1, 30)

        fig, ax = plt.subplots(figsize=(10,6))
        vp = ax.violinplot(X)
        for pc in vp['bodies']:
            pc.set_facecolor('orange')
            pc.set_edgecolor('orange')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Layer (encoded) TC digi sum distribution [a.u.]', color='blue')

        ax2 = ax.twinx()

        ax2.violinplot(Y)
        ax2.set_ylabel('Layer (truth) energy distribution [MeV]', color='orange')
        plt.title('Distribution of (encoded) layer TC digis and energy truth vs. layer index')
        plt.grid(b=True, axis='y', which='major', color='#666666', linestyle='--')
        plt.savefig(self.output().path, dpi=200)

class PlotEnergyVsLayerAndModuleByIndex(DataPlotTask): 
    """Plot module energy sums for each layer for one example."""

    exampleIndex = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(self.data_plot_dir, 'EnergyVsLayerAndModule_Sample{}.png'.format(self.exampleIndex)))

    def run(self):
        head = "Ereg"
        X, Y_M  = loader.data_loader(self.input().path, [head], "MOD", split=False)
        _, Y_S2 = loader.data_loader(self.input().path, [head], "ST2", split=False)
        X = stage_sum(X, "MOD")[self.exampleIndex].reshape(30, -1)
        Y_M = Y_M[head][self.exampleIndex].reshape(30, -1)
        Y_S2 = Y_S2[head][self.exampleIndex]
        X, Y_M = np.swapaxes(X, 0, 1), np.swapaxes(Y_M, 0, 1)

        fig, ax = plt.subplots(1, 2, figsize=(13,6))
        fig.suptitle('Shower sample {}: e- with {:.1f} GeV ({} digis)'.format(self.exampleIndex, Y_S2, np.sum(X)))

        import seaborn as sns
        sns.heatmap(X, ax=ax[0], cmap="plasma", cbar_kws={'label': 'Module (digi) energy sum [a.u.]'})
        ax[0].set_title("Digi module sums in layers")
        ax[0].set_xlabel('Layer')
        ax[0].set_ylabel('Module')
        sns.heatmap(Y_M, ax=ax[1], cmap="plasma", cbar_kws={'label': 'Module (truth) energy sum [MeV]'})
        ax[1].set_title("Truth module energy in layers")
        ax[1].set_xlabel('Layer')
        ax[1].set_ylabel('Module')
        fig.tight_layout() 

        plt.savefig(self.output().path, dpi=200)


class PlotAccumulatedEnergiesVsLayerAndModule(DataPlotTask):
    """Plot module energy sums for each layer for all examples."""

    def output(self):
        return luigi.LocalTarget(os.path.join(self.data_plot_dir, 'AccumulatedEnergiesVsLayerAndModule.png'))

    def run(self):
        head = "Ereg"
        X, Y_M  = loader.data_loader(self.input().path, [head], "MOD", split=False)
        X = stage_sum( stage_sum(X, "MOD"), "SAMP" ).reshape(30, -1)
        Y_M = stage_sum(Y_M[head], "SAMP").reshape(30, -1)
        X, Y_M = np.swapaxes(X, 0, 1), np.swapaxes(Y_M, 0, 1)

        fig, ax = plt.subplots(1, 2, figsize=(13,6))
        fig.suptitle('e- accumulated (encoded) TC digi module sums and energy truths per layer')

        import seaborn as sns
        sns.heatmap(X, ax=ax[0], cmap="plasma", cbar_kws={'label': 'Module (digi) energy sum [a.u.]'})
        ax[0].set_title("Digi module sums in layers")
        ax[0].set_xlabel('Layer')
        ax[0].set_ylabel('Module')
        sns.heatmap(Y_M, ax=ax[1], cmap="plasma", cbar_kws={'label': 'Module (truth) energy sum [MeV]'})
        ax[1].set_title("Truth module energy in layers")
        ax[1].set_xlabel('Layer')
        ax[1].set_ylabel('Module')
        fig.tight_layout()

        plt.savefig(self.output().path, dpi=200)

class PlotTCcodeVsADCval(DataPlotTask):
    """Illustrate effect of encoding from truths to digis."""

    def output(self):
        return luigi.LocalTarget(os.path.join(self.data_plot_dir, 'ADCvalVsTCcode.png'))

    def run(self):

        # Illustrate maximal difference between digit and encoded digit
        from digi.ROC import E_keV_per_MIP, ADC_per_MIP, TCcode_from_ADC
        E_MeV_per_ADC = (E_keV_per_MIP * 1e-3) / ADC_per_MIP

        mod1 = [12 for _ in range(3*4*4*1)]
        mod2 = 12*3*4*4*1 # assume rest are 0.

        print("Sum of many values at end of linear region:", np.sum(TCcode_from_ADC(mod1)), file=sys.stderr)
        print("Sum of one large value:", np.sum(TCcode_from_ADC(mod2)), file=sys.stderr)

        fig, ax = plt.subplots()

        adc_range = range(1, int(1e5))
        xs = [i * E_MeV_per_ADC for i in adc_range]
        ys = TCcode_from_ADC([i for i in adc_range])
        ax.scatter(xs, ys)
        ax.set_xlabel('Module energy value [MeV]')
        ax.set_ylabel('Encoded TC digi code [a.u.]')
        ax.set_title('Conversion to 4E3M: Encoded TC digi code vs. Truth ADC value')

        axins = ax.inset_axes([0.55, 0.60, 0.40, 0.2])
        axins.scatter(xs[int(0.6e5):], ys[int(0.6e5):])
        axins2 = ax.inset_axes([0.15, 0.1, 0.8, 0.4])
        axins2.scatter(xs[:50], ys[:50])
        plt.savefig(self.output().path, dpi=200)

### Data plot task wrapper

@inherits(DataParams)  # fetch and pass through 
class DataPlots(luigi.WrapperTask):

    bins = luigi.IntParameter()
    exampleIndex = luigi.IntParameter()

    def requires(self):
        yield self.clone(PlotDigiVsTruth, stage='ST2')
        yield self.clone(PlotDigiVsTruth, stage='ST1')
        yield self.clone(PlotDigiVsTruth, stage='MOD')

        yield self.clone(PlotDecodedDigiVsTruth, stage='ST2')
        yield self.clone(PlotDecodedDigiVsTruth, stage='ST1')
        yield self.clone(PlotDecodedDigiVsTruth, stage='MOD')

        yield self.clone(PlotTruthEnergyHist, stage='ST2', nbins=self.bins)
        yield self.clone(PlotTruthEnergyHist, stage='ST1', nbins=self.bins)
        yield self.clone(PlotTruthEnergyHist, stage='MOD', nbins=self.bins)

        yield self.clone(PlotEnergyVsLayerByIndex, exampleIndex=self.exampleIndex)
        yield self.clone(PlotEnergyDistribVsLayer)
        yield self.clone(PlotEnergyVsLayerAndModuleByIndex, exampleIndex=self.exampleIndex)
        yield self.clone(PlotAccumulatedEnergiesVsLayerAndModule)
        yield self.clone(PlotTCcodeVsADCval)
