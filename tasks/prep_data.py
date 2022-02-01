import luigi
import pickle
import numpy as np

from common.tasks import DataTask

class PrepareData(DataTask):
    """Task to convert pickled dataset into npz array."""

    def output(self):
        return luigi.LocalTarget(self.data_target, format=luigi.format.Nop)

    def requires(self):
        return None

    def apply_threshold(self, digis, E_mod):
        mask = E_mod <= self.thresholdMeV
        E_mod[mask] = 0
        digis[mask] = 0
        return digis, E_mod

    def normalize(self, digis):
        return (digis - np.min(digis)) / (np.max(digis) - np.min(digis))

    def reshape8x8(self, digis):
        shape = list(digis.shape[:-4]) + [8,8,1]
        res = np.zeros(shape=shape)
        res[:, :, :, :, :4, :4] = digis[:, :, :, :, 2, :, :]
        res[:, :, :, :, :4, 4:] = np.rot90(digis[:, :, :, :, 1, :, :])
        res[:, :, :, :, 4:, 4:] = digis[:, :, :, :, 0, :, :]
        return res

    def run(self):

        with open(self.data_source, 'rb') as fin:
            data = pickle.load(fin)

        digis = data['digi'].reshape(-1,30,19,3,3,4,4,1)
        E_st2 = data['meta'][:,1]
        E_st1 = data['layer_energy']
        E_mod = data['module_energy']
        pdgID = data['meta'][:,0]
        gamma = data['meta'][:,2]
        theta = data['meta'][:,3]
        angle = np.stack((gamma, theta), axis=1)

        digis, E_mod = self.apply_threshold(digis, E_mod)  # apply threshold (high pass)
        for modifier in self.dataModifiers: digis = getattr(self, modifier)(digis)

        with self.output().open('wb') as fout:
            np.savez(fout, Reco_ST2=digis, Ereg_MOD=E_mod, Ereg_ST1=E_st1, Ereg_ST2=E_st2, Class_ST2=pdgID, Preg_ST2=angle)
