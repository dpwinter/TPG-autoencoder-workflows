# Autoencoder study for the L1 trigger of the CMS HGCAL upgrade

**Work very much in progress!**

## Introduction

### Ultimate goal

Transformation of HGCAL data to a manifold of (much) reduced dimensionality without loss of physical information. The dimension of the latent space will eventually be chosen to match the maximum trigger data rates in L1.

### Open questions to address

1. Auto encoding of calorimeter data: What is the smallest-possible dimension, e.g. for electromagnetic showers?
2. How to merge data from neighboring sensitive entities?
3. ...


## 1. Introductory example: Autoencoder for MNIST dataset

Encoding of MNIST data (28x28x1) to N-dimensional latent space with subsequent decoding. 

### Pre-requisites

Basically, `pip3 install -r requirements.txt`, or in more detail:
* Python 3 (Yes, I finally moved on from Python 2...)
* **luigi** for workflow management ([readthedocs](https://luigi.readthedocs.io/en/stable/))
* **tensorflow** (v2.2) ([website](https://www.tensorflow.org/install))
* **matplotlib** for state-of-the-art data visualisation ([website](https://matplotlib.org))
* **numpy** ([website](https://numpy.org))
* **imageio** for converting images to videos/gifs ([website](imageio))

### Running the code

* Set the ```WORKFLOW_DIR``` and ```OUTPUT_DIR``` environmental variables in ```setup.sh```. ```ROOT_SOURCE``` can be ignored for now.
* Start the luigi deamon in a separate window: `luigid`.
* ```source setup.sh```.
* Type ```SetupExampleModel <NDIM>``` with ```<NDIM>``` being the dimensionality of the input space.

### Example output with <NDIM>=28
![](img/MNIST_example_epoch0.png)
![](img/MNIST_example_cost.png)
![](img/MNIST_example_epoch50.png)


## Outlook: Potential next steps
- [ ] Setup work node with GPU, e.g. ```pclcdgpu.cern.ch```.
- [ ] Tune model on MNIST data to gain more insights on latent space dimensionality, network architecture, cost function, etc.
- [ ] Apply model on (simulated) 28-layer CE-E TB data. Define & implement physics-driven quality quantifiers.
- [ ] Own G4 simulation of HGCAL-like calorimeters. One could use [this tool](https://github.com/ThorbenQuast/HGCal_TB_Geant4/tree/master/simulation) for that purpose.
- [ ] ...