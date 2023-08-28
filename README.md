# Reconstructing axion-like particles from beam dumps with simulation-based inference

The code present here is used in [this paper](https://arxiv.org/abs/2308.01353)

In our work we used different detector setups with different geometries. Here we provide the code necessary to generate training samples, construct and train the cINN and derive the results of our paper.

## Repository content

We provide an [example notebook](/Example.ipynb) which works (hopefully) out of the box. In this example notebook the test samples are read out and the posterior on them can be evaluated with a pre-trained network. In this example notebook we also show how the mass and uncertainty on the mass can be derived. When these are derived for all the test samples we reproduce the performance plots in our paper. 

In [data](/data/) we provide the 4 test datasets which can be used directly with the provided example notebook. The code used to generate new samples and the Pythia simulated B meson spectra are also present. Further details can be found inside this folder. The B meson spectra have been obtained from ALPINIST https://arxiv.org/abs/2201.05170

In [yamls](/yamls/) you can save the files containing the detector and architecture parameters. The script [train_arch_cINN](/train_arch_cINN.py) takes as input a YAML file and returns a trained network which is saved in [modelscINN](/modelscINN/). 

[feat_extractor](/feat_extractor.py) is a useful script that reads the events out of the .csv files and return arrays with the low-level observables of interest (photon energies, photon hits, photon angles) in addition to the model parameters (ALP mass, ALP lifetime)

## Dependencies

Code uses (version refers to the software version used in the paper, not the minimum requirements):


- Python 3.6.15

- Tensorflow 2.6.0

- Tensorflow probability 0.14.1

- Keras 2.6.0

- Sklearn 0.23.2

- Numpy 1.19.5

- Pandas 1.1.5

- Matplotlib 3.3.4 (only needed for plotting)

- Corner.py 2.2.1 (only needed for plotting)
