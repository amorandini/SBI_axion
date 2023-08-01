The code present here is used in (ref to arxiv link when available)

In our work we used different detector setups with different geometries. Here we provide the code necessary to generate training samples, construct and train the cINN and derive the results of our paper.

We provide an example notebook which works (hopefully) out of the box. In this example notebook the test samples are read out and the posterior on them can be evaluated with a pre-trained network. In this example notebook we also show how the mass and uncertainty on the mass can be derived. When these are derived for all the test samples we reproduce the performance plots in our paper. 

In /data/ we provide the 4 test datasets which can be used directly with the provided example notebook. The code used to generate new samples and the Pythia simulated B meson spectra are also present. Further details can be found inside this folder. The B meson spectra have been obtained from ALPINIST https://arxiv.org/abs/2201.05170

In the folder /yamls/ there are supposed to be files containing the detector and architecture parameters. The script /train_arch_cINN.py takes as input a YAML file and returns a trained network which is saved in modelscINN. 

/feat_extractor.py is a useful script that reads the events out of the .csv files and return arrays with the low-level observables of interest (photon energies, photon hits, photon angles) in addition to the model parameters (ALP mass, ALP lifetime)
