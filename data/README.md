The code contained in this folder can be used to generate training events and also test sets.

[ALP_decay](ALP_decay.py) contains the actual physical process simulation. The geometry of the dector has been hard-coded here.
If you want to consider a detector with a different geometry (with respect to the on axis detector consider), you should modify it here.
Geometry is contained in lines 22-26

The [ALP_decay](ALP_decay.py) code relies on simulated Pythia showers present in the file beauty_100kEvts_pp_8.2_400GeV_ptHat300MeV.txt

[generate_dataset](generate_dataset.py) is the code needed to generate the events and if you are not interested in modifying the detector geometry, the only piece of code you need.
Here you can specify the mass and lifetime range of interest.
The code generates sets composed by n_obs events. How many sets you generate is set by n_sets.
If you are interested in fixed mass and lifetimes, just set m_min=m_max, tau_min=tau_max.

Tips:
The speed of the event generation depends CRITICALLY on the mass and lifetimes of interest, so we highly recommend starting with a small n_sets to get a grasp of the total needed time.
The code benefits GREATLY from parallelization, so we recommend running on several CPUs in parallel for larger datasets generation.
Event weigths can be 0 within machine precision, this can be relevant for unweighted events.
