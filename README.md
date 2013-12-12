Washington-ZAMS
=============

**WARNING**: This code is under heavy development and not ready to be used.

Takes a star cluster with membership probabilities assigned to its stars
by a given decontamination algorithm and produces a fiducial sequence
tracing its most likely members.


Outline
------

1. Read data_output file to store names and parameters of each cluster:
   sub dir, name, center, radius, number of members.
2. Read clusters_data_isos.dat file to store isochrone parameters for each
   cluster.
3. Read the photometric data file for each cluster.
4. Read most_prob_memb file for each cluster to store the probabilities
and CMD coordinates assigned to each star.

5. Create the finding chart using the x,y coordinates, the assigned center
coordinates and the radius.
6. Create the r>R_c CMD using the color and magnitude of each star, the
assigned center and radius for the cluster.
7. Create the cluster CMD using the same process as above.
8. Create the last CMD using the data from the last file read.


Data I/O
------------

Want to contribute? Great! There are two ways to add markups.

### Data input

Idfdfgfg


### Data output

Outputs a data file for each cluster containing the sequence traced by its
members plus a few 


Packages
-------

The dependencies listed are required to run the code. The versions are the ones used,
could work with older versions but I can't guarantee it.

* [Numpy 1.8.0](http://www.numpy.org/) -- `sudo pip install numpy`
* [SciPy 0.12.0](http://www.scipy.org/) -- `sudo pip install scipy`
* [Matplotlib 1.2.1](http://matplotlib.org/) -- `sudo pip install matplotlib`