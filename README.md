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


# Outline of steps that follow:
#
# Get CMD coordinates and probabilities from prob_memb_avrg list. Used
# for the third plot.
# Calculate a probability limit above which stars will be used to draw
# the final sequence using a Gaussian fit.
# Obtain intrinsic position of stars above this probability limit.
# Obtain new CMD limits based on these intrinsic positions.
# Assign weights to these corrected stars according to the probabilities
# they have.
# Obtain the (weighted) KDE for these weighted stars.
# Generate a fiducial sequence making use of the KDE's contours.
# Interpolate this sequence to obtain the final sequence.
# Write final interpolated sequence to data file.

Data I/O
------------

### Data input

Needs several files as input to run:

1. File containing clusters data: location in disk, name center and radius.
2. ZAMS of several metallicities from `zams.data` file (present in repo)
3. Main photometric data file for each cluster.
4. File containing cluster's parameters: extinction, age, metallicity and distance modulus.
5. File containing membership probabilities for stars in the cluster region.

### Data output

1. Data file for each cluster containing the interpolated sequence traced
through its most probable members.
2. `.png` file for each cluster with 4 `CMD`s: all stars outside cluster region,
stars inside cluster region, stars inside colored according to its probabilities
and only most probable members and the sequence they trace.
3. Several `final_ZAMS_*.png` files, one for each metallicty interval defined.


Packages
-------

The dependencies listed are required to run the code. The versions are the ones I used,
could work with older versions but I can't guarantee it.

* [Numpy 1.8.0](http://www.numpy.org/) -- `sudo pip install numpy`
* [SciPy 0.12.0](http://www.scipy.org/) -- `sudo pip install scipy`
* [Matplotlib 1.2.1](http://matplotlib.org/) -- `sudo pip install matplotlib`