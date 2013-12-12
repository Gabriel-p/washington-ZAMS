Washington-ZAMS
=============

**WARNING**: This code is under heavy development and not ready to be used.

Takes a star cluster with membership probabilities assigned to its stars
by a given decontamination algorithm and produces a fiducial sequence
tracing its most likely members.


Outline
------

1. Read `data_output` file to store names and parameters for each cluster:
   sub dir, name, center and radius.
2. Read `zams.data` file to store theoretical ZAMS.
3. Input minimum probability threshold (automatic value `mu` can be used)
4. Ask if fine tuning parameters should be used.
5. Ask if all clusters should be processed or only those inside the `manual_accept` list.
6. Apply these processes to each cluster:
 6.1. Read the photometric data file for each cluster.
 6.2. Classify stars as in or out of cluster boundaries.
 6.3. Read `clusters_data_isos.dat` file to store fitted parameters for each cluster.
 6.4. Read `most_prob_memb` file for each cluster to store the probabilities
and CMD coordinates assigned to each star.
 6.5. Calculate `mu` value if selected to be used. This is the mean of a Gaussian
 fit of all probabilities assigned to stars inside the cluster region.
 6.6. Store CMD coordinates and probabilities only for stars above the threshold.
 6.7. Obtain intrinsic position of stars above the probability threshold.
 6.8. Obtain CMD limits based on these intrinsic positions.
 6.9. Assign weights to these corrected stars according to the membership probabilities
 they've been assigned.
 6.10. Obtain the (weighted) KDE for these (weighted) stars.
 6.11. Generate a fiducial sequence making use of the KDE contours' extreme points.
 6.12. Interpolate this sequence to obtain the final sequence.
 6.13. Write final interpolated sequence to data file.
 6.14. Create out `.png` file for this cluster.
7. Plot interpolated sequences and final ZAMS for each metallicity interval defined.

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