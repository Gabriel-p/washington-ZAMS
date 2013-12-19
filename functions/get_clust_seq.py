# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 20:06:07 2013

@author: gabriel
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def clust_seqences(cluster, x, y, x_lim, y_lim, lev_num, kde, cluster_region,
                   kernel):
    '''This is the central function. It generates the countour plots around the
    cluster members. The extreme points of these contour levels are used to trace
    the ZAMS fiducial line for each cluster, according to a minimum level value
    allowed. The first method interpolates those points and then discards
    points according to the xy limits. The second method first discards points
    based on the xy limits and then interpolates the remaining ones.
    '''
    
    # This list will hold the points obtained through the contour curves,
    # the first sublist are the x coordinates of the points and the second
    # the y coordinates.
    contour_seq = [[], []]

    # Store contour levels.
    CS = plt.contour(x, y, kde)
    
    for i,clc in enumerate(CS.collections):
        for j,pth in enumerate(clc.get_paths()):
            cts = pth.vertices
            d = sp.spatial.distance.cdist(cts,cts)
            x_c,y_c = cts[list(sp.unravel_index(sp.argmax(d),d.shape))].T
            # Only store points that belong to contour PDF values that belong
            # to the uper curves, ie: do not use those with index < lev_num.
            if i >= lev_num:
                contour_seq[0].append(round(x_c[0],4))
                contour_seq[1].append(round(y_c[0],4))
                contour_seq[0].append(round(x_c[1],4))
                contour_seq[1].append(round(y_c[1],4))

    # If the sequence is an empty list don't attempt to plot the
    # polynomial fit.
    if contour_seq:
        
        poli_order = 2 # Order of the polynome.        
        
        # Method 1.
        # 1- Obtain the sequence's fitting polinome.
        poli = np.polyfit(contour_seq[1], contour_seq[0], poli_order)
        y_pol = np.linspace(min(contour_seq[1]),
                            max(contour_seq[1]), 50)
        p = np.poly1d(poli)
        x_pol = [p(i) for i in y_pol]
        # 2- Trim the interpolated sequence to the range in xy axis.
        y_pol_trim, x_pol_trim = zip(*[(ia,ib) for (ia, ib) in \
        zip(y_pol, x_pol) if x_lim[0] <= ib <= x_lim[1] and \
        y_lim[0] <= ia <= y_lim[1]])

        # Method 2.
        # 1- Trim the sequence to the xy range.
        y_trim, x_trim = zip(*[(ia,ib) for (ia, ib) in \
        zip(contour_seq[1], contour_seq[0]) if x_lim[0] <= ib <= x_lim[1] and \
        y_lim[0] <= ia <= y_lim[1]])
        # 2- Obtain the sequence's fitting polinome.
        poli = np.polyfit(y_trim, x_trim, poli_order)
        y_trim_pol = np.linspace(min(y_trim), max(y_trim), 50)
        p = np.poly1d(poli)
        x_trim_pol = [p(i) for i in y_trim_pol]
        
    else:
        x_pol_trim, y_pol_trim, x_trim_pol, y_trim_pol = [], [], [], []

    return x_pol_trim, y_pol_trim, x_trim_pol, y_trim_pol