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
    the ZAMS fiducial line for each cluster in the first block and the stars
    inside these contours (up to a maximum level) are used iun the second block.
    '''
    
    # This list will hold the points obtained through the contour curves,
    # the first sublist are the x coordinates of the points and the second
    # the y coordinates.
    seq_contour, seq_stars = [[], []], [[], []]

    # Store contour levels.
    CS = plt.contour(x, y, kde)
    # Store level values for contour levels.
    levels = CS.levels
    
    for i,clc in enumerate(CS.collections):
        for j,pth in enumerate(clc.get_paths()):
            cts = pth.vertices
            d = sp.spatial.distance.cdist(cts,cts)
            x_c,y_c = cts[list(sp.unravel_index(sp.argmax(d),d.shape))].T
            # Only store points that belong to contour PDF values larger
            # than lev_min and that belong to the uper curves, ie: do not
            # use those with index < lev_num.
            if i >= lev_num:
                # Only store points within these limits.
                if x_lim[0] <= x_c[0] <= x_lim[1] and \
                y_lim[0] <= y_c[0] <= y_lim[1]:
                    seq_contour[0].append(round(x_c[0],4))
                    seq_contour[1].append(round(y_c[0],4))
                if x_lim[0] <= x_c[1] <= x_lim[1] and \
                y_lim[0] <= y_c[1] <= y_lim[1]:
                    seq_contour[0].append(round(x_c[1],4))
                    seq_contour[1].append(round(y_c[1],4))

    # This block is similar to the process above but his one generates the
    # countour plots around the cluster members and then makes use of those
    # stars inside the maximum contour allowed to trace the ZAMS instead of
    # the diametral points in the contours themselves.
    for star in cluster_region:
        if lev_num > 0.:
            kde_star = kernel((star[0], star[1]))
            for i,clc in enumerate(CS.collections):
                # Only use stars inside the allowed max contour value.
                if i >= lev_num:
                    if kde_star >= levels[i]:
                        # Only store stars within these limits.
                        seq_stars[0].append(star[0])
                        seq_stars[1].append(star[1])
                        # Break so as not to store the same star more than once.
                        break
        else:
            # If lev_num is negative use ALL stars within the limits.
            if x_lim[0] <= star[0] <= x_lim[1] and \
            y_lim[0] <= star[1] <= y_lim[1]:
                seq_stars[0].append(star[0])
                seq_stars[1].append(star[1])
            

    # If the sequence is an empty list don't attempt to plot the
    # polynomial fit.
    if seq_contour[0]:
        # Obtain the sequence's fitting polinome.
        poli_order = 2 # Order of the polynome.
        poli = np.polyfit(seq_contour[1], seq_contour[0], poli_order)
        y_pol = np.linspace(min(seq_contour[1]),
                            max(seq_contour[1]), 50)
        p = np.poly1d(poli)
        x_pol = [p(i) for i in y_pol]

        # Trim the interpolated sequence to the range in y axis.
        y_pol_trim, x_pol_trim = zip(*[(ia,ib) for (ia, ib) in \
        zip(y_pol, x_pol) if y_lim[0] <= ia <= y_lim[1]])      
    else:
        x_pol_trim, y_pol_trim = [], []

    # If the sequence is an empty list don't attempt to
    # plot the polynomial fit.
    if seq_stars[0]:
        # Keep only those stars within the range in y axis.
        y_trim_2, x_trim_2 = zip(*[(ia,ib) for (ia, ib) in \
        zip(seq_stars[1], seq_stars[0]) if y_lim[0] <= ia <= y_lim[1]])                    
        
        # Obtain the sequence's fitting polinome.
        poli_order = 2 # Order of the polynome.
        poli = np.polyfit(y_trim_2, x_trim_2, poli_order)
        y_pol_trim_2 = np.linspace(min(y_trim_2), max(y_trim_2), 50)
        p = np.poly1d(poli)
        x_pol_trim_2 = [p(i) for i in y_pol_trim_2]
    else:
        x_pol_trim_2, y_pol_trim_2 = [], []

    return x_pol_trim, y_pol_trim, x_pol_trim_2, y_pol_trim_2