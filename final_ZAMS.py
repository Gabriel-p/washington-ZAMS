# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:49:53 2013

@author: gabriel
"""

'''
Trace a cluster's star sequence using the intrinsic (corrected) CMD and the
probabilities assigned to each star by a decontamination algorithm.

See README.md file for more information.
'''


import functions.get_data as gd
import functions.err_accpt_rejct as ear
import functions.get_in_out as gio
from functions.get_isochrones import get_isochrones as g_i
from functions.cluster_cmds import make_cluster_cmds as m_c_c
from functions.final_plot import make_final_plot as m_f_p

import numpy as np
from os.path import join, getsize
from os.path import expanduser
import glob
from scipy import stats
import scipy as sp
from itertools import chain

import matplotlib.pyplot as plt
from scipy.stats import norm


    
def clust_seqences(cluster, x, y, lev_min, lev_num, kde, cluster_region, kernel):
    '''This is the central function. It generates the countour plots around the
    cluster members. The extreme points of these contour levels are used to trace
    the ZAMS fiducial line for each cluster.
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
            # use those with index <= lev_num.
            if levels[i] > lev_min and i > lev_num:
                # Only store points within these limits.
                    seq_contour[0].append(round(x_c[0],4))
                    seq_contour[1].append(round(y_c[0],4))
                    seq_contour[0].append(round(x_c[1],4))
                    seq_contour[1].append(round(y_c[1],4))

    # This block is similar to the process above but his one generates the
    # countour plots around the cluster members and then makes use of those
    # stars inside the maximum contour allowed to trace the ZAMS instead of
    # the diametral points in the contours themselves.
    for star in cluster_region:
        kde_star = kernel((star[0], star[1]))
        for i,clc in enumerate(CS.collections):
            # Only use stars inside the allowed max contour and min level value.
            if levels[i] > lev_min and i > lev_num:
                if kde_star >= levels[i]:
                    # Only store stars within these limits.
                    seq_stars[0].append(star[0])
                    seq_stars[1].append(star[1])
                    break

    return seq_contour, seq_stars
        

def intrsc_values(col_obsrv, mag_obsrv, e_bv, dist_mod):
    '''
    Takes *observed* color and magnitude lists and returns corrected or
    intrinsic lists. Depends on the system selected/used.
    '''
    # For Washington system.
    #
    # E(C-T1) = 1.97*E(B-V) = (C-T1) - (C-T)o
    # M_T1 = T1 + 0.58*E(B-V) - (m-M)o - 3.2*E(B-V)
    #
    # (C-T1)o = (C-T1) - 1.97*E(B-V)
    # M_T1 = T1 + 0.58*E(B-V) - (m-M)o - 3.2*E(B-V)
    #
    col_intrsc = np.array(col_obsrv) - 1.97*e_bv
    mag_intrsc = np.array(mag_obsrv) + 0.58*e_bv - dist_mod - 3.2*e_bv
    
    return col_intrsc, mag_intrsc
    

def get_cluster_params():
    '''
    Read data_output file to store names and parameters of each cluster:
    sub dir, name, center and radius.
    '''
    
    # Set 'home' dir.
    home = expanduser("~")
    
    # Location of the data_output file
    out_dir = home+'/clusters/clusters_out/washington_KDE-Scott/'
    data_out_file = out_dir+'data_output'
    
    sub_dirs, cl_names, centers, radius = [], [], [[],[]], []
    with open(data_out_file, mode="r") as d_o_f:
        for line in d_o_f:
            li=line.strip()
            # Jump comments.
            if not li.startswith("#"):
                reader = li.split()            
                sub_dirs.append(reader[0].split('/')[0])
                cl_names.append(reader[0].split('/')[1])
                centers[0].append(float(reader[1]))
                centers[1].append(float(reader[2]))
                radius.append(float(reader[3]))
                
    return sub_dirs, cl_names, centers, radius, out_dir


def get_zams():
    '''
    Get ZAMS in its intrinsic position from file for plotting purposes.
    '''
    
    zams_file = 'zams.data'
    data = np.loadtxt(zams_file, unpack=True)
    # Convert z to [Fe/H] using the y=A+B*log10(x) zunzun.com function and the
    # x,y values:
    #   z    [Fe/H]
    # 0.001  -1.3
    # 0.004  -0.7
    # 0.008  -0.4
    # 0.019  0.0
    #    A, B = 1.7354259305164, 1.013629121876
    #    feh = A + B*np.log10(z)
    metals_z = [0.001, 0.004, 0.008, 0.0098, 0.0138, 0.019]
    metals_feh = [-1.3, -0.7, -0.4, -0.3, -0.15, 0.0]
    
    # List that holds all the isochrones of different metallicities.
    zam_met = [[] for _ in range(len(metals_z))]
    
    # Store each isochrone of a given metallicity in a list.
    for indx, metal_val in enumerate(metals_z):
        zam_met[indx] = map(list, zip(*(col for col in zip(*data) if\
        col[0] == metal_val)))
    
    return zam_met, metals_z, metals_feh


def get_manual_select(myfile):
    '''
    Read data file that contains the manually selected clusters and their
    fine tuning parameters to trace either its fiducial ZAMS sequence or its
    evolved isochrone section.
    '''
    manual_accept, ft_ylim, ft_level, ft_method = [], [], [], []
    with open(myfile, mode="r") as f_in:
        for line in f_in:
            li=line.strip()
            # Jump comments.
            if not li.startswith("#"):
                reader = li.split()            
                manual_accept.append(reader[0])
                ft_ylim.append([float(reader[1]), float(reader[2])])
                ft_level.append([float(reader[3]), float(reader[4])])
                ft_method.append(float(reader[5]))
    
    return manual_accept, ft_ylim, ft_level, ft_method
    
    

def get_probs(out_dir, sub_dir, cluster):
    '''
    Read the members file for each cluster and store the probabilities
    and CMD coordinates assigned to each star in the cluster region.
    '''
    prob_memb_avrg = []
    file_path = join(out_dir+sub_dir+'/'+cluster+'_memb.dat')
    with open(file_path, mode="r") as m_f:
        # Check if file is empty.
        flag_area_stronger = False if getsize(file_path) > 44 else True
        for line in m_f:
            li=line.strip()
            # Jump comments.
            if not li.startswith("#"):
                reader = li.split()     
                if reader[0] == '99.0':
                    prob_memb_avrg.append(map(float, reader))
                    
    return flag_area_stronger, prob_memb_avrg
                        
                        

def write_seq_file(out_dir, cluster, x_pol, y_pol):
    '''
    Write interpolated sequence to output file.
    '''
    out_file = join(out_dir+'fitted_zams'+'/'+cluster+'_ZAMS.dat')
    line = zip(*[['%.4f' % i for i in x_pol], ['%.4f' % i for i in y_pol]])
    with open(out_file, 'w') as f_out:
        f_out.write("#x_zams y_zams\n")
        for item in line:
            f_out.write('{:<7} {:>5}'.format(*item))
            f_out.write('\n')


def write_iso_file(out_dir, cluster, x_pol, y_pol):
    '''
    Write interpolated isochrone part to output file.
    '''
    out_file = join(out_dir+'fitted_zams'+'/'+cluster+'_iso.dat')
    line = zip(*[['%.4f' % i for i in x_pol], ['%.4f' % i for i in y_pol]])
    with open(out_file, 'w') as f_out:
        f_out.write("#x_iso y_iso\n")
        for item in line:
            f_out.write('{:<7} {:>5}'.format(*item))
            f_out.write('\n')
            

# **********************************************************************
# End of functions.



# Call function to obtain clusters locations, names. etc.
sub_dirs, cl_names, centers, radius, out_dir = get_cluster_params()


# Get ZAMS located at its instinsic position for plotting purposes.
zam_met, metals_z, metals_feh = get_zams()


# Ask for minimum probability threshold.
use_mu = False
prob_quest = raw_input('Use mu as probability threshold? (y/n): ')
if prob_quest == 'y':
    use_mu = True
else:
    min_prob = float(raw_input('Input minimum probability value to use: '))
    
    
# Ask if all clusters should be processed or only those in a list.
use_all_clust = raw_input('Use all clusters? (y/n): ')
if use_all_clust == 'n':
    # Read file that stores the manually selected clusters and read their fine
    # tuning parameters.
    myfile = 'manual_selection_zams.dat'
    zams_manual_accept, ft_z_ylim, ft_z_level, ft_z_method = \
    get_manual_select(myfile)
else:
    zams_manual_accept = []


# Read file that stores the manually selected clusters from which to obtain
# an isochrone. Store their names and fine tuning parameters.
myfile = 'manual_selection_isos.dat'
iso_manual_accept, ft_i_ylim, ft_i_level, ft_i_method = \
get_manual_select(myfile)


# Stores the CMD sequence obtained for each cluster.
clust_zams, clust_isoch = [], []
# Also store the parameters associated with each cluster.
clust_zams_params, clust_isoch_params = [], []

# Loop through all clusters processed.
for indx, sub_dir in enumerate(sub_dirs):
    cluster = cl_names[indx]

    # Check if cluster is in list.
    flag_all = False    
    if use_all_clust == 'y':
        run_cluster = True
        flag_all = True
    else:
        if cluster in zams_manual_accept or cluster in iso_manual_accept:
            run_cluster = True
        else:
            run_cluster = False
        
    if run_cluster:
        print sub_dir, cluster
        # Location of the photometric data file for each cluster.
        data_phot = '/media/rest/Dropbox/GABRIEL/CARRERA/3-POS-DOC/trabajo/\
data_all/cumulos-datos-fotometricos/'        
        
        # Get photometric data for cluster.
        filename = glob.glob(join(data_phot, sub_dir, cluster + '.*'))[0]
        id_star, x_data, y_data, mag_data, e_mag, col1_data, e_col1 = \
        gd.get_data(data_phot, sub_dir, filename)
        
        # Accept and reject stars based on their errors.
        bright_end, popt_mag, popt_umag, pol_mag, popt_col1, popt_ucol1, \
        pol_col1, mag_val_left, mag_val_right, col1_val_left, col1_val_right, \
        acpt_stars, rjct_stars = ear.err_accpt_rejct(id_star, x_data, y_data,
                                                     mag_data, e_mag, col1_data,
                                                     e_col1)

        clust_rad = [radius[indx], 0.]
        center_cl = [centers[0][indx], centers[1][indx]]
        # Get stars in and out of cluster's radius.
        stars_in, stars_out, stars_in_rjct, stars_out_rjct =  \
        gio.get_in_out(center_cl, clust_rad[0], acpt_stars, rjct_stars)
        
        # Path where the code is running
        ocaat_path = '/media/rest/github/OCAAT_code/'
        clust_name = cluster
        # Get manually fitted parameters for cluster, if these exist.
        cl_e_bv, cl_age, cl_feh, cl_dmod, iso_moved, zams_iso, iso_intrsc = \
        g_i(ocaat_path, clust_name)
                                                     
        # Read the members file for each cluster and store the probabilities
        # and CMD coordinates assigned to each star in the cluster region.
        flag_area_stronger, prob_memb_avrg = get_probs(out_dir, sub_dir, cluster)
    
    
        # Check if decont algorithm was applied.
        if not(flag_area_stronger):
            
            # Fit gaussian to probabilities distribution. The mean will act as
            # the prob threshold. Only stars with prob values above this mean
            # will be used to trace the sequence.
            if use_mu == True:
                prob_data = [star[8] for star in prob_memb_avrg]
                # Best Gaussian fit of data.
                (mu, sigma) = norm.fit(prob_data)
                min_prob = mu

            # Create list with stars with probs above min_prob.
            memb_above_lim = [[], [], []]
            for star in prob_memb_avrg:
                if star[8] >= min_prob:
                    memb_above_lim[0].append(star[6])
                    memb_above_lim[1].append(star[4])
                    memb_above_lim[2].append(star[8])

            # Get intrinsic color and magnitudes.
            col_intrsc, mag_intrsc = intrsc_values(memb_above_lim[0],
                                                   memb_above_lim[1], cl_e_bv, \
                                                   cl_dmod) 
                                                   
            # Define a cluster region in its intrinsic positions.                                       
            cluster_region = zip(*[col_intrsc, mag_intrsc])

            # Obtain limits selected as to make the intrinsic CMD axis 1:1.
            col1_min_int, col1_max_int = min(col_intrsc)-0.2, max(col_intrsc)+0.2
            mag_min_int, mag_max_int = max(mag_intrsc)+1., min(mag_intrsc)-1.
            delta_x = col1_max_int - col1_min_int
            delta_y = mag_min_int - mag_max_int
            center_x = (col1_max_int + col1_min_int)/2.
            center_y = (mag_max_int + mag_min_int)/2.
            if delta_y >= delta_x:
                col1_min_int, col1_max_int = (center_x-delta_y/2.),\
                (center_x+delta_y/2.)
            else:
                mag_max_int, mag_min_int = (center_y-delta_x/2.),\
                (center_y+delta_x/2.) 
                          
                          
            # Generate new stars located at the same positions of each star in
            # the list of most probable members. The number of new stars
            # generated in each star position is the weight assigned to that
            # star times 10. We do this so the KDE obtained below incorporates
            # the information of the weights, ie: the membership probabilities.
            col_intrsc_w = list(chain.from_iterable([i] * int(round(j* 10)) \
            for i, j in zip(col_intrsc, memb_above_lim[2])))
            mag_intrsc_w = list(chain.from_iterable([i] * int(round(j* 10)) \
            for i, j in zip(mag_intrsc, memb_above_lim[2])))        
        
  
            # Get KDE for CMD intrinsic position of most probable members.
            x, y = np.mgrid[col1_min_int:col1_max_int:100j,
                            mag_min_int:mag_max_int:100j]
            positions = np.vstack([x.ravel(), y.ravel()])
            values = np.vstack([col_intrsc_w, mag_intrsc_w])
            # The results are HEAVILY dependant on the bandwidth used here.
            # See: http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
            kernel = stats.gaussian_kde(values, bw_method = None)
            kde = np.reshape(kernel(positions).T, x.shape)
            
            if cluster in zams_manual_accept or flag_all:
                
                # Store y_lim and level values for this cluster.
                if cluster in zams_manual_accept:
                    indx = zams_manual_accept.index(cluster)
                    y_lim = ft_z_ylim[indx]
                    lev_min, lev_num = ft_z_level[indx][0], ft_z_level[indx][1]
                else:
                    y_lim = [-10., 10.]
                    lev_min, lev_num = 0., 0.
                
                # Call the function that returns the sequence determined by
                # the two points further from each other in each contour
                # level and the function that returns the sequence determined by
                # the stars inside the maximum contour level set.
                seq_contour, seq_stars = clust_seqences(cluster, x, y,
                                                        lev_min, lev_num, kde,
                                                        cluster_region, kernel)

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
                    # Trim the interpolated sequence to the range in y axis.
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
                    
                    
                # Write interpolating points to file and store in lists
                # according to the method that was selected.
                def_method = True
                if cluster in zams_manual_accept:
                    indx = zams_manual_accept.index(cluster)
                    if ft_z_method[indx] != 0:
                        print ft_z_method[indx]
                        x_pol_sel, y_pol_sel = x_pol_trim_2, y_pol_trim_2
                        def_method = False
                    else:
                        x_pol_sel, y_pol_sel = x_pol_trim, y_pol_trim
                else:
                    x_pol_sel, y_pol_sel = x_pol_trim, y_pol_trim
                        
                if x_pol_sel:
                    # Store the interpolated trimmed sequence obtained for this
                    # cluster in final list.
                    clust_zams.append([x_pol_sel, y_pol_sel])
                    # Also store the parameters associated with this cluster.
                    clust_zams_params.append([cluster, cl_e_bv, cl_age, cl_feh,
                                              cl_dmod])                        
                            
                    # Write interpolated sequence to output file.
                    write_seq_file(out_dir, cluster, x_pol_sel, y_pol_sel)                     
                    
            # If cluster is not to be processed.
            else:
                x_pol_trim, y_pol_trim, x_pol_trim_2, y_pol_trim_2 = [], [], [],\
                []
            
            
            if cluster in iso_manual_accept:
                
                indx = iso_manual_accept.index(cluster)
                y_lim_iso = ft_i_ylim[indx]
                lev_min_iso, lev_num_iso = ft_i_level[indx][0], ft_i_level[indx][1]
                
                # Call the function that returns the evolved part of the isochrone
                # for this cluster.
                isoch_seq_contour, isoch_seq_stars = clust_seqences(cluster, x,\
                y, lev_min_iso, lev_num_iso, kde, cluster_region, kernel)
            
            
                # If the sequence is an empty list don't attempt to plot the
                # polynomial fit.
                if isoch_seq_contour[0]:
                    # Obtain the sequence's fitting polinome.
                    poli_order = 2 # Order of the polynome.
                    poli = np.polyfit(isoch_seq_contour[1], isoch_seq_contour[0], poli_order)
                    y_pol_iso = np.linspace(min(isoch_seq_contour[1]),
                                        max(isoch_seq_contour[1]), 50)
                    p = np.poly1d(poli)
                    x_pol_iso = [p(i) for i in y_pol_iso]

                    # Trim the interpolated sequence to the range in y axis.
                    y_pol_trim_iso, x_pol_trim_iso = zip(*[(ia,ib) for (ia, ib) in \
                    zip(y_pol_iso, x_pol_iso) if y_lim_iso[0] <= ia <= y_lim_iso[1]])      
                else:
                    x_pol_trim_iso, y_pol_trim_iso = [], []

                # If the sequence is an empty list don't attempt to
                # plot the polynomial fit.
                if isoch_seq_stars[0]:
                    # Trim the interpolated sequence to the range in y axis.
                    y_trim_iso_2, x_trim_iso_2 = zip(*[(ia,ib) for (ia, ib) in \
                    zip(isoch_seq_stars[1], isoch_seq_stars[0]) if y_lim_iso[0] <= ia <= y_lim_iso[1]])
                    
                    # Obtain the sequence's fitting polinome.
                    poli_order = 2 # Order of the polynome.
                    poli = np.polyfit(y_trim_iso_2, x_trim_iso_2, poli_order)
                    y_pol_trim_iso_2 = np.linspace(min(y_trim_iso_2), max(y_trim_iso_2), 50)
                    p = np.poly1d(poli)
                    x_pol_trim_iso_2 = [p(i) for i in y_pol_trim_iso_2]
                else:
                    x_pol_trim_iso_2, y_pol_trim_iso_2 = [], []

                # Write interpolating points to file and store in lists
                # according to the method that was selected.
                def_method = True
                if cluster in iso_manual_accept:
                    indx = iso_manual_accept.index(cluster)
                    if ft_i_method[indx] != 0:
                        x_pol_sel, y_pol_sel = x_pol_trim_iso_2, y_pol_trim_iso_2
                        def_method = False
                    else:
                        x_pol_sel, y_pol_sel = x_pol_trim_iso, y_pol_trim_iso
                else:
                    x_pol_sel, y_pol_sel = x_pol_trim_iso, y_pol_trim_iso
                        
                if x_pol_sel:
                    # Store the interpolated trimmed sequence obtained for this
                    # cluster in final list.
                    clust_isoch.append([x_pol_sel, y_pol_sel])
                    # Also store the parameters associated with this cluster.
                    clust_isoch_params.append([cluster, cl_e_bv, cl_age, cl_feh,
                                              cl_dmod])
                    # Write interpolated sequence to output file.
                    write_iso_file(out_dir, cluster, x_pol_sel, y_pol_sel)
                      
                    
            else:
                x_pol_trim_iso, y_pol_trim_iso, x_pol_trim_iso_2,\
                y_pol_trim_iso_2 = [], [], [], []
        
        
            # Call function to create CMDs for this cluster.
            m_c_c(sub_dir, cluster, col1_data, mag_data, stars_out_rjct,
                  stars_out, stars_in_rjct, stars_in, prob_memb_avrg,
                  popt_mag, popt_col1, cl_e_bv, cl_age, cl_feh, cl_dmod,
                  iso_moved, iso_intrsc, zams_iso, col1_min_int, col1_max_int, 
                  mag_min_int, mag_max_int, min_prob, x, y, kde,
                  col_intrsc, mag_intrsc, memb_above_lim,
                  zam_met, metals_feh, x_pol_trim, y_pol_trim, x_pol_trim_2,
                  y_pol_trim_2, def_method, x_pol_trim_iso, y_pol_trim_iso,
                  x_pol_trim_iso_2, y_pol_trim_iso_2, out_dir)
        



def final_zams(clust_zams, clust_zams_params, m_rang, indx_met):
    '''
    Takes several cluster sequences and joins them into a sinclge ZAMS through
    interpolation.
    '''
    
    # Store in arrays the ages, names and names + ages for clusters inside the
    # metallicty range being processed.
    ages, names, names_feh = [], [], []
    for seq_param in clust_zams_params:
        if m_rang[0] <= seq_param[3] <= m_rang[1]:
            ages.append(seq_param[2])
            names.append(seq_param[0])
            names_feh.append(seq_param[0]+' ('+str(round(seq_param[2], 2))+')')
    ages = np.array(ages)

    # Skip if no sequences are inside this metallicity range.
    if len(ages) > 0:
    
        # Store interpolated (and possibly trimmed) sequences in single list.
        final_zams_poli = []
        for indx, seq in enumerate(clust_zams):
            if m_rang[0] <= clust_zams_params[indx][3] <= m_rang[1]:
                final_zams_poli.append([list(seq[0]), list(seq[1])]) 

        # Sort all lists according to age.
        ages_s, names_s, names_feh_s, final_zams_poli_s = \
        map(list, zip(*sorted(zip(ages, names, names_feh, final_zams_poli),
                              reverse=True)))         
        
        # Rearrange sequences into single list composed of two sub-lists: the
        # first one holds the colors and the second one the magnitudes.
        single_seq_list = [[i for v in r for i in v] for r in \
        zip(*final_zams_poli_s)]
        
        # Generate interpolated final ZAMS using the clusters sequences.
        pol_ord = 3
        poli_zams = np.polyfit(single_seq_list[1], single_seq_list[0], pol_ord)
        zy_pol = np.linspace(min(single_seq_list[1]),
                            max(single_seq_list[1]), 50)
        p = np.poly1d(poli_zams)
        zx_pol = [p(i) for i in zy_pol]

        # Write interpolated ZAMS to output file.
        out_file = join(out_dir+'fitted_zams/final_ZAMS_%d.dat' % indx_met)
        line = zip(*[['%.2f' % i for i in zx_pol], ['%.2f' % i for i in zy_pol]])
        with open(out_file, 'w') as f_out:
            f_out.write("#x_zams y_zams\n")
            for item in line:
                f_out.write('{:<7} {:>5}'.format(*item))
                f_out.write('\n')
        
    else:
        ages_s, names_s, names_feh_s, final_zams_poli_s, zx_pol, zy_pol =\
        [], [], [], [], [], []

    return ages_s, names_s, names_feh_s, final_zams_poli_s, zx_pol, zy_pol
    

def metal_isoch(clust_isoch, clust_isoch_params, zx_pol, zy_pol, m_rang):
    '''
    Selects those isochrone sequences located inside the metallicity range given.
    '''
    
    # Store in arrays the isoch sequences and params for clusters inside the
    # metallicty range being processed.
    clust_isoch_met, clust_isoch_params_met, iso_ages = [], [], []
    for indx,iso_param in enumerate(clust_isoch_params):
        if m_rang[0] <= iso_param[3] <= m_rang[1]:
            iso_ages.append(iso_param[2])
            clust_isoch_met.append(clust_isoch[indx])
            clust_isoch_params_met.append(iso_param)
            
    # Check for duplicated ages.
    
    # Interpolate a single isochrone from those with the same age.
    
    # Replace all the isochrones of same age with the newly interpolated one.
    
    # Interpolate points close to the final ZAMS so as to smooth the section
    # were they intersect.
    clust_isoch_met_z = []
    for isoch in clust_isoch_met:
        
        # Iterate though ZAMS points ordered in increasing order in y.
        for indx,y1_i in enumerate(zy_pol):
            if 0. <(y1_i-isoch[1][-1])<=0.4 and zx_pol[indx] > isoch[0][-1]:
                y3 = [y1_i]
                x3 = [zx_pol[indx]]
                break
       
        from scipy.interpolate import spline
        sx = np.array(isoch[0]+x3)
        sy = np.array(isoch[1].tolist()+y3)
        t  = np.arange(sx.size,dtype=float)
        t /= t[-1]
        N  = np.linspace(0,1,1000)
        SX = spline(t,sx,N,order=2)
        SY = spline(t,sy,N,order=2)
        
        # Append interpolated isochrone.
        clust_isoch_met_z.append([SX.tolist(), SY.tolist()])
    
    # Sort all lists according to age.
    if iso_ages:
        iso_ages_s, clust_isoch_met_s, clust_isoch_params_met_s = \
        map(list, zip(*sorted(zip(iso_ages, clust_isoch_met_z,
                                  clust_isoch_params_met), reverse=True)))
    else:
        iso_ages_s, clust_isoch_met_s, clust_isoch_params_met_s = [], [], []
        
    return iso_ages_s, clust_isoch_met_s, clust_isoch_params_met_s


print '\nPlotting sequences by metallicity interval'
# Define metallicity intervals to plot.
metal_ranges = [[-1.4, -0.71], [-0.7, -0.7], [-0.4, -0.4], [-0.3, -0.3],
                [-0.15, 0.01]]


# Create a plot for each metallicity range defined above.
for indx_met,m_rang in enumerate(metal_ranges):
    print 'Plotting %d' % indx_met
    
    
    # Call function that generates the final ZAMS from all the unique sequences.
    ages_s, names_s, names_feh_s, final_zams_poli_s, zx_pol, zy_pol = \
    final_zams(clust_zams, clust_zams_params, m_rang, indx_met)
    
    # Call function that selects only isochrones in the given metallicity range.
    iso_ages, clust_isoch_met, clust_isoch_params_met = metal_isoch(clust_isoch,\
    clust_isoch_params, zx_pol, zy_pol, m_rang)

    if len(ages_s) != 0:
        # Call function to generate plot for the metallicity range.
        m_f_p(out_dir, indx_met, m_rang[0], m_rang[1], zam_met, metals_z,
              metals_feh, ages_s, names_s, names_feh_s, final_zams_poli_s,
              zx_pol, zy_pol, iso_ages, clust_isoch_met, clust_isoch_params_met)
    else:
        print 'Skipped %d' % indx_met
        

print 'End.'