#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 2023
@author: obarrois

Python version of PARODY-JA4.56-Base/Base_Field Matlab file 'Matlab/curlbal_QGA_4.m'.
Loads graphics file and compute curl balances of coriolis, lorentz and vorticity.
Can also plot results in meridional slices.

Script to ONLY plot the force balances (one needs to use an other script to compute them)!
"""

#import os
#import sys
from vizuals import merContour, equatContour, radialContour
#from parobpy.core_properties import icb_radius, cmb_radius
from parobpy.plot_lib import merid_outline
from parobpy.parob_lib import curl_sph# get_curl, 
from parobpy.load_parody import parodyload, load_dimensionless, load_basefield
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cmocean.cm as cmo
#import shtns

l_show = True # allow Figures to be displayed or not 
if ( not l_show ): #-- Recommended if running the script only to compute and store Figures!
    import matplotlib
    matplotlib.use('Agg')  # backend for no display if needed

#----------------------------------------------------------------------------%%
#-- INPUT PARAMETERS

#-- Lehnert number, \lambda = B/{sqrt(rho mu) * Omega*d}.
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '3e7', 'Pm0o25', 1.46e-3, 1.0, 'b3', '0.000280800' # 3e-7 S=1214.1
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1', 1.1e-3, 1.0, 'b4', '0.000040320' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1', 1.1e-3, 1.0, 'bbi1e7', '0.000071360' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1BIS', 1.1e-3, 1.0, 'bbi1e7B', '0.000020800' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1_per', 1.1e-3, 1.0, 'bbi1e7p', '0.000147200' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1_perSin20000/Lund4e2', 4.2e-3, 1.0, 'bbi1e7ps20000lS', '0.000109200' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1_PathBase/perSin70000pb', 1.48e-3, 1.0, 'bbi1e7ps70000pb', '0.000183040' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1_PathBase/perSin4096pb/ElssTwice', 1.57e-3, 1.0, 'bbi1e7ps4096pbhE', '0.000183040' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1_PathBase/perSin20000pb/ElssHalf', 7.84e-4, 1.0, 'bbi1e7ps20000pblE', '0.000409600' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1_PathBase/perSin10000pb/ElssTrue', 1.1e-3, 0.565, 'bbi1e7ps10000pbtE', '0.000410880' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1_PathBase/perSin85000pb/IY66', 1.1e-3, 1.0, 'bbi1e7ps85000pbiY66', '0.000183040' # 1e-7
run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1_SimpBase/perSin10000', 1.1e-3, 1.0, 'bbi1e7ps10000', '0.000600000' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1_SimpBase/perSin10000/ElssTwice', 1.57e-3, 1.0, 'bbi1e7ps10000hE', '0.000614400' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1_SimpBase/perSin70000/ElssHalf', 7.84e-4, 1.0, 'bbi1e7ps70000lE', '0.000612800' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1_SimpBase/perSin160000/BY66-IY66', 1.1e-3, 1.0, 'bbi1e7ps160000bfY66i', '0.000586000' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1_SimpBase/perSin20000/IY66', 1.1e-3, 1.0, 'bbi1e7ps20000iY66', '0.000628800' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1Long', 1.1e-3, 1.0, 'b4-long', '0.000614400' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'PathLund3o2e3', 5.53e-4, 1.0, 'b4o6', '0.000015600' # 6.3e-9 (1e7 grid)
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Stoch', 1.1e-3, 1.0, 'b6', '0.000053120' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'All_Parameters_Modified/Pm1o44e-2', 3.50e-3, 1.0, 'b4-2', '0.000008640' # 1e-7 # All param. modified!
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o0e-1', 1.1e-3, 1.0, 'b4-3', '0.000090240' # 7e-8
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm2o88e-1', 1.1e-3, 1.0, 'b3-bis', '0.000144000' # 2e-7
#un_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'PathBase', 1.1e-3, 1.0, 'b4-p', '0.000008755' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'All_Parameters_Modified/PmHalf', 1.56e-3, 1.0, 'b4-2-bis', '0.000154240' # 1e-7 # All param. modified!
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'All_Parameters_Modified/ElssHalf', 1.40e-3, 0.8, 'b4-2-ter', '0.000155200' # 1e-7 # All param. modified!
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Lund8e2', 1.1e-3, 0.5, 'b4-5', '0.000288000' # 2e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Lund3o2e3', 1.1e-3, 2.0, 'b4-5-bis', '0.000077920' # 5e-8
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Lund1o6e4', 1.1e-3, 5.0, 'b4-5-ter', '0.000031680' # 2e-8 #Actually Lund8e3!!
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'All_Parameters_Modified/Lund2', 1.57e-3, 1.0, 'b4-5-bis', '0.000288000' # 2e-7 # All param. modified!
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Lehn2o2e-4', 2.2e-4, 0.2, 'b4-4-ter', '0.000072000' # 2e-8
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Lehn5o5e-4', 5.5e-4, 0.5, 'b4-4', '0.000058080' # 5e-8
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Lehn2o2e-3', 2.2e-3, 2.0, 'b4-4-bis', '0.000152960' # 2e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'All_Parameters_Modified/Lehn2', 7.84e-4, 1.0, 'b4-4-bis', '0.000162800' # 5e-8 # All param. modified!
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e8', 'PathLund3o2e3', 5.53e-4, 1.0, 'b4o6-2', '0.000008800' # 6.3e-9 (1e8 grid)
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e8', 'PathLund3o2e3BIS', 5.53e-4, 1.0, 'b4o6-2', '0.000009080' # 6.3e-9 (1e8 grid)BIS
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e8', 'Pm0o46e-1', 6.2e-4, 1.0, 'b4o5', '0.000021840' # 1e-8
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e8', 'Pm0o46e-1', 6.2e-4, 1.0, 'bbi1e8', '0.000015040' # 1e-8
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e8', 'Pm0o46e-1BIS', 6.2e-4, 1.0, 'b4o5', '0.000024896' # 1e-8
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '6e9', 'Pm0o36e-1', 5.53e-4, 1.0, 'b4o63', '0.000015560' # 6.3e-9
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '6e9', 'Pm0o36e-1BIS', 5.53e-4, 1.0, 'b4o63B', '0.000019760' # 6.3e-9
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '3e10', '', 2.6e-4, 1.0, 'b5', '0.000001456' # 3e-10

#saveDir = '/gpfs/users/obarrois/Parodys/Outputs/Data/' # if you use python directly from cluster
#directory = '/gpfs/users/obarrois/Work/Waves1e7/' # if you use python directly from cluster
directory = '/Users/obarrois/Desktop/stella/Work/Waves'+run_Ek+'/'+run_Pm+'/' # if you mount stella on own desk

fig_aspect = 1 # figure aspect ratio
n_levels = 31 # default no. of contour levels
Vmax = 250 # default max Vp
Bmax = 2.5 # default max Bp

rank = 0 # default proc number when l_mpi == False or None

l_spectral = True # using spectral derivatives in theta and phi for curl? (advised:  = True)
l_trunc = False # truncation for SH?

l_resym = False # re-symmetrizing the data if minc !=1
l_curl_b0 = False # re-computing curl of Background field B0
l_check_b0 = False # display background B0 field
l_check_all = False # display all check plot figures
l_redim = True # re-dimensionalisation of the quantities
l_tau_A = True # re-dimensionalisation on the Alfven or on the Viscous time-scales
l_recalc_dt = True #-- probably better to recompute d . /dt it in this script where the re-dimensionalisation is done "properly"

#-- I/O; NOTE: does not seem to replace files if they exist even if l_save*!
saveDir = '/Users/obarrois/Desktop/Parodys/Outputs/Waves'+run_Ek+'/Balances/Data/'#Old_Computations/ # path to save colmunar balance files
#saveDir = '/Users/obarrois/Desktop/stella/Parodys/Outputs/Waves'+run_Ek+'/Data/' # if you use python from perso to remote data directly from cluster
l_read_bal = 1 # read balances?
l_read_balzavg = 1 # save zAvg-balances?
l_read_ja = 1 # not for every cases...
tagRead = '_-phi-Avg_n500_perSin10000'#_-phi-Avg_n500_Long'#_-phi-Avg_n500_Y66'#_-phi-Avg_n600'#_-phi-Avg_n500_lowLund'#_-phi-Avg_n500_lowLehn'#_-phi-Avg_n500_lowerPm'# #  # tag at the end of the forces files if needed
l_save = 1 # save main figures?
l_spdf = 1 # save figures in pdf format? (Format Barrois and Aubert 2024)
l_fix_CB = 1 # impose a pre-defined Colorbar or let matplotlib decide?
l_old_plot = 0 # old plots, before Barrois and Aubert 2024
savePlot = '/Users/obarrois/Desktop/Parodys/Outputs/Waves'+run_Ek+'/Balances/' # path to save main figures
l_make_movie = False # make a serie of figures for a movie?
saveMovie = '/Users/obarrois/Desktop/Parodys/Outputs/Waves'+run_Ek+'/' # path to save files for a movie

#------------------------------------------------------------------------------
#%% Initialisation

Gt_file = 'Gt={}.{}'.format(timestamp,run_ID)
filename = directory + Gt_file

n_samples = 1
(version, time, DeltaU, Coriolis, Lorentz, Buoyancy, ForcingU,
            DeltaT, ForcingT, DeltaB, ForcingB, Ek, Ra, Pm, Pr,
            nr, ntheta, nphi, azsym, radius, theta, phi, _, _, _,
            _, _, _, _) = parodyload(filename)

#basename = directory+'basefield.mat' # if you use python directly from cluster
if ( run_Pm == 'PathBase' ):
    basedir = '/Users/obarrois/Desktop/stella/Work/' # basefield copied on stella
    basename = basedir+'basefield_path.mat' # if you use python directly from cluster
else:
    basedir = '/Users/obarrois/Desktop/stella/Work/Waves'+run_Ek+'/' # basefield copied on stella
    basename = basedir+'basefield.mat' # if you use python directly from cluster
B0r, B0t, B0p, j0r, j0t, j0p = load_basefield(basename,nr,ntheta,nphi)

NR, Ek, Ra, Pr, Pm, minc, mcl, fi, rf = load_dimensionless(run_ID, directory)
Elsasser = 1.33 #-- Computed from Bessel B0_tot field #NOTE: Actually Brms from bessel --> must be squared to calculate Lehnert: Elsasser = Brms**2 = 1.33**2
#Elsasser = Elsasser*np.sqrt(0.5) #-- if changing parameters #/(4./3.)
#Lehnert = np.sqrt(Elsasser**2*Ek/Pm)
#Lundquist = np.sqrt(Elsasser**2*Pm/Ek)

#Lehnert = 1.1e-3*np.sqrt(2.)
#Ek = 7.e-8
#Pm = 0.1

phir = phi.copy()
nphir = nphi
if ( l_resym and minc !=1 ):
    from parobpy.parob_lib import symmetrize
    nphir = nphi*minc##+1 NOTE: last point is spurious in MagIC.symmetrize
    phir = np.linspace(phi[0],3*phi[-1], nphir)
    B0r = symmetrize(B0r,ms=minc); j0r = symmetrize(j0r,ms=minc)
    B0t = symmetrize(B0t,ms=minc); j0t = symmetrize(j0t,ms=minc)
    B0p = symmetrize(B0p,ms=minc); j0p = symmetrize(j0p,ms=minc)
amp_B = np.sqrt(amp_B) #-- sqrt(amp_b) because of parody unit in B_rms
B0r*=amp_B; B0t*=amp_B; B0p*=amp_B
j0r*=amp_B; j0t*=amp_B; j0p*=amp_B

#-- if needed:
if ( l_curl_b0 ):
    #-- grid mesh helpers
    grid = [radius, theta, phir]
    r3D = np.zeros_like(B0r)
    for i in range(nr):
        r3D[:,:,i] = radius[i]
    #
    th3D = np.zeros_like(B0r)
    for j in range(ntheta):
        th3D[:,j,:] = theta[j]
    #
    sint3D = np.sin(th3D)
    cost3D = np.cos(th3D)
    s3D = r3D*sint3D
    grid_help = [r3D, sint3D, s3D]
    #
    #-- prepare sh_struct for Spectral transforms
    print('call get curl, l_spectral = ', l_spectral)
    if ( l_spectral ):
        import shtns
        if ( l_trunc ):
            l_max = 16
            m_max = 16
        else: # NOTE: WARNING!!: need to get the l_max/m_max of the simulation
            l_max = 133# nphir//3
            #m_max = 48# l_max# nphi//3#
            if ( l_resym and minc !=1 ):
                m_max = nphir//3#--> 48
            else:
                m_max = nphi//3#--> 16
        #
        sh = shtns.sht(l_max, m_max)
        nlat, nlon = sh.set_grid(nphi=nphir, nlat=ntheta)
        if ( False ):#if ( not l_resym and minc !=1 ): # Test to get tab2 like in curlbal_QGA_4_JA.m
            sh.imtab = sh.m.copy()
            for lm in range(sh.nlm):
                if ( np.mod(lm,3) != 0 ):
                    sh.imtab[lm] = 0
            sh.m = sh.imtab
    else:
        sh = None
    #
    #-- If one wants to benchmark curl computation
    j0r2, j0t2, j0p2 = curl_sph(grid_help, B0r, B0t, B0p, l_mpi=False, l_spectral=False, sh_struct=None)
    #j0r2, j0t2, j0p2 = curl_sph(grid_help, B0r, B0t, B0p, l_mpi=l_mpi, l_spectral=l_spectral, sh_struct=sh)
    if ( np.sum(np.isnan(j0r2)) + np.sum(np.isnan(j0t2)) + np.sum(np.isnan(j0p2)) != 0 ): print('Warning!:: nan in curl B0 computations!') # to check if there are any problems
    print('sum diff j0_read, j0_recomputed = ', np.sum(j0r != j0r2) + np.sum(j0t != j0t2) + np.sum(j0p != j0p2))
    if ( l_check_b0 and rank==0 ):
        n_levels=21
        merContour(j0r.mean(axis=0), radius, levels=n_levels, cm='PuOr_r')
        merContour(j0t.mean(axis=0), radius, levels=n_levels, cm='PuOr_r')
        merContour(j0p.mean(axis=0), radius, levels=n_levels, cm='PuOr_r')
        merContour(j0r2.mean(axis=0), radius, levels=n_levels, cm='PuOr_r')
        merContour(j0t2.mean(axis=0), radius, levels=n_levels, cm='PuOr_r')
        merContour(j0p2.mean(axis=0), radius, levels=n_levels, cm='PuOr_r')
        merContour((j0r-j0r2).mean(axis=0), radius, levels=n_levels, cm=cmo.balance)
        merContour((j0t-j0t2).mean(axis=0), radius, levels=n_levels, cm=cmo.balance)
        merContour((j0p-j0p2).mean(axis=0), radius, levels=n_levels, cm=cmo.balance)
        radialContour(j0r[:,:,0], rad=0, levels=n_levels, cm='PuOr_r')
        radialContour(j0t[:,:,0], rad=0, levels=n_levels, cm='PuOr_r')
        radialContour(j0p[:,:,0], rad=0, levels=n_levels, cm='PuOr_r')
        radialContour(j0r2[:,:,0], rad=0, levels=n_levels, cm='PuOr_r')
        radialContour(j0t2[:,:,0], rad=0, levels=n_levels, cm='PuOr_r')
        radialContour(j0p2[:,:,0], rad=0, levels=n_levels, cm='PuOr_r')
        radialContour((j0r-j0r2)[:,:,0], rad=0, levels=n_levels, cm=cmo.balance)
        radialContour((j0t-j0t2)[:,:,0], rad=0, levels=n_levels, cm=cmo.balance)
        radialContour((j0p-j0p2)[:,:,0], rad=0, levels=n_levels, cm=cmo.balance)
        plt.show()
    #
    #-- checking plots for b0
    if ( l_check_b0 and rank==0 ):
        n_s = 4
        n_levels=31
        radialContour(B0r[:,:,-1], rad=0, levels=n_levels, cm='PuOr_r')
        radialContour(B0t[:,:,-1], rad=0, levels=n_levels, cm='PuOr_r')
        radialContour(B0p[:,:,-1], rad=0, levels=n_levels, cm='PuOr_r')
        merContour(B0r.mean(axis=0), radius, levels=n_levels, cm='PuOr_r')
        merContour(B0t.mean(axis=0), radius, levels=n_levels, cm='PuOr_r')
        merContour(B0p.mean(axis=0), radius, levels=n_levels, cm='PuOr_r')
        radialContour(B02r[:24,:,-1], rad=0, levels=n_levels, cm='PuOr_r')
        radialContour(B02t[:24,:,-1], rad=0, levels=n_levels, cm='PuOr_r')
        radialContour(B02p[:24,:,-1], rad=0, levels=n_levels, cm='PuOr_r')
        merContour(B02r.mean(axis=0), radius, levels=n_levels, cm='PuOr_r')
        merContour(B02t.mean(axis=0), radius, levels=n_levels, cm='PuOr_r')
        merContour(B02p.mean(axis=0), radius, levels=n_levels, cm='PuOr_r')
        plt.show()

#------------------------------------------------------------------------------
#%% Read force curls

#-- read the forces directly from a save file or re-compute them
if ( l_read_bal ):
    import h5py
    f = h5py.File(saveDir+'balances'+tagRead+'.hdf5', 'r')
    print('Reading columnar balances with Fields: %s' % f.keys())
    fkeys = list(f.keys())
    timet = np.array(f['time'])
    radius = np.array(f['radius'])
    theta = np.array(f['theta'])
    phir = np.array(f['phi'])
    Vp = np.array(f['VpZonal'])
    ups = np.array(f['upSlice'])
    omzs = np.array(f['omzSlice'])
    Lorzs = np.array(f['LorzSlice'])
    Corzs = np.array(f['CorzSlice'])
    f.close()
    n_samples = ups.shape[0]
    #-- Only slice or a phi-averaged quantities are stored in _col.files: no need to re-compute phi-average
    Vpmt = Vp[1:].mean(axis=0)
    upmtmm = ups[1:].mean(axis=0)
    omzmtmm = omzs[1:].mean(axis=0)
    Lorzmtmm = Lorzs[1:].mean(axis=0)
    Corzmtmm = Corzs[1:].mean(axis=0)
    titleTimeAvg = 'slice-time-Avg'
else:
    timet = np.zeros((n_samples),)
    Vp = np.zeros((n_samples,ntheta,nr),)
    ups = np.zeros_like(Vp)
    omzs = np.zeros_like(ups)
    Lorzs = np.zeros_like(omzs)
    Corzs = np.zeros_like(omzs)
    #-- zeros
    Vpmt = Vp[1:].mean(axis=0)
    upmtmm = ups[1:].mean(axis=0)
    omzmtmm = omzs[1:].mean(axis=0)
    Lorzmtmm = Lorzs[1:].mean(axis=0)
    Corzmtmm = Corzs[1:].mean(axis=0)
    titleTimeAvg = 'slice-time-Avg'

#-- Time and Forces Redimensionalisation, if needed
if ( l_tau_A ):
    tredim = 1./(Ek/Lehnert)
    Lordim = 1./(Pm*Lehnert)
    Cordim = 1./(Lehnert)
else:
    tredim = 1.#/(Ek/Lehnert)*1/(Lehnert/Ek)
    Lordim = 1./(Pm*Ek)#/(Pm*Lehnert)*(Lehnert/Ek)
    Cordim = 1./Ek#/(Lehnert)*(Lehnert/Ek)
tdim = timet*tredim#/(Ek/Lehnert)
if ( l_redim ):
    Lorzs=Lorzs*Lordim#/(Pm*Lehnert)
    Corzs=Corzs*Cordim#/(Lehnert)

#-- checking plots
if ( l_check_all and rank==0 ):
    n_levels = 31
    Vmax = np.amax(abs(Vpmt))#/10.
    umax = np.amax(abs(upmtmm))#/10.
    omax = np.amax(abs(omzmtmm))/10.
    Lmax = np.amax(abs(Lorzmtmm))/10.
    Cmax = np.amax(abs(Corzmtmm))/10.
    merContour(Vpmt, radius, levels=n_levels, vmin=-Vmax, vmax=Vmax, cm=cmo.balance)
    merContour(upmtmm, radius, levels=n_levels, vmin=-umax, vmax=umax, cm=cmo.balance)
    merContour(omzmtmm, radius, levels=n_levels, vmin=-omax, vmax=omax, cm=cmo.tarn_r)
    merContour(Lorzmtmm, radius, levels=n_levels, vmin=-Lmax, vmax=Lmax, cm='PuOr_r')
    merContour(Corzmtmm, radius, levels=n_levels, vmin=-Cmax, vmax=Cmax, cm='seismic')
    plt.figure()
    plt.plot(radius, Vpmt[ntheta//2,:],lw=2.1,label='Vp_zonal')
    plt.plot(radius, upmtmm[ntheta//2,:],lw=2.1,label='up')
    plt.plot(radius, omzmtmm[ntheta//2,:],lw=2.1,label='$\omega_z$')
    plt.plot(radius, Lorzmtmm[ntheta//2,:],lw=2.1,label='Lorentz')
    plt.plot(radius, Corzmtmm[ntheta//2,:],lw=2.1,label='Coriolis')
    plt.ylabel('equatorial section of '+titleTimeAvg+' Forces')
    plt.grid(ls=':')
    plt.legend()
    plt.show()

#-- plot all figures and save them for a movie if l_make_movie
if ( l_make_movie and rank==0 ):
    n_levels = 51
    for n_s in range(n_samples):
        Vmax = np.amax(abs(Vp[n_s]))/2.
        umax = np.amax(abs(ups[n_s]))/2.
        omax = np.amax(abs(omzs[n_s]))/10.
        Lmax = np.amax(abs(Lorzs[n_s]))/20.
        Cmax = np.amax(abs(Corzs[n_s]))/20.
        if ( np.mod(n_s,4)==0 ): plt.close('all')
        print('Plotting Forces for file n°{}/{}, t={}'.format(n_s+1, n_samples, timet[n_s]))
        merContour(Vp[n_s]/Vmax, radius, levels=n_levels, vmin=-1., vmax=1., cm=cmo.balance)#vmin=-Vmax, vmax=Vmax, 
        merContour(ups[n_s]/umax, radius, levels=n_levels, vmin=-1., vmax=1., cm=cmo.balance)#vmin=-umax, vmax=umax, 
        merContour(omzs[n_s]/omax, radius, levels=n_levels, vmin=-1., vmax=1., cm=cmo.tarn_r)#vmin=-omax, vmax=omax, 
        merContour(Lorzs[n_s]/Lmax, radius, levels=n_levels, vmin=-1., vmax=1., cm='PuOr_r')#vmin=-Lmax, vmax=Lmax, 
        merContour(Corzs[n_s]/Cmax, radius, levels=n_levels, vmin=-1., vmax=1., cm='seismic')#RdYlBu_r')#vmin=-Cmax, vmax=Cmax, 
        if ( l_save ):
            plt.figure(1)
            plt.savefig(saveMovie+'frame_Uzon_fr_'+str(n_s)+'.png')
            plt.figure(2)
            plt.savefig(saveMovie+'frame_up_fr_'+str(n_s)+'.png')
            #plt.figure(3)
            #plt.savefig(saveMovie+'frame_omz_fr_'+str(n_s)+'.png')
            #plt.figure(4)
            #plt.savefig(saveMovie+'frame_Lorz_fr_'+str(n_s)+'.png')
            #plt.figure(5)
            #plt.savefig(saveMovie+'frame_Corz_fr_'+str(n_s)+'.png')
        else:
            if ( l_check_all and np.mod(n_s,4)==3 ): plt.show()
    if ( n_s > 3 ):
        plt.close('all')
    else:
        plt.show()

#------------------------------------------------------------------------------
#%% Read z-Avg balances

rrad = radius[::-1] # zavg() expect reversed radius as in pizza/MagIC (r_cmb = r[0])
nsmax =  201#--> to get nsmax ~ nsmaxja # nsmax = int(nr*1.5 + 16) # to maintain nsmax ~ nrmax outside of TC (but will give nsmax > nrmax)

#-- read the columnar averaged forces directly from a save file or re-compute them
if ( l_read_balzavg ):
    #import h5py
    f = h5py.File(saveDir+'Zavg_bal'+tagRead+'.hdf5', 'r')
    print('Reading columnar balances with Fields: %s' % f.keys())
    fkeys = list(f.keys())
    sr = np.array(f['srad'])
    timet = np.array(f['time'])
    Vpcol = np.array(f['Vpcol'])
    upcol = np.array(f['upcol'])
    omzcol = np.array(f['omzcol'])
    Lorcol = np.array(f['Lorcol'])
    Corcol = np.array(f['Corcol'])
    dVpcol = np.array(f['dVpcol'])
    domzcol = np.array(f['domzcol'])
    #AgeosCorcol = np.array(f['AgeosCorcol']) # only a by-product of Lorcol and dVpcol
    f.close()
    #-- Forces Redimensionalisation, if needed
    if ( l_redim ):
        Lorcol=Lorcol*Lordim#/(Pm*Lehnert)
        Corcol=Corcol*Cordim#/(Lehnert)
        #-- New standard from March 10 2023:
        #--     d . /dt computed using timet (adimensionalised as in Parody): has to be re-dimensionaliased here as well
        dVpcol=dVpcol/tredim
        domzcol=domzcol/tredim
else:
    sr = np.zeros((nsmax),)
    timet = np.zeros((n_samples),)
    Vpcol = np.zeros((n_samples,nsmax),)
    upcol = np.zeros_like(Vpcol)
    omzcol = np.zeros_like(Vpcol)
    Lorcol = np.zeros_like(Vpcol)
    Corcol = np.zeros_like(Vpcol)
    dVpcol = np.zeros_like(Vpcol)
    domzcol = np.zeros_like(Vpcol)

if ( l_recalc_dt ):
    #-- Vp and domzdt balances
    dVpcol = np.zeros_like(upcol)
    domzcol = np.zeros_like(omzcol)
    o2dt = 0.5/np.diff(timet.astype(np.float32)).mean() #dt=mean(diff(tt));
    for n_s in range(1,n_samples-1):
        dVpcol[n_s,:] = (upcol[n_s+1,:] - upcol[n_s-1,:])*o2dt
        domzcol[n_s,:] = (omzcol[n_s+1,:] - omzcol[n_s-1,:])*o2dt
    dVpcol = dVpcol/tredim
    domzcol = domzcol/tredim

#-- Remaining of the balance: should be equivalent to the Ageostrophic Forces!
AgeosCorcol=Lorcol-dVpcol
Residuals = Lorcol-Corcol-domzcol

#-- Testing filter z-avg
if ( l_read_ja ):#l_read_balzavg ):#
    #import h5py
    f = h5py.File(saveDir+'Zavg_bal_ja'+tagRead+'.hdf5', 'r')
    print('Reading columnar JA balances with Fields: %s' % f.keys())
    fkeys = list(f.keys())
    srja = np.array(f['radiusja'])
    omzcolja = np.array(f['omzcolja'])
    Lorcolja = np.array(f['Lorcolja'])
    Corcolja = np.array(f['Corcolja'])
    domzcolja = np.array(f['domzcolja'])
    f.close()
    #-- Forces Redimensionalisation, if needed
    if ( l_redim ):
        Lorcolja=Lorcolja*Lordim#/(Pm*Lehnert)
        Corcolja=Corcolja*Cordim#/(Lehnert)
        #-- New standard from March 10 2023:
        #--     d . /dt computed using timet (adimensionalised as in Parody): has to be re-dimensionaliased here as well
        domzcolja=domzcolja/tredim
else:
    srja = np.zeros((nsmax),)
    omzcolja = np.zeros((n_samples,nsmax),)
    Lorcolja = np.zeros_like(omzcolja)
    Corcolja = np.zeros_like(omzcolja)
    domzcolja = np.zeros_like(omzcolja)

if ( l_recalc_dt ):
    #-- Vp and domzdt balances
    domzcolja = np.zeros_like(omzcolja)
    o2dt = 0.5/np.diff(timet.astype(np.float32)).mean() #dt=mean(diff(tt));
    for n_s in range(1,n_samples-1):
        domzcolja[n_s,:] = (omzcolja[n_s+1,:] - omzcolja[n_s-1,:])*o2dt
    domzcolja = domzcolja/tredim

#-- Remaining of the balance: should be equivalent to the Ageostrophic Forces!
Residualsja = Lorcolja-Corcolja-domzcolja

#-- checking plots
if ( l_check_all and rank==0 ):
    n_s = 8
    plt.figure()
    plt.plot(sr, Lorcol[n_s,:], 'r-o', lw=2.1, label='Lorentz')
    plt.plot(sr, Vpcol[n_s,:], 'y-.x', lw=2.1, alpha=0.8, label='Vp zonal')
    plt.plot(sr, omzcol[n_s,:], 'g-.*', lw=2.1, alpha=0.8, label='Vorticity')
    plt.plot(sr, Corcol[n_s,:], 'b--+', lw=2.1, alpha=0.8, label='Coriolis')
    plt.grid(ls=':')
    plt.xlim(sr[-1],sr[0])
    plt.xlabel('s', fontsize=30)
    plt.ylabel('z-avg of Forces[t,phi]', fontsize=30)
    plt.gca().xaxis.set_tick_params(labelsize=32)
    plt.gca().yaxis.set_tick_params(labelsize=32)
    plt.legend(fontsize=16)

    plt.figure()
    plt.plot(sr, Lorcol.mean(axis=0), 'r-o', lw=2.1, label='Lorentz')
    plt.plot(sr, Vpcol.mean(axis=0), 'y-.x', lw=2.1, alpha=0.8, label='Vp zonal')
    plt.plot(sr, omzcol.mean(axis=0), 'g-.*', lw=2.1, alpha=0.8, label='Vorticity')
    plt.plot(sr, Corcol.mean(axis=0), 'b--+', lw=2.1, alpha=0.8, label='Coriolis')
    plt.grid(ls=':')
    plt.xlim(sr[-1],sr[0])
    plt.xlabel('s', fontsize=30)
    plt.ylabel('Time-avg z-avg of Forces[phi]', fontsize=30)
    plt.gca().xaxis.set_tick_params(labelsize=32)
    plt.gca().yaxis.set_tick_params(labelsize=32)
    plt.legend(fontsize=16)
    plt.show()

#------------------------------------------------------------------------------
#%% Slice summary plot

#-- Plot time and phi averaged Meridional slices
if ( rank==0 ):
    Vmax = np.amax(abs(Vpmt))#np.amax(abs(upmtmm))#
    ommax= np.amax(abs(omzmtmm))/20.
    Bmax = np.amax(abs(Lorzmtmm))/150.
    Cmax = np.amax(abs(Corzmtmm))/80.

    w, h = plt.figaspect(fig_aspect)
    fig = plt.figure(constrained_layout=True, figsize = (1.5*w,2*h))
    spec = gridspec.GridSpec(ncols = 2, nrows = 2, figure=fig)

    #-- Velocity
    ax = fig.add_subplot(spec[0,0])
    X = np.outer(np.sin(theta),radius)
    Y = np.outer(np.cos(theta),radius)
    Z = Vpmt#upmtmm#
    Z_lim = Vmax
    levels = np.linspace(-Z_lim,Z_lim,n_levels)
    c = ax.contourf(X,Y,Z,levels,cmap=cmo.balance,extend='both')
    cbar=plt.colorbar(c,ax=ax, aspect = 50, ticks=levels[::2])
    #cbar.ax.set_title(r'$\mathbf{u}$')
    cbar.ax.set_title(r'$\mathbf{u}_0$')
    merid_outline(ax,radius)
    ax.axis('off')

    #-- Vorticity
    ax = fig.add_subplot(spec[0,1])
    Z = omzmtmm
    Z_lim = ommax
    levels = np.linspace(-Z_lim,Z_lim,n_levels)
    c = ax.contourf(X,Y,Z,levels,cmap=cmo.tarn_r,extend='both')
    cbar=plt.colorbar(c,ax=ax, aspect = 50, ticks=levels[::2])
    cbar.ax.set_title(r'$\omega_z$')
    merid_outline(ax,radius)
    ax.axis('off')

    #-- Lorentz
    ax = fig.add_subplot(spec[1,0])
    Z = Lorzmtmm
    Z_lim = Bmax
    levels = np.linspace(-Z_lim,Z_lim,n_levels)
    c = ax.contourf(X,Y,Z,levels,cmap='PuOr_r',extend='both')
    cbar=plt.colorbar(c,ax=ax, aspect = 50, ticks=levels[::2])
    cbar.ax.set_title(r'$\mathbf{B}$')
    merid_outline(ax,radius)
    ax.axis('off')

    #-- Coriolis
    ax = fig.add_subplot(spec[1,1])
    Z = Corzmtmm
    Z_lim = Cmax
    levels = np.linspace(-Z_lim,Z_lim,n_levels)
    c = ax.contourf(X,Y,Z,levels,cmap='seismic',extend='both')
    cbar=plt.colorbar(c,ax=ax, aspect = 50, ticks=levels[::2])
    cbar.ax.set_title(r'$\mathbf{Cor}$')
    merid_outline(ax,radius)
    ax.axis('off')

#-- Save summary plot
if ( l_save and rank==0 ):
    #if not os.path.exists(saveDir+'/{}'.format(run_ID)):
    #    os.makedirs(saveDir+'/{}'.format(run_ID)) #-- create a new directory if needed
    fig.savefig(saveMovie+'{}/Mer_'.format('Balances')+titleTimeAvg+'_Balances.png',format='png',
                dpi=200,bbox_inches='tight') # NOTE: slightly less convenient to use ''.format() when appending strings
    fig.savefig(saveMovie+'{}/Mer_'.format('Balances')+titleTimeAvg+'_Balances.pdf', format='pdf',
                dpi=200,bbox_inches='tight')
    print('Figures saved as {}Mer_'.format(savePlot)+titleTimeAvg+'_Balances.*')
    plt.close('all')
else:
    if ( rank==0 ): fig.show()

#------------------------------------------------------------------------------
#%% Compute trajectories along Aflven speeds

dirva = '/Users/obarrois/Desktop/stella/Work/Waves'+run_Ek+'/'
try:
    Valfven = np.loadtxt(dirva+'Va')
    l_plot_Va = True
except FileNotFoundError:
    print("Alfven speed: File Not Found")
    Valfven = np.zeros((nr),)
    l_plot_Va = False
trajcolor='#32CD32' # Lime # '#99CC32' # Jaune-Vert # '#FF7F00' # Orange

n_Asteps = 25000
rtraj = np.zeros((n_Asteps),); ttraj = np.zeros_like(rtraj)
#rtraj[0] = srja[0]; ttraj[0] = tdim[0]
rtraj[0] = 0.54; ttraj[0] = 0.
advtime = 1.e-2
for istep in range(1,n_Asteps):
    i = np.sum(radius<=rtraj[istep-1]) # i = np.sum(srja<=rtraj[istep-1]) #i=sum(r<=rtraj(step-1));
    j = np.sum(tdim<=ttraj[istep-1]) #j=sum(tt<=ttraj(step-1));
    Val = Valfven[i-1]
    #
    #print(istep, i, j, Val, rtraj[istep-1])
    rtraj[istep] = rtraj[istep-1] + Val*advtime #rtraj(step)=rtraj(step-1)+Val*advtime;
    if ( rtraj[istep] > 0.995*radius[-1] ): #if rtraj(step)>0.99*r(nr);
        advtime = 0.
    ttraj[istep] = ttraj[istep-1] + advtime #ttraj(step)=ttraj(step-1)+advtime;


l_double_plot=True
if ( l_double_plot ):
    dirva = '/Users/obarrois/Desktop/stella/Work/Waves'+run_Ek+'/'
    try:
        Valfven = np.loadtxt(dirva+'Va_loc_fast')
        l_plot_Va = True
    except FileNotFoundError:
        print("Alfven speed: File Not Found")
        Valfven = np.zeros((nr),)
        l_plot_Va = False
    trajcolor2= '#FF7F00' # Orange # '#32CD32' # Lime # '#99CC32' # Jaune-Vert # 
    #Valfven[:] = 0.8

    n_Asteps = 25000
    rtraj2 = np.zeros((n_Asteps),); ttraj2 = np.zeros_like(rtraj)
    #rtraj2[0] = srja[0]; ttraj2[0] = tdim[0]
    rtraj2[0] = 0.54; ttraj2[0] = 0.
    advtime = 5.e-3#1.e-2#
    for istep in range(1,n_Asteps):
        i = np.sum(radius<=rtraj2[istep-1]) #i=sum(r<=rtraj(step-1));
        j = np.sum(tdim<=ttraj2[istep-1]) #j=sum(tt<=ttraj(step-1));
        Val = Valfven[i]
        #
        #print(istep, i, j, Val, rtraj2[istep-1])
        rtraj2[istep] = rtraj2[istep-1] + Val*advtime #rtraj(step)=rtraj(step-1)+Val*advtime;
        if ( rtraj2[istep] > 0.995*radius[-1] ): #if rtraj(step)>0.99*r(nr);
            advtime = 0.
        ttraj2[istep] = ttraj2[istep-1] + advtime #ttraj(step)=ttraj(step-1)+advtime;

#------------------------------------------------------------------------------
#%% Adjusting colorbars and else for differnt set ups

#-- Palette
WA_Dar1_lightBlue = '#5bbcd6'
WA_Dar1_Green = '#01a08a'
WA_Dar1_Gold = '#f2ad00'
WA_Dar2_Cyan = '#abddde'
WA_Ziss_Yellow = '#ebcc2a'
WA_Ziss_Gold = '#e1af00'
WA_FFox_Yellow = '#e2d201'
WA_FFox_Cyan = '#46acc8'
WA_Rush_Green = '#0c775e'
WA_BRoc_Dark = '#0d1606'
WA_BRoc_Yellow = '#fad510'
WA_BRoc_Green = '#344822'
WA_BRoc_Black = '#1e1e1e'
WA_Mon3_Cyan = '#85d4e3'
Personal_Cyan = '#1cfaf7'

#-- Adjusting colorsfor trajs #CB Barrois and Aubert 2024
trajcolor = WA_BRoc_Dark
trajcolor2= WA_BRoc_Dark#Personal_Cyan#WA_Dar1_lightBlue#WA_BRoc_Yellow

#-- Labels for plots in Article #e.g. Barrois and Aubert 2024
PLabels = [r'$(a)$', r'$(a)$', r'$(b)$', r'$(c)$', r'$(d)$', r'$(e)$',  r'$(f)$', r'$(g)$', r'$(h)$', r'$(i)$',  r'$(j)$']

#-- Adjusting colorbar to different cases
if ( run_Ek == '1e7' ):
    rprate = 1
    if ( run_Pm == 'Pm1o44e-1' ):
        if ( l_tau_A ):
            #CBmax=5.e3#7.e3#8.e7#
            CBmax=6.e3 #CB Barrois and Aubert 2024
        else:
            CBmax=8.e7#
    elif ( run_Pm == 'Pm1o0e-1' ):
        if ( l_tau_A ):
            CBmax=1.e4#3.e8
        else:
            CBmax=3.e8
    else:
        if ( l_tau_A ):
            CBmax=6e3##1e3#3.e4#5.e3#8.e0
        else:
            CBmax=1.e8
elif ( run_Ek == '1e8' ):
    rprate = 2
    if ( l_tau_A ):
        CBmax=2.e4
    else:
        CBmax=4.e9
elif ( run_Ek == '6e9' ):
    rprate = 2
    if ( l_tau_A ):
        CBmax=2.e4
    else:
        CBmax=4.e9
elif ( run_Ek == '3e10' ):
    rprate = 2
    PLabels = [r'$(b)$', r'$(d)$', r'$(e)$', r'$(f)$', r'$(g)$',  r'$(h)$', r'$(i)$', r'$(j)$', r'$(k)$',  r'$(l)$']
    if ( l_tau_A ):
        #CBmax=7.e4
        CBmax=3.e4 #6.e4 #CB Barrois and Aubert 2024
    else:
        CBmax=5.e4#7.e4
else:
    rprate = 1; CBmax=1.e3

#-- .Ext for saving plots #pdf for Barrois and Aubert 2024
pext = 'png'
if ( l_spdf ):  pext = 'pdf'

if ( not l_old_plot ):
    #-- Adjusting trajs when using Imshow
    #-- Adimensionalising the axis because plt.imshow is not entirely satisfiying!
    tjplot = (len(tdim[::-1])-1)-(ttraj - ttraj[0])/ tdim.max()*len(tdim[::-1])
    rjplot = (rtraj - rtraj[0])/(rtraj - rtraj[0]).max()*(len(srja[::rprate])-1)
    if ( l_double_plot ):
        #-- Adimensionalising the axis because plt.imshow is not entirely satisfiying!
        tjplot2 = (len(tdim[::-1])-1)-(ttraj2 - ttraj2[0])/ tdim.max()*len(tdim[::-1])
        rjplot2 = (rtraj2 - rtraj2[0])/(rtraj2 - rtraj2[0]).max()*(len(srja[::rprate])-1)

#------------------------------------------------------------------------------
#%% Test compute directly Va (to avoid saving one file per run...)

if ( False ):
    from scipy.integrate import simps
    Bs = sint3D*B0r + cost3D*B0t #Bs=mmsint.*Br+mmcost.*Bt;
    Bs2 = Bs**2 #Bs2=Bs.^2;
    #
    tmp = simps((Bs2.mean(axis=0)*np.pi).T*np.sin(theta), theta)
    ABs2 = -simps(tmp*radius**2, radius)
    #
    #-- Va norm
    integs = Bs2.mean(axis=0)
    hs, rs, Bs2col = zavg(integs[..., ::-1], rrad, nsmax, colat=theta, minc=1, normed=True, save=False) 
    #
    # sc2=squeeze(mean(Bs2))';
    Bs2colja = np.zeros_like(omzcolja[0])
    for i in range(nsmaxja):
        Bs2colja[n_s,i] = np.sum(integs[n_s]*zslice[:,:,i]) #sum(sum(sc2.*squeeze(slice(:,:,ir))));

    Va = Bs2col/1.33 #Va=Bscol(ng:nr)./1.33; #1.33 --> Bs2.mean()? no...
    #
    #-- Va fast
    integs = Bs2.mean[1]
    hs, rs, Bs2col = zavg(integs[..., ::-1], rrad, nsmax, colat=theta, minc=1, normed=True, save=False) 
    #
    # sc2=squeeze(Bs2(1,:,:))';
    Bs2colja = np.zeros_like(omzcolja[0])
    for i in range(nsmaxja):
        Bs2colja[n_s,i] = np.sum(integs[n_s]*zslice[:,:,i]) #sum(sum(sc2.*squeeze(slice(:,:,ir))));

    Va_fast = Bs2col/1.33 #Va=Bscol(ng:nr)./1.33;
    #
    #-- Va slow
    integs = Bs2.mean[1]
    hs, rs, Bs2col = zavg(integs[..., ::-1], rrad, nsmax, colat=theta, minc=1, normed=True, save=False) 
    #
    # sc2=squeeze(Bs2(nphi//12,:,:))';
    Bs2colja = np.zeros_like(omzcolja[0])
    for i in range(nsmaxja):
        Bs2colja[n_s,i] = np.sum(integs[n_s]*zslice[:,:,i]) #sum(sum(sc2.*squeeze(slice(:,:,ir))));

    Va_slow = Bs2col/1.33 #Va=Bscol(ng:nr)./1.33;

#-- Not sure about the next few lines
if ( False ):
    from parobpy.core_properties import earth_radius
    Valfmean = np.loadtxt(dirva+'Va') # NOTE: Val defined with radius from parody: not practical to use sr or cylindrical radius
    hs = np.sqrt(radius[-1]**2 - (radius*np.sin(theta[ntheta//2]))**2) #np.sqrt(sr[0]**2 - sr[1:]**2)
    #Omega = 2.* np.pi* earth_radius / (23. * 60.*60. + 56*60. + 4.)  # 1 rotation per day
    #Omega/=Valfmean[:-1].mean()#/=Ek
    Omega=1./Lehnert
    k_wave = 32; m_wave = 3
    k_0 = (m_wave*Omega/hs[2:-2].mean()**2)**(1/3)
    #ks_0 = (m_wave*Omega*Valfmean[:-1].mean()/Valfmean[:-1]/hs[:-1]**2)**(1/3)
    #-- Alfven waves --> actually mixed between Aflven and MC waves
    omVa = Valfmean*k_wave + m_wave*Omega/(k_wave**2 * hs**2)
    omMiC= m_wave*Omega/(k_wave**2 * hs**2) + np.sqrt( (Valfmean*k_wave)**2 + (m_wave*Omega/(k_wave**2 * hs**2))**2 )
    #omMiC= m_wave*Omega/(k_wave**2 * hs**2) + np.sqrt( (Lehnert/Ek*k_wave)**2 + (m_wave*Omega/(k_wave**2 * hs**2))**2 )
    k_max = 71
    omMiT = np.zeros((k_max,len(Valfmean)),)
    for k_wave  in range(k_max):
        omMiT[k_wave] = m_wave*Omega/(k_wave**2 * hs**2) + np.sqrt( (Valfmean*k_wave)**2 + (m_wave*Omega/(k_wave**2 * hs**2))**2 )
        #omMiT[k_wave] = m_wave*Omega/(k_wave**2 * hs**2) + np.sqrt( (Lehnert/Ek*k_wave)**2 + (m_wave*Omega/(k_wave**2 * hs**2))**2 )

#------------------------------------------------------------------------------
#%% Main Plots

#-- Plot s-z-avg-profiles of different force balances as a function of time
if ( not l_old_plot ):
    import matplotlib.transforms as mtransforms
    zavgcolormap = cmo.curl#cmo.tarn_r#cmo.balance#'seismic'#
    plt.rcParams['text.usetex'] = True
    plt.rc('font', **{'family': 'serif'})#, 'serif': ['Computer Modern']})
    #plt.rc('text', usetex=True)
    tdim*=2. #Alfven time to physically redimensionalise
    sr*=2258.5 #Alfven time to physically redimensionalise
    y_lim = 14.02#7.07#tdim[-1]# 6.03# 1.76# 3.52# 5.3# 2.08# 
    n_tstart = 142-28#--> to remove the transient part: 130+12(otherwise shows 4 in t-axis) for Path cases
    l_plot_Va = False
    CNorm = np.amax(abs(Corcol))#1.0#
    CBmax = 1e0#4e3#
    ILab = 0
if ( rank==0 ):
    n_steps=-rprate
    ny_labels = 6
    ny = tdim.shape[0]
    y_steps = int(ny /(ny_labels - 1))
    y_post = np.arange(0,ny,y_steps)
    y_axis = np.round(tdim[::-y_steps], 4) # to get time going up
    nx_labels = 4
    nx = sr[::n_steps].shape[0]
    x_steps = int(nx/(nx_labels - 1))
    x_post = np.arange(0,nx,x_steps)
    x_axis = np.round(sr[::n_steps][::x_steps],1)
    #-- Zonal Velocity
    pmax = np.amax(abs(Vpcol[:,::n_steps]))/2.
    if ( l_old_plot ):
        plt.figure()
        plt.imshow(Vpcol[::-1,::n_steps], vmin=-pmax, vmax=pmax, cmap=cmo.balance)
        if ( l_plot_Va ): plt.plot(rjplot, tjplot, color=trajcolor, ls='-',marker='', lw='3.2', alpha=0.8) # Va_normal
        #if ( l_plot_Va ): plt.plot(rjplot2, tjplot2, color=trajcolor2, ls='-',marker='', lw='3.2', alpha=0.8) # Va_fast
        plt.title('z-Avg uphi Zonal')
        plt.ylabel('Time')
        plt.xlabel('Cylindrical radius')
        plt.xticks(x_post,x_axis)
        plt.yticks(y_post,y_axis)
        plt.colorbar(shrink=0.5, orientation='vertical')
        #-- Velocity
        if ( False ):
            pmax = np.amax(abs(Vpcol[:,::n_steps]))/2.
            plt.figure()
            plt.imshow((Vpcol*Vpcol*np.cos(3*np.pi/4.))[::-1,::n_steps], vmin=-pmax, vmax=pmax, cmap=cmo.balance)
            plt.title('z-Avg uphi')
            plt.ylabel('Time')
            plt.xlabel('Cylindrical radius')
            plt.xticks(x_post,x_axis)
            plt.yticks(y_post,y_axis)
            plt.colorbar(shrink=0.5, orientation='vertical')
    else:
        #llevels=np.linspace(-pmax,pmax,64)
        clevels=np.linspace(-pmax,pmax,7)
        fig = plt.figure(figsize=(11.5, 11.3))
        ax = plt.subplot(111)
        #cf = ax.contourf(sr,tdim,Vpcol,levels=llevels,extend='both',cmap=cmo.balance)#'seismic')#
        cf = ax.pcolormesh(sr,tdim,Vpcol,vmin=-pmax,vmax=pmax,antialiased=True,shading='gouraud',rasterized=True,cmap='seismic')#cmo.balance)#
        if ( l_plot_Va ): ax.plot(rtraj, ttraj, color=trajcolor, ls='-',marker='', lw='6.4', alpha=0.9) # Va_mean
        plt.xlabel(r'Cylindrical radius', fontsize=36)
        plt.ylabel(r'Time', fontsize=36)
        plt.ylim(tdim[n_tstart],y_lim)#tdim[-1])#1.76)#3.52)#5.3)#2.08)#
        plt.gca().xaxis.set_tick_params(labelsize=32)
        plt.gca().yaxis.set_tick_params(labelsize=32)
        cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.05, orientation='vertical', ticks=clevels, format=r'${x:.1f}$')
        cb.ax.tick_params(labelsize=32)
        transAx = mtransforms.ScaledTranslation(8.35+10/72, -45/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, PLabels[0+ILab], transform=ax.transAxes + transAx,
                fontsize=36, va='bottom', fontfamily='serif')
        plt.tight_layout()
        #plt.show()
    #-- Velocity
    pmax = np.amax(abs(upcol[:,::n_steps]))/2.
    plt.figure()
    plt.imshow(upcol[::-1,::n_steps], vmin=-pmax, vmax=pmax, cmap=cmo.balance)
    plt.title('z-Avg uphi')
    plt.ylabel('Time')
    plt.xlabel('Cylindrical radius')
    plt.xticks(x_post,x_axis)
    plt.yticks(y_post,y_axis)
    plt.colorbar(shrink=0.5, orientation='vertical')
    #-- d Velocity / dt
    pmax = np.amax(abs(dVpcol[:,::n_steps]))/2.
    plt.figure()
    plt.imshow(dVpcol[::-1,::n_steps], vmin=-pmax, vmax=pmax, cmap=cmo.balance)
    plt.title('z-Avg d uphi / dt')
    plt.ylabel('Time')
    plt.xlabel('Cylindrical radius')
    plt.xticks(x_post,x_axis)
    plt.yticks(y_post,y_axis)
    plt.colorbar(shrink=0.5, orientation='vertical')
    #-- Vorticity
    pmax = np.amax(abs(omzcol[:,::n_steps]))/2.
    plt.figure()
    plt.imshow(omzcol[::-1,::n_steps], vmin=-pmax, vmax=pmax, cmap=cmo.tarn_r)
    plt.title('z-Avg $\omega_z$')
    plt.ylabel('Time')
    plt.xlabel('Cylindrical radius')
    plt.xticks(x_post,x_axis)
    plt.yticks(y_post,y_axis)
    plt.colorbar(shrink=0.5, orientation='vertical')
    #-- Inertia = d Vorticity / dt
    if ( l_fix_CB ):
        pmax = CBmax
    else:
        pmax = np.amax(abs(domzcol[:,::n_steps]))#/250.#
    if ( l_old_plot ):
        plt.figure()
        plt.imshow(domzcol[::-1,::n_steps], vmin=-pmax, vmax=pmax, cmap='seismic')#cmo.tarn_r)#
        if ( l_plot_Va ): plt.plot(rjplot2, tjplot2, color=trajcolor2, ls='-',marker='', lw='3.4', alpha=0.9) # Va_fast
        plt.title('z-Avg d $\omega_z$ / dt')
        plt.ylabel('Time')
        plt.xlabel('Cylindrical radius')
        plt.xticks(x_post,x_axis)
        plt.yticks(y_post,y_axis)
        plt.colorbar(shrink=0.5, orientation='vertical')
    else:
        #llevels=np.linspace(-pmax,pmax,64)
        clevels=np.linspace(-pmax,pmax,7)
        if ( l_spdf ):
            #fig = plt.figure(figsize=(11.9, 11.3))#figsize=(12.1, 11.3))--> for 3e10
            fig = plt.figure(figsize=(11.08, 11.3)) # NOTE: Remove cb for Fig. Barrois and Aubert 2024 (or just go for subplot at some point)
        else:
            #fig = plt.figure(figsize=(11.9, 11.3))#figsize=(12.1, 11.3))--> for 3e10
            fig = plt.figure(figsize=(11.9, 11.3))
        ax = plt.subplot(111)
        #cf = ax.contourf(sr,tdim,domzcol,levels=llevels,extend='both',cmap=cmo.balance)#'seismic')#
        cf = ax.pcolormesh(sr,tdim,domzcol/CNorm,vmin=-pmax,vmax=pmax,antialiased=True,shading='gouraud',rasterized=True,cmap=zavgcolormap)#
        #cf = ax.imshow(domzcol[::-1,::n_steps], vmin=-pmax, vmax=pmax, cmap='seismic')#cmo.balance)#
        if ( l_plot_Va ):
            ax.plot(rtraj, ttraj, color=trajcolor, ls='-',marker='', lw='6.4', alpha=0.9) # Va_mean
            ax.plot(rtraj2, ttraj2, color=trajcolor2, ls='-.',marker='', lw='6.4', alpha=0.9) # Va_fast
        plt.xlabel(r'Cylindrical radius, ($km$)', fontsize=36)
        plt.ylabel(r'Time, ($y$)', fontsize=36)
        plt.ylim(tdim[n_tstart],y_lim)#tdim[-1])#
        plt.gca().xaxis.set_tick_params(labelsize=32)
        plt.gca().yaxis.set_tick_params(labelsize=32)
        cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.05, orientation='vertical', ticks=clevels)#, format=r'${x:.1f}$')#
        cb.ax.tick_params(labelsize=32)
        transAx = mtransforms.ScaledTranslation(8.65+10/72, -45/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, PLabels[1+ILab], transform=ax.transAxes + transAx,
                fontsize=36, va='bottom', fontfamily='serif')
        plt.tight_layout()
        if ( l_spdf ):  cb.remove()
    #-- Lorentz
    if ( l_fix_CB ):
        pmax = CBmax
    else:
        pmax = np.amax(abs(Lorcol[:,::n_steps]))#/250.#
    if ( l_old_plot ):
        plt.figure()
        plt.imshow(Lorcol[::-1,::n_steps], vmin=-pmax, vmax=pmax, cmap='seismic')#'PuOr_r')#
        if ( l_plot_Va ): plt.plot(rjplot2, tjplot2, color=trajcolor2, ls='-',marker='', lw='3.4', alpha=0.9) # Va_fast
        plt.title('z-Avg Lorentz')
        plt.ylabel('Time')
        plt.xlabel('Cylindrical radius')
        plt.xticks(x_post,x_axis)
        plt.yticks(y_post,y_axis)
        plt.colorbar(shrink=0.5, orientation='vertical')
    else:
        #llevels=np.linspace(-pmax,pmax,64)
        clevels=np.linspace(-pmax,pmax,7)
        if ( l_spdf ):
            fig = plt.figure(figsize=(11.08, 11.3)) # NOTE: Remove cb for Fig. Barrois and Aubert 2024 (or just go for subplot at some point)
        else:
            fig = plt.figure(figsize=(11.9, 11.3))
        ax = plt.subplot(111)
        #cf = ax.contourf(sr,tdim,Lorcol,levels=llevels,extend='both',cmap=zavgcolormap)#
        cf = ax.pcolormesh(sr,tdim,Lorcol/CNorm,vmin=-pmax,vmax=pmax,antialiased=True,shading='gouraud',rasterized=True,cmap=zavgcolormap)#
        #cf = ax.imshow(Lorcol[::-1,::n_steps], vmin=-pmax, vmax=pmax, cmap='seismic')#cmo.balance)#
        if ( l_plot_Va ):
            ax.plot(rtraj, ttraj, color=trajcolor, ls='-',marker='', lw='6.4', alpha=0.9) # Va_mean
            ax.plot(rtraj2, ttraj2, color=trajcolor2, ls='-.',marker='', lw='6.4', alpha=0.9) # Va_fast
        plt.xlabel(r'Cylindrical radius, ($km$)', fontsize=36)
        plt.ylabel(r'Time, ($y$)', fontsize=36)
        plt.ylim(tdim[n_tstart],y_lim)#tdim[-1])#
        plt.gca().xaxis.set_tick_params(labelsize=32)
        plt.gca().yaxis.set_tick_params(labelsize=32)
        cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.05, orientation='vertical', ticks=clevels)#, format=r'${x:.1f}$')#
        cb.ax.tick_params(labelsize=32)
        transAx = mtransforms.ScaledTranslation(8.65+10/72, -45/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, PLabels[2+ILab], transform=ax.transAxes + transAx,
                fontsize=36, va='bottom', fontfamily='serif')
        plt.tight_layout()
        if ( l_spdf ):  cb.remove()
    #-- Coriolis
    if ( l_fix_CB ):
        pmax = CBmax
    else:
        pmax = np.amax(abs(Corcol[:,::n_steps]))#/25.#
    if ( l_old_plot ):
        plt.figure()
        plt.imshow(Corcol[::-1,::n_steps], vmin=-pmax, vmax=pmax, cmap='seismic')#cmo.balance)#
        if ( l_plot_Va ): plt.plot(rjplot2, tjplot2, color=trajcolor2, ls='-',marker='', lw='3.4', alpha=0.9) # Va_fast
        plt.title('z-Avg Coriolis')
        plt.ylabel('Time')
        plt.xlabel('Cylindrical radius')
        plt.xticks(x_post,x_axis)
        plt.yticks(y_post,y_axis)
        plt.colorbar(shrink=0.5, orientation='vertical')
    else:
        #llevels=np.linspace(-pmax,pmax,64)
        clevels=np.linspace(-pmax,pmax,7)
        fig = plt.figure(figsize=(11.9, 11.3))
        ax = plt.subplot(111)
        #cf = ax.contourf(sr,tdim,Corcol,levels=llevels,extend='both',cmap=cmo.balance)#'seismic')#
        cf = ax.pcolormesh(sr,tdim,Corcol/CNorm,vmin=-pmax,vmax=pmax,antialiased=True,shading='gouraud',rasterized=True,cmap=zavgcolormap)#
        #cf = ax.imshow(Corcol[::-1,::n_steps], vmin=-pmax, vmax=pmax, cmap='seismic')#cmo.balance)#
        if ( l_plot_Va ):
            ax.plot(rtraj, ttraj, color=trajcolor, ls='-',marker='', lw='6.4', alpha=0.9) # Va_mean
            ax.plot(rtraj2, ttraj2, color=trajcolor2, ls='-.',marker='', lw='6.4', alpha=0.9) # Va_fast
        plt.xlabel(r'Cylindrical radius, ($km$)', fontsize=36)
        plt.ylabel(r'Time, ($y$)', fontsize=36)
        plt.ylim(tdim[n_tstart],y_lim)#tdim[-1])#
        plt.gca().xaxis.set_tick_params(labelsize=32)
        plt.gca().yaxis.set_tick_params(labelsize=32)
        cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.05, orientation='vertical', ticks=clevels)#, format=r'${x:.1f}$')#
        cb.ax.tick_params(labelsize=32)
        transAx = mtransforms.ScaledTranslation(8.65+10/72, -45/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, PLabels[3+ILab], transform=ax.transAxes + transAx,
                fontsize=36, va='bottom', fontfamily='serif')
        plt.tight_layout()
        #plt.show()
    #-- AGeos
    pmax = np.amax(abs(AgeosCorcol[:,::n_steps]))/2.
    plt.figure()
    plt.imshow(AgeosCorcol[::-1,::n_steps], vmin=-pmax, vmax=pmax, cmap='RdYlBu_r')
    plt.title('z-Avg AgeosCorcol')
    plt.ylabel('Time')
    plt.xlabel('Cylindrical radius')
    plt.xticks(x_post,x_axis)
    plt.yticks(y_post,y_axis)
    plt.colorbar(shrink=0.5, orientation='vertical')
    #-- Residuals
    if ( l_fix_CB ):
        pmax = CBmax
    else:
        pmax = np.amax(abs(Residuals[:,::n_steps]))#/250.#
    if ( l_old_plot ):
        plt.figure()
        plt.imshow(Residuals[::-1,::n_steps], vmin=-pmax, vmax=pmax, cmap='seismic')#'RdYlBu_r')#
        plt.title('z-Avg Lorentz - Coriolis - d $\omega_z$ / dt')
        plt.ylabel('Time')
        plt.xlabel('Cylindrical radius')
        plt.xticks(x_post,x_axis)
        plt.yticks(y_post,y_axis)
        plt.colorbar(shrink=0.5, orientation='vertical')
    else:
        #llevels=np.linspace(-pmax,pmax,64)
        clevels=np.linspace(-pmax,pmax,7)
        fig = plt.figure(figsize=(11.9, 11.3))
        ax = plt.subplot(111)
        #cf = ax.contourf(sr,tdim,Residuals,levels=llevels,extend='both',cmap=cmo.balance)#'seismic')#
        cf = ax.pcolormesh(sr,tdim,Residuals,vmin=-pmax,vmax=pmax,antialiased=True,shading='gouraud',rasterized=True,cmap=zavgcolormap)#
        #cf = ax.imshow(Residuals[::-1,::n_steps], vmin=-pmax, vmax=pmax, cmap='seismic')#cmo.balance)#
        plt.xlabel(r'Cylindrical radius', fontsize=36)
        plt.ylabel(r'Time', fontsize=36)
        plt.ylim(tdim[n_tstart],y_lim)#tdim[-1])#
        plt.gca().xaxis.set_tick_params(labelsize=32)
        plt.gca().yaxis.set_tick_params(labelsize=32)
        cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.05, orientation='vertical', ticks=clevels)#, format=r'${x:.1f}$')#
        cb.ax.tick_params(labelsize=32)
        transAx = mtransforms.ScaledTranslation(8.65+10/72, -45/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, PLabels[0+ILab], transform=ax.transAxes + transAx,
                fontsize=36, va='bottom', fontfamily='serif')
        plt.tight_layout()
    if ( not l_save ): plt.show()

#-- Save columnar phi-force balance plots
if ( l_save and rank==0 ):
    size_plot = (10.2, 10.8)
    plt.figure(1)
    if ( l_old_plot ):
        plt.gcf().set_size_inches(size_plot, forward=False)
        plt.savefig(savePlot+'Radial_time-phi-Balance_VpZon-Col_Old.png')
    else:
        plt.savefig(savePlot+'Radial_time-phi-Balance_VpZon-Col.'+pext)
    plt.figure(2)
    plt.gcf().set_size_inches(size_plot, forward=False)
    plt.savefig(savePlot+'Radial_time-phi-Balance_uphi-Col.png')
    plt.figure(3)
    plt.gcf().set_size_inches(size_plot, forward=False)
    plt.savefig(savePlot+'Radial_time-phi-Balance_duphidt-Col.png')
    plt.figure(4)
    plt.gcf().set_size_inches(size_plot, forward=False)
    plt.savefig(savePlot+'Radial_time-phi-Balance_omz-Col.png')
    plt.figure(5)
    if ( l_old_plot ):
        plt.gcf().set_size_inches(size_plot, forward=False)
        plt.savefig(savePlot+'Radial_time-phi-Balance_domzdt-Col_Old.png')
    else:
        plt.savefig(savePlot+'Radial_time-phi-Balance_domzdt-Col.'+pext)
    plt.figure(6)
    if ( l_old_plot ):
        plt.gcf().set_size_inches(size_plot, forward=False)
        plt.savefig(savePlot+'Radial_time-phi-Balance_Lorz-Col_Old.png')
    else:
        plt.savefig(savePlot+'Radial_time-phi-Balance_Lorz-Col.'+pext)
    plt.figure(7)
    if ( l_old_plot ):
        plt.gcf().set_size_inches(size_plot, forward=False)
        plt.savefig(savePlot+'Radial_time-phi-Balance_Corz-Col_Old.png')
    else:
        plt.savefig(savePlot+'Radial_time-phi-Balance_Corz-Col.'+pext)
    plt.figure(8)
    plt.gcf().set_size_inches(size_plot, forward=False)
    plt.savefig(savePlot+'Radial_time-phi-Balance_AgeosCorz-Col.png')
    plt.figure(9)
    if ( l_old_plot ):
        plt.gcf().set_size_inches(size_plot, forward=False)
        plt.savefig(savePlot+'Radial_time-phi-Balance_Residuals-Col_Old.png')
    else:
        plt.savefig(savePlot+'Radial_time-phi-Balance_Residuals-Col.'+pext)
    plt.close('all')

#------------------------------------------------------------------------------
#%% Secondary Plot

#-- Testing filter z-avg
if ( rank==0 ):
    plt.rcParams['text.usetex'] = False
    n_steps=rprate
    nx_labels = 4
    nx = srja[::n_steps].shape[0]
    x_steps = int(nx/(nx_labels - 1))
    x_post = np.arange(0,nx,x_steps)
    x_axis = np.round(srja[::n_steps][::x_steps],1)
    #-- d Vorticity / dt
    if ( l_fix_CB ):
        pmax = CBmax
    else:
        pmax = np.amax(abs(domzcolja[:,::n_steps]))#/250.#
    plt.figure()
    plt.imshow(domzcolja[::-1,::n_steps], vmin=-pmax, vmax=pmax, cmap='seismic')#cmo.tarn_r)#
    if ( l_plot_Va ): plt.plot(rjplot, tjplot, color=trajcolor, ls='-',marker='', lw='3.2', alpha=0.8) # Va_normal
    if ( l_plot_Va ): plt.plot(rjplot2, tjplot2, color=trajcolor2, ls='-',marker='', lw='3.2', alpha=0.8) # Va_fast
    plt.title('z-Avg d $\omega_z$ / dt')
    plt.ylabel('Time')
    plt.xlabel('Cylindrical radius')
    plt.xticks(x_post,x_axis)
    plt.yticks(y_post,y_axis)
    plt.colorbar(shrink=0.5, orientation='vertical')
    #-- Lorentz
    if ( l_fix_CB ):
        pmax = CBmax
    else:
        pmax = np.amax(abs(Lorcolja[:,::n_steps]))#/250.#
    plt.figure()
    plt.imshow(Lorcolja[::-1,::n_steps], vmin=-pmax, vmax=pmax, cmap='seismic')#'PuOr_r')#
    plt.title('z-Avg Lorentz')
    plt.ylabel('Time')
    plt.xlabel('Cylindrical radius')
    plt.xticks(x_post,x_axis)
    plt.yticks(y_post,y_axis)
    plt.colorbar(shrink=0.5, orientation='vertical')
    #-- Coriolis
    if ( l_fix_CB ):
        pmax = CBmax
    else:
        pmax = np.amax(abs(Corcolja[:,::n_steps]))#/25.#
    plt.figure()
    plt.imshow(Corcolja[::-1,::n_steps], vmin=-pmax, vmax=pmax, cmap='seismic')#cmo.balance)#
    plt.title('z-Avg Coriolis')
    plt.ylabel('Time')
    plt.xlabel('Cylindrical radius')
    plt.xticks(x_post,x_axis)
    plt.yticks(y_post,y_axis)
    plt.colorbar(shrink=0.5, orientation='vertical')
    #-- Residuals
    if ( l_fix_CB ):
        pmax = CBmax
    else:
        pmax = np.amax(abs(Residualsja[:,::n_steps]))#/250.#
    plt.figure()
    plt.imshow(Residualsja[::-1,::n_steps], vmin=-pmax, vmax=pmax, cmap='seismic')#'RdYlBu_r')#
    plt.title('z-Avg Lorentz - Coriolis - d $\omega_z$ / dt')
    plt.ylabel('Time')
    plt.xlabel('Cylindrical radius')
    plt.xticks(x_post,x_axis)
    plt.yticks(y_post,y_axis)
    plt.colorbar(shrink=0.5, orientation='vertical')
    if ( not l_save ): plt.show()
    #
    #-- Save columnar phi-force balance produced by JA plots
    if ( l_save and rank==0 ):
        size_plot = (10.2, 10.8)
        plt.figure(1)
        plt.gcf().set_size_inches(size_plot, forward=False)
        plt.savefig(savePlot+'Radial_time-phi-Balance_domzdt-JA.png')
        plt.figure(2)
        plt.gcf().set_size_inches(size_plot, forward=False)
        plt.savefig(savePlot+'Radial_time-phi-Balance_Lorentz-JA.png')
        plt.figure(3)
        plt.gcf().set_size_inches(size_plot, forward=False)
        plt.savefig(savePlot+'Radial_time-phi-Balance_Coriolis-JA.png')
        plt.figure(4)
        plt.gcf().set_size_inches(size_plot, forward=False)
        plt.savefig(savePlot+'Radial_time-phi-Balance_Residuals-JA.png')
        plt.close('all')

plt.close('all')

#-- End Script
