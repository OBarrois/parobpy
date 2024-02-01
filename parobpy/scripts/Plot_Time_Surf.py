#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 2023
@author: obarrois

Rewriting of PARODY-JA4.56-Base/Matlab Matlab file '/makemove_eq.m' to only extract equatorial sections.
Loads graphics files and save all equatorial plane frames with a sampling rate t_sampling.

Then one can construct a plot with several time-frames of equatorial planes for fields of interest.
"""

#import os
#import sys
from vizuals import hammer2cart, radialContour#, equatContour#, merContour
#from parobpy.core_properties import icb_radius, cmb_radius
from parobpy.load_parody import parodyload, surfaceload, list_St_files, load_dimensionless, load_basefield#, list_Gt_files#, load_parody_serie, load_surface_serie
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import cmocean.cm as cmo
#import shtns

#-- Functions
#matplotlib.use('Agg')  # backend for no display if needed

#----------------------------------------------------------------------------%%
#-- INPUT PARAMETERS

#-- Lehnert number, \lambda = B/{sqrt(rho mu) * Omega*d}.
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '3e7', 'Pm0o25', 1.46e-3, 1.0, 'b3', '0.000280800' # 3e-7 S=1214.1
run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1', 1.1e-3, 1.0, 'b4', '0.000040320' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'PathBase', 1.1e-3, 1.0, 'b4-p', '0.000008755' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e8', '', 5.4e-4, 1.0, 'b4o5', '0.000024896' # 1e-8
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '6e9', 'Pm0o36e-1', 5.53e-4, 1.0, 'b4o63', '0.000015560' # 6.3e-9
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '3e10', '', 2.6e-4, 1.0, 'b5', '0.000001456' # 3e-10

#saveDir = '/gpfs/users/obarrois/Parodys/Outputs/Data/' # if you use python directly from cluster
#directory = '/gpfs/users/obarrois/Work/Waves1e7/' # if you use python directly from cluster
#directory = '/Users/obarrois/Desktop/dfgnas3/Waves/Data_3e10/' # run at 3e-10 on jaubert disk
directory = '/Users/obarrois/Desktop/stella/Work/Waves'+run_Ek+'/'+run_Pm+'/' # if you mount stella on own desk

l_phi_avg = False # remove phi avg from fields
l_back_b0 = False # add back the background magnetic field b0 --> should be False because we are interested in the perturbations (we only see B0 otherwise)

#l_sample = True # sample Gt time files instead of computing for every t=... (advised: could be really slow otherwise)
tstart = 0 # time rank of the first Gt file read. Should be <= ntime and can be mofidify to scan != samples with t_srate
t_srate = 1 # sampling rate of Gt files if needed. Should be <= ntime

l_check_all = False # display all check plot figures
l_save = 1 # save figures?
savePlot = '/Users/obarrois/Desktop/Parodys/Outputs/Waves'+run_Ek+'/' # path to save files

#------------------------------------------------------------------------------
#%% Load data

Gt_file = 'Gt={}.{}'.format(timestamp,run_ID)
filename = directory + Gt_file

(version, time, DeltaU, Coriolis, Lorentz, Buoyancy, ForcingU,
            DeltaT, ForcingT, DeltaB, ForcingB, Ek, Ra, Pm, Pr,
            nr, ntheta, nphi, azsym, radius, theta, phi, _, _, _,
            _, _, _, _) = parodyload(filename)

if ( run_Pm == 'PathBase' ):
    basedir = '/Users/obarrois/Desktop/stella/Work/' # basefield copied on stella
    basename = basedir+'basefield_path.mat' # if you use python directly from cluster
else:
    basedir = '/Users/obarrois/Desktop/stella/Work/Waves'+run_Ek+'/' # basefield copied on stella
    basename = basedir+'basefield.mat' # if you use python directly from cluster
B0r, B0t, B0p, _, _, _ = load_basefield(basename,nr,ntheta,nphi)

NR, Ek, Ra, Pr, Pm, minc, mcl, fi, rf = load_dimensionless(run_ID, directory)

phir = phi.copy()
nphir = nphi
if ( minc !=1 ):
    from parobpy.parob_lib import symmetrize
    nphir = nphi*minc##+1 NOTE: last point is spurious in MagIC.symmetrize
    phir = np.linspace(phi[0],3*phi[-1], nphir)
    B0r = symmetrize(B0r,ms=minc)#; j0r = symmetrize(j0r,ms=minc)
    B0t = symmetrize(B0t,ms=minc)#; j0t = symmetrize(j0t,ms=minc)
    B0p = symmetrize(B0p,ms=minc)#; j0p = symmetrize(j0p,ms=minc)
amp_B = np.sqrt(amp_B) #-- sqrt(amp_b) because of parody unit in B_rms
B0r*=amp_B; B0t*=amp_B; B0p*=amp_B
#j0r*=amp_B; j0t*=amp_B; j0p*=amp_B

#------------------------------------------------------------------------------
#%% Read, process and save chosen data

Gt_file_l = list_St_files(run_ID,directory) # Find all St_no in folder
n_samples = len(Gt_file_l[tstart::t_srate])

timett = np.zeros((n_samples),)
brt = np.zeros((n_samples, nphir, ntheta),)
dbrt = np.zeros_like(brt)
upt = np.zeros_like(brt)
upft = np.zeros_like(brt)

#-- Selecting data for plotting if needed
nplots = 6
tpk = int(len(timett[:500])/nplots)
#t_pick = np.arange(1,len(timett[:500]),tpk)
nplots = 4
t_pick = np.array([99, 199, 299, 399, 499])

n_t=-1; n_s=-1
#-- Start loop
for file in np.array(Gt_file_l)[t_pick]:
#for file in Gt_file_l[tstart::t_srate]:#[249:250]:#
    #n_s+=1
    n_t+=1; n_s = t_pick[n_t]
    #------------------------------------------------------------------------------
    #-- loading pointtime data
    print('Loading {} (({}/{})/{})'.format(file, n_s+1, n_samples, len(Gt_file_l)))
    filename = '{}/{}'.format(directory,file)
    (_, time, _, _, _, _, _, 
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, _, up,
        Br, dtBr) = surfaceload(filename)
    timett[n_s] = time
    #
    #-- resymmetrize the data if minc > 1
    if ( minc !=1 ): # no need to re-symmetrize if plotting MerContour of phi-avg (or even slice)
        #from parobpy.parob_lib import symmetrize
        #phi = np.linspace(phir[0],2*np.pi-phir[0], nphi) #--> already done for B0r normally
        print('Re-symmetrizing the data for file n°{}/{}, t={}'.format(n_s+1, n_samples, timett[n_s]))
        dtBr = symmetrize(dtBr,ms=minc)
        Br = symmetrize(Br,ms=minc)
        #ut = symmetrize(ut,ms=minc)
        up = symmetrize(up,ms=minc)
    #
    #-- Vp = up_m=0 = \overline{up} = axisymmetric or zonal azimuthal velocity
    Vp = up.mean(axis=0)
    #-- up_fluct = up - up_phi-Avg, if needed
    upfluct = np.zeros_like(up)
    for n_p in range(nphir):
        #print('Removing mean field for phi {}/{}'.format(n_p+1, nphir))
        upfluct[n_p] = up[n_p] - Vp
    #
    #-- Select equatorial plane and save the field
    brt[n_s] = Br
    dbrt[n_s] = dtBr
    upt[n_s] = up
    upft[n_s] = upfluct

#------------------------------------------------------------------------------
#%% Additional processing if needed

#-- Adding back the background magnetic field B0 to B
if ( l_back_b0 ):
    print('Adding magnetic background field for sample')
    for n_s in range(n_samples):
        brt[n_s]+= B0r[:,:,nr-1]
#
#-- Removing the phi-avg for better waves spooting
if ( l_phi_avg ):
    brpm = brt.mean(axis=1)
    dtbrpm = dbrt.mean(axis=0)
    uppm = upt.mean(axis=1)
    print('Removing phi-Avg field')
    for n_p in range(nphir):
        brt[:,n_p]-= brpm
        dbrt[:,n_p]-= dtbrpm
        upt[:,n_p]-= uppm

#-- Redim time
#Lehnert = np.sqrt(1.33**2*(Ek/Pm))
tredim = 1./(Ek/Lehnert)
tdimt = timett*tredim#/(Ek/Lehnert)

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

#-- Labels for plots in Article #e.g. Barrois and Aubert 2024
PLabels = [r'$(a)$', r'$(a)$', r'$(b)$', r'$(c)$', r'$(d)$', r'$(e)$',  r'$(f)$', r'$(g)$', r'$(h)$', r'$(i)$',  r'$(j)$']

#------------------------------------------------------------------------------
#%% Main plot section

#-- Meshgrids and constants
eps = 1.e-3
phip = np.linspace(-np.pi, np.pi, nphir)
thetap = np.linspace(np.pi/2, -np.pi/2, ntheta)
pphi, ttheta = np.mgrid[-np.pi:np.pi:nphir*1j, np.pi/2.:-np.pi/2.:ntheta*1j]
lon2 = pphi*180./np.pi
lat2 = ttheta*180./np.pi

circles = np.r_[-60., -30., 0., 30., 60.]
delon = 60.
meridians = np.arange(-180+delon, 180, delon)

xx, yy = hammer2cart(ttheta, pphi)

#-- Field to plot
fieldplot = brt.copy()#upft.copy()#upt.copy()#dbrt.copy()#

#-- Fig. Barrois and Aubert 2024
plt.rcParams['text.usetex'] = True
#-- Aggregated equatContour plot phi-vs-s at different times
k=-1
#fig = plt.figure(figsize=(26, 3.6))#figsize=(25.2, 4.6))
fig = plt.figure(figsize=(25.55, 3.85)) #NOTE: cannot unpack non-iterable Figure object
#fig, axs = plt.subplots(1, nplots+1, figsize=(21, 7.4))#figsize=(22.1, 7.4))#, sharex=True)#, layout='constrained')
for n_t in t_pick:
    cmax = 2.e-4#abs(fieldplot[n_t,:,:]).max()*0.92
    llevels=np.linspace(-cmax,cmax,64)
    clevels=np.linspace(-cmax,cmax,4)
    k+=1
    ax = plt.subplot2grid((1,nplots+1), (0,k))
    #axs[k] = plt.subplot(1,nplots+1,k+1)
    #ax = axs[k]
    cf = ax.contourf(xx,yy,fieldplot[n_t,:,:],levels=llevels,extend='both',cmap='PuOr_r')#'seismic')#cmo.balance)#
    #cf = radialContour(fieldplot[n_t,:,:],rad=1,cm='PuOr_r')#levels=llevels,extend='both',cmap='seismic')#'PuOr_r')#cmo.balance)#
    for lat0 in circles:
        x0, y0 = hammer2cart(lat0*np.pi/180., phip)
        ax.plot(x0, y0, 'k:', lw=0.7)
    for lon0 in meridians:
        x0, y0 = hammer2cart(thetap, lon0*np.pi/180.)
        ax.plot(x0, y0, 'k:', lw=0.7)
    xout, yout = hammer2cart(thetap, -np.pi-eps)
    xin, yin = hammer2cart(thetap, np.pi+eps)
    ax.plot(xout, yout, 'k-')
    ax.plot(xin, yin, 'k-')
    ax.axis('off')
    plt.gca().xaxis.set_tick_params(labelsize=32)
    plt.gca().yaxis.set_tick_params(labelsize=32)
    #plt.title(r'time $t = $'+str(np.round(tdimt[n_t],2)), fontsize=32)# NOTE: Really better to use "format", e.g.: $print('{:1.2f}'.format(123.456))
    plt.title(r'time $t = {:.2f}$'.format(tdimt[n_t]), fontsize=32)
    cb = fig.colorbar(cf, ax=ax, fraction=0.08, pad=0.085, orientation='horizontal', ticks=clevels, format=r'${x:.0e}$')#1f}')
    cb.ax.tick_params(labelsize=21)#30)#
    if ( k==0 ):
        transAx = mtransforms.ScaledTranslation(+10/72, -5/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, PLabels[3], transform=ax.transAxes + transAx,
                fontsize=36, va='bottom', fontfamily='serif')
#
plt.tight_layout()
#cb = fig.colorbar(cf, ax=axs[:], fraction=0.02, pad=0.015, orientation='horizontal', ticks=clevels, format=r'${x:.0f}$')
#cb.ax.tick_params(labelsize=32)
if ( not l_save ):  plt.show()

#-- Save True Fields analysis
if ( l_save ):
    plt.savefig(savePlot+'Azimuth-Theta-SurfContour_br.png')
    plt.close('all')

#-- End Script
