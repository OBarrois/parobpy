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
#from vizuals import equatContour#, merContour, radialContour
#from parobpy.core_properties import icb_radius, cmb_radius
from parobpy.load_parody import parodyload, load_dimensionless, load_basefield, list_Gt_files#, surfaceload, list_St_files, load_parody_serie, load_surface_serie
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

l_resym = False # re-symmetrizing the data if minc !=1
l_phi_avg = False # remove phi avg from fields
l_back_b0 = False # add back the background magnetic field b0 --> should be False because we are interested in the perturbations (we only see B0 otherwise)

#l_sample = True # sample Gt time files instead of computing for every t=... (advised: could be really slow otherwise)
tstart = 0 # time rank of the first Gt file read. Should be <= ntime and can be mofidify to scan != samples with t_srate
t_srate = 1 # sampling rate of Gt files if needed. Should be <= ntime
l_rnorm = False # Normalise fields by the maximum at each r-radius or not
l_deminc = False # Plot fields including azimuthal symmetries or not (if minc !=1)

l_plot_fluct = True # produce and save u_Tot or u_fluct fields
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
if ( l_resym and minc !=1 ):
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

Gt_file_l = list_Gt_files(run_ID,directory) # Find all Gt_no in folder
n_samples = len(Gt_file_l[tstart::t_srate])

timett = np.zeros((n_samples),)
brt = np.zeros((n_samples, nphir, nr),)
urt = np.zeros_like(brt)
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
        _, _, _, _, _, _, _, ur, _, up,
        Br, _, Bp, _) = parodyload(filename)
    timett[n_s] = time
    #
    #-- resymmetrize the data if minc > 1
    if ( l_resym and minc !=1 ): # no need to re-symmetrize if plotting MerContour of phi-avg (or even slice)
        #from parobpy.parob_lib import symmetrize
        #phi = np.linspace(phir[0],2*np.pi-phir[0], nphi) #--> already done for B0r normally
        print('Re-symmetrizing the data for file n°{}/{}, t={}'.format(n_s+1, n_samples, timett[n_s]))
        #Bt = symmetrize(Bt,ms=minc)
        Bp = symmetrize(Bp,ms=minc)
        ur = symmetrize(ur,ms=minc)
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
    brt[n_s] = Br[:,ntheta//2,:]
    urt[n_s] = ur[:,ntheta//2,:]
    upt[n_s] = up[:,ntheta//2,:]
    upft[n_s] = upfluct[:,ntheta//2,:]

#------------------------------------------------------------------------------
#%% Additional processing if needed

#-- Adding back the background magnetic field B0 to B
if ( l_back_b0 ):
    print('Adding magnetic background field for sample')
    for n_s in range(n_samples):
        brt[n_s]+= B0r[:,ntheta//2,:]
#
#-- Removing the phi-avg for better waves spooting
if ( l_phi_avg ):
    brpm = brt.mean(axis=1)
    urpm = urt.mean(axis=1)
    uppm = upt.mean(axis=1)
    print('Removing phi-Avg field')
    for n_p in range(nphir):
        brt[:,n_p]-= brpm
        urt[:,n_p]-= urpm
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
if ( l_deminc and l_resym ):
    rr, pphi = np.meshgrid(radius, phir)
else:
    #npminc = nphi#//minc
    rr, pphi = np.meshgrid(radius, phi)
xx = rr*np.cos(pphi)
yy = rr*np.sin(pphi)
#-- Adjusting colorsfor trajs #CB Barrois and Aubert 2024
Atrajcolor=WA_BRoc_Dark#'#32CD32' # Lime # '#99CC32' # Jaune-Vert # '#FF7F00' # Orange
Ftrajcolor= WA_BRoc_Dark#Personal_Cyan#'#FF7F00' # Orange # '#32CD32' # Lime # '#99CC32' # Jaune-Vert # 
Strajcolor= 'blue'#FF7F00' # Orange # '#32CD32' # Lime # '#99CC32' # Jaune-Vert # 

#-- Field to plot
if ( l_plot_fluct ):
    fieldplot = upft.copy()#upt.copy()#bft.copy()#
    cmax = 4.#upft# 12.#upt# 
    plab = PLabels[2]
else:
    fieldplot = upt.copy()#
    cmax = 12.#upt#
    plab = PLabels[1]

#-- Fig. Barrois and Aubert 2024
plt.rcParams['text.usetex'] = True
#-- Aggregated equatContour plot phi-vs-s at different times
k=-1
#fig = plt.figure(figsize=(26, 3.6))#figsize=(25.2, 5.1))
fig = plt.figure(figsize=(25.55, 4.90)) #NOTE: cannot unpack non-iterable Figure object
#fig, axs = plt.subplots(1, nplots+1, figsize=(21, 7.4))#figsize=(22.1, 7.4))#, sharex=True)#, layout='constrained')
for n_t in t_pick:
    #cmax = abs(fieldplot[n_t,:,:]).max()*0.92
    llevels=np.linspace(-cmax,cmax,64)
    clevels=np.linspace(-cmax,cmax,5)
    k+=1
    ax = plt.subplot2grid((1,nplots+1), (0,k))
    #axs[k] = plt.subplot(1,nplots+1,k+1)
    #ax = axs[k]
    if ( not l_rnorm ):
        if ( l_deminc and l_resym ):
            cf = ax.contourf(xx,yy,fieldplot[n_t,:,:],levels=llevels,extend='both',cmap='seismic')#'PuOr_r')#cmo.balance)#
        else:
            cf = ax.contourf(xx,yy,fieldplot[n_t,:nphi,:],levels=llevels,extend='both',cmap='seismic')#'PuOr_r')#cmo.balance)#
            #cf = ax.pcolormesh(xx,yy,fieldplot[n_t,:nphi,:],vmin=-cmax,vmax=cmax,antialiased=True,shading='gouraud',rasterized=True,cmap='seismic')#'PuOr_r')#cmo.balance)#
    else:
        if ( l_deminc and l_resym ):
            cf = ax.contourf(xx,yy,(fieldplot[n_t,:,:]/abs(fieldplot[n_t,:,:]).max(axis=0)),vmin=-1.,vmax=1.02,levels=32,cmap='seismic')#'PuOr_r')#cmo.balance)#
        else:
            cf = ax.contourf(xx,yy,(fieldplot[n_t,:nphi,:]/abs(fieldplot[n_t,:nphi,:]).max(axis=0)),vmin=-1.,vmax=1.02,levels=32,cmap='seismic')#'PuOr_r')#cmo.balance)#
    ax.plot(radius[0]*np.cos(phi), radius[0]*np.sin(phi), 'k-', lw=1.5)
    ax.plot(radius[-1]*np.cos(phi), radius[-1]*np.sin(phi), 'k-', lw=1.5)
    if ( (not l_deminc) and (minc!=1) ):
        xa = radius[-1]*np.cos(phi[0])
        ya = radius[-1]*np.sin(phi[0])
        xb = radius[0]*np.cos(phi[0])
        x0 = np.linspace(xa, xb, 32)
        y0 = np.tan(phi[0])*(x0-xa)+ya
        ax.plot(x0, y0+1.e-2, color=Ftrajcolor, ls='-.', lw=6.4, alpha=0.9)
        xc = radius[-1]*np.cos(phi[-1])
        yc = radius[-1]*np.sin(phi[-1])
        xd = radius[0]*np.cos(phi[-1])
        x1 = np.linspace(xc, xd, 32)
        y1 = np.tan(phi[-1])*(x1-xc)+yc
        ax.plot(x1, y1, color=Ftrajcolor, ls='-.', lw=6.4, alpha=0.9)
        xe = radius[-1]*np.cos(phi[24])
        ye = radius[-1]*np.sin(phi[24])
        xf = radius[0]*np.cos(phi[24])
        x2 = np.linspace(xe, xf, 32)
        y2 = np.tan(phi[24])*(x2-xe)+ye
        ax.plot(x2, y2, color=Ftrajcolor, ls='-.', lw=6.4, alpha=0.9)
        ax.plot(x0, y0, 'k-', lw=1.1)
        ax.plot(x1, y1, 'k-', lw=1.1)
    #
    #-- To avoid lines eating borders
    if ( xx.min()<0 ):
        ax.set_xlim(1.02*xx.min(), 1.02*xx.max())
    elif ( xx.min()==0 ):
        ax.set_xlim(xx.min()-0.02, 1.02*xx.max())
    else:
        ax.set_xlim(0.98*xx.min(), 1.02*xx.max())
    if ( yy.min()<0 ):
        ax.set_ylim(1.02*yy.min(), 1.02*yy.max())
    elif ( yy.min()==0 ):
        ax.set_ylim(yy.min()-0.02, 1.02*yy.max())
    else:
        ax.set_ylim(0.98*yy.min(), 1.02*yy.max())
    ax.axis('off')
    plt.gca().xaxis.set_tick_params(labelsize=32)
    plt.gca().yaxis.set_tick_params(labelsize=32)
    #plt.title(r'time $t = $'+str(np.round(tdimt[n_t],2)), fontsize=32)# NOTE: Really better to use "format", e.g.: $print('{:1.2f}'.format(123.456))
    plt.title(r'time $t = {:.2f}$'.format(tdimt[n_t]), fontsize=32)
    cb = fig.colorbar(cf, ax=ax, fraction=0.05, pad=0.085, orientation='horizontal', ticks=clevels, format=r'${x:.1f}$')
    cb.ax.tick_params(labelsize=32)
    if ( k==0 ):
        transAx = mtransforms.ScaledTranslation(+10/72, -5/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, plab, transform=ax.transAxes + transAx,
                fontsize=36, va='bottom', fontfamily='serif')
#
plt.tight_layout()
if ( not l_save ):  plt.show()

#-- Save True Fields analysis
if ( l_save ):
    if ( l_plot_fluct ):
        plt.savefig(savePlot+'Azimuth-Radius-Diagram-EquatContour_Minc_uphi.png')
    else:
        plt.savefig(savePlot+'Azimuth-Radius-Diagram-EquatContour_Minc_uphi-Zonal.png')
    plt.close('all')


#-- Fig. Answer to Referee: Barrois and Aubert 2024
#-- Aggregated plot Time-vs-Azimuth at different radii. NOTE: Need to read and store all equatorial frames.
eps = 1.e-3

l_plot_fluct = 1
#-- Field to plot
if ( l_plot_fluct == 1 ):
    fieldplot = upft.copy()#upt.copy()#brt.copy()#
    cmax = 8.0e0#1.0e-4#2.#4.#upft# 12.#upt# 
    cmapplot = plt.cm.seismic#plt.cm.PuOr_r#cmo.balance#
    plab = PLabels[2]
    ncl = 5
elif ( l_plot_fluct == 2 ):
    fieldplot = brt.copy()#
    cmax = 1.0e-4#8.0e-5#
    cmapplot = plt.cm.PuOr_r#
    plab = PLabels[3]
    ncl = 3
else:
    fieldplot = upt.copy()#
    cmax = 12.#6.#12.#upt#
    cmapplot = plt.cm.seismic#
    plab = PLabels[1]
    ncl = 5

nplots = 4
t_pick = np.array([3, 100, 200, 300, 397])#np.array([3, 243, 486, 729, 972])#
xlevels=(0.5, 1.0, 1.5, 2.0)#(0.0, 0.7, 1.4, 2.1)#
ylevels=(0.4, 0.8, 1.2, 1.6)
#
k=-1
fig = plt.figure(figsize=(21, 7.4))
for n_r in t_pick:
    #cmax = 6. #abs(fieldxtd[:,:nphi,n_s]).max()*0.92#
    llevels=np.linspace(-cmax,cmax,64)
    clevels=np.linspace(-cmax,cmax,ncl)#5)#3)#
    k+=1
    ax = plt.subplot2grid((1,nplots+1), (0,k))
    cf = ax.contourf(phi,tdimt,fieldplot[:,:nphi,n_r],levels=llevels,extend='both',cmap=cmapplot)#
    plt.xlabel(r'Azimuth, $\phi$', fontsize=36)
    if (k==0):  plt.ylabel(r'Time, $t$', fontsize=36)
    plt.xticks(xlevels)
    plt.yticks(ylevels)
    plt.gca().xaxis.set_tick_params(labelsize=32)
    plt.gca().yaxis.set_tick_params(labelsize=32)
    #plt.title(r'time $t = $'+str(np.round(tana[n_t],2)))
    plt.title(r'$r = {:.2f}$'.format(radius[n_r]), fontsize=32)
    if ( l_plot_fluct == 2 ):
        cb = fig.colorbar(cf, ax=ax, fraction=0.05, pad=0.16, orientation='horizontal', ticks=clevels, format=r'${x:.0e}$')
        cb.ax.tick_params(labelsize=24)#
    else:
        cb = fig.colorbar(cf, ax=ax, fraction=0.05, pad=0.16, orientation='horizontal', ticks=clevels, format=r'${x:.0f}$')
        cb.ax.tick_params(labelsize=32)#24)#
plt.tight_layout()
plt.show()
#
#-- Save True Fields analysis
if ( l_save and rank==0 ):
    plt.figure(1)
    plt.savefig(savePlot+'ATime-Azimuthal-Diagram_field-Col.png')
    plt.close('all')
            
#-- End Script
