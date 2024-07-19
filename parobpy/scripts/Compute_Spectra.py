#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 2024
@author: obarrois

Loads graphics files and compute kinetic and magnetic Energies from frames with a sampling rate t_sampling.

Then one can save the quantities of interest or plot directly Ekin and Emag.
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

def Smooth_8N(field_2D):
    field_2D_n8 = field_2D.copy()
    for i in range(field_2D.shape[0]):
        for j in range(field_2D.shape[1]):
            if ( np.isnan(field_2D[i,j]) ):
                #-- Compute mean over the 8 neighbours, excluding the spurious point(s)
                field_2D_n8[i,j] = 1.
                try:
                    sum_n8 = 0.; count = 0.
                    for ii in range(4):
                        for jj in range(4):
                            if ( (i-1+ii != i)  and (j-1+jj != j) and (not np.isnan(field_2D[i-1+ii,j-1+jj]))  ):
                                sum_n8+= field_2D[i-1+ii,j-1+jj]
                                count+= 1
                    if ( count == 0. ):  count = 1.
                    field_2D_n8[i,j] = sum_n8/count
                except IndexError:
                    pass
    if ( np.isnan(field_2D_n8.sum()) ):
        field_2D_n8+=1.
    return field_2D_n8

#----------------------------------------------------------------------------%%
#-- INPUT PARAMETERS

#-- Lehnert number, \lambda = B/{sqrt(rho mu) * Omega*d}.
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '3e7', 'Pm0o25', 1.46e-3, 1.0, 'b3', '0.000280800' # 3e-7 S=1214.1
run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1', 1.1e-3, 1.0, 'b4', '0.000040320' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1', 1.1e-3, 1.0, 'bbi1e7', '0.000071360' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1_per', 1.1e-3, 1.0, 'bbi1e7p', '0.000147200' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'PathBase', 1.1e-3, 1.0, 'b4-p', '0.000008755' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'PathLund3o2e3', 5.53e-4, 1.0, 'b4o6', '0.000015600' # 6.3e-9 (1e7 grid)
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e8', '', 5.4e-4, 1.0, 'b4o5', '0.000024896' # 1e-8
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '6e9', 'Pm0o36e-1', 5.53e-4, 1.0, 'b4o63', '0.000015560' # 6.3e-9
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '3e10', '', 2.6e-4, 1.0, 'b5', '0.000001456' # 3e-10

#saveDir = '/gpfs/users/obarrois/Parodys/Outputs/Data/' # if you use python directly from cluster
#directory = '/gpfs/users/obarrois/Work/Waves1e7/' # if you use python directly from cluster
#directory = '/Users/obarrois/Desktop/dfgnas3/Waves/Data_3e10/' # run at 3e-10 on jaubert disk
directory = '/Users/obarrois/Desktop/stella/Work/IC_Impulse/Waves'+run_Ek+'/'+run_Pm+'/' # if you mount stella on own desk

l_noIC = True # Excluding z-avg fields inside of the TC?
l_resym = False # re-symmetrizing the data if minc !=1
l_back_b0 = False # add back the background magnetic field b0 --> should be False because we are interested in the perturbations (we only see B0 otherwise)
phi_sample = 3 # longitude to extract for phi sampling

#l_sample = True # sample Gt time files instead of computing for every t=... (advised: could be really slow otherwise)
tstart = 0 # time rank of the first Gt file read. Should be <= ntime and can be mofidify to scan != samples with t_srate
t_srate = 1 # sampling rate of Gt files if needed. Should be <= ntime
l_subplot = False # Select only a subsampling of the frames for faster computation

l_plot_mean = False # produce and save phi-sampled or phi-Averaged Energies
l_check_all = False # display all check plot figures
l_save = 1 # save figures?
l_zavg = 1 # using python method to compute z-avg of the forces
savePlot = '/Users/obarrois/Desktop/Parodys/Outputs/IC_Impulse/Waves'+run_Ek+'/Signal/Spectra/' # path to save files

#------------------------------------------------------------------------------
#%% Load data

Gt_file = 'Gt={}.{}'.format(timestamp,run_ID)
filename = directory + Gt_file

(version, time, DeltaU, Coriolis, Lorentz, Buoyancy, ForcingU,
            DeltaT, ForcingT, DeltaB, ForcingB, Ek, Ra, Pm, Pr,
            nr, ntheta, nphi, azsym, radius, theta, phi, _, _, _,
            _, _, _, _) = parodyload(filename)

if ( run_Pm == 'PathBase' ):
    basedir = '/Users/obarrois/Desktop/stella/Work/IC_Impulse/' # basefield copied on stella
    basename = basedir+'basefield_path.mat' # if you use python directly from cluster
else:
    basedir = '/Users/obarrois/Desktop/stella/Work/IC_Impulse/Waves'+run_Ek+'/' # basefield copied on stella
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

#-- Time and Forces Redimensionalisation, if needed
#Lehnert = np.sqrt(1.33**2*(Ek/Pm))
tredim = 1./(Ek/Lehnert)
Bdim = 1./(Pm*Ek)#(Pm*Ek)/(Pm*Lehnert)# NOTE: Viscous scaling because all quantities are directly from Parody (would have to rescale U and B differently if one wants to use Alfvén units)

if ( l_check_all ):
    #-- Read spectra if needed
    directory = '/Users/obarrois/Desktop/stella/Work/IC_Impulse/Waves'+run_Ek+'/'+run_Pm+'/' # if you mount stella on own desk
    read_spec = np.loadtxt(directory+'spec_l.'+run_ID,dtype=str)
    l_spec = np.array(read_spec[:,0],dtype=np.int16)
    l_max = l_spec[-1]; l_h = 30
    Spec = np.zeros((4,l_max+1),dtype=np.float64)
    for n_spec in range(l_max+1):
        #print(read_spec[:,1][n_spec])
        #print(read_spec[:,1][n_spec][:10], 'E', read_spec[:,1][n_spec][11:])
        Spec[0,n_spec] = np.float64(read_spec[:,1][n_spec][:10]+'E'+read_spec[:,1][n_spec][11:])
        Spec[1,n_spec] = np.float64(read_spec[:,2][n_spec][:10]+'E'+read_spec[:,2][n_spec][11:])
        Spec[2,n_spec] = np.float64(read_spec[:,3][n_spec][:10]+'E'+read_spec[:,3][n_spec][11:])
        Spec[3,n_spec] = np.float64(read_spec[:,4][n_spec][:10]+'E'+read_spec[:,4][n_spec][11:])
    #
    #-- Plot spectra if needed
    cmax = np.amax(Spec[:,1:])*1.1; cmin = np.amin(Spec[:,1:])*0.9
    fig = plt.figure(figsize=(14.5, 12.5))
    ax = plt.subplot(111)
    ax.loglog(l_spec[1:], Spec[1][1:], color='b', ls='-',marker='', lw='3.1', alpha=0.9, label=r'$\widehat{E}_\mathrm{kin}(\ell)$') # u_avg
    ax.loglog(l_spec[1:], Spec[0][1:], color='c', ls='--',marker='', lw='2.6', alpha=0.7, label=r'$E_\mathrm{kin}(t_\mathrm{end}, \ell)$') # u_last
    ax.loglog(l_spec[1:], Spec[3][1:], color='r', ls='-',marker='', lw='3.1', alpha=0.9, label=r'$\widehat{E}_\mathrm{mag}(\ell)$') # b_avg
    ax.loglog(l_spec[1:], Spec[2][1:], color='#FF7F00', ls='--',marker='', lw='2.6', alpha=0.7, label=r'$E_\mathrm{mag}(t_\mathrm{end}, \ell)$') # b_last
    ax.loglog([l_h, l_h], [cmin, cmax], color='k', ls=':',marker='', lw='2.7', alpha=0.8, label=r'Hyperdiffusive cut-off degree, $\ell_h$') # l_h
    plt.xlim(l_spec[1],l_max)
    plt.ylim(cmin,cmax)
    plt.xlabel(r'Spherical Harmonic degree, $\ell$', fontsize=36)#
    plt.ylabel(r'Energy spectra', fontsize=36)#
    plt.gca().xaxis.set_tick_params(which='major', labelsize=32)
    plt.gca().xaxis.set_tick_params(which='minor', labelsize=32)
    plt.gca().yaxis.set_tick_params(which='major', labelsize=32)
    plt.gca().yaxis.set_tick_params(which='minor', labelsize=32)
    plt.grid(ls=':')
    plt.legend(fontsize=32)
    #
    plt.tight_layout()
    transAx = mtransforms.ScaledTranslation(-1.95+40/72, -35/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, r'$(a)$', transform=ax.transAxes + transAx,
            fontsize=36, va='bottom', fontfamily='serif')
    if ( not l_save ):  plt.show()
    #plt.show()
    #
    #-- Save Spectra if needed
    if ( l_save ):
        plt.figure(1)
        plt.savefig(savePlot+'Spectra_Emag+Ekin.pdf')
        plt.close('all')


#------------------------------------------------------------------------------
#%% Read, process and save chosen data

Gt_file_l = list_Gt_files(run_ID,directory) # Find all Gt_no in folder
n_samples = len(Gt_file_l[tstart::t_srate])

timet = np.zeros((n_samples),)
#-- 3D quantities
Ekin = np.zeros((n_samples, ntheta, nr),)# nphir, ntheta, nr),)#
Eflu = np.zeros_like(Ekin)
Ezon = np.zeros_like(Ekin)
Emag = np.zeros_like(Ekin)
Emaf = np.zeros_like(Ekin)
Emaz = np.zeros_like(Ekin)
Ekinm = np.zeros_like(Ekin)
Eflum = np.zeros_like(Ekin)
Emagm = np.zeros_like(Ekin)
Emafm = np.zeros_like(Ekin)

#-- Selecting data for plotting if needed
if ( l_subplot ):
    nplots = 6
    tpk = int(len(timet[:500])/nplots)
    #t_pick = np.arange(1,len(timet[:500]),tpk)
    nplots = 4
    t_pick = np.array([99, 199, 299, 399, 499])
else:
    t_pick = np.arange(tstart,len(Gt_file_l),t_srate)

n_s=-1#; n_t=-1
#-- Start loop
#for file in np.array(Gt_file_l)[t_pick]:#[249:250]:#
for file in Gt_file_l[tstart::t_srate]:#[249:250]:#
    n_s+=1
    #n_t+=1; n_s = t_pick[n_t]
    #------------------------------------------------------------------------------
    #-- loading pointtime data
    print('Loading {} (({}/{})/{})'.format(file, n_s+1, n_samples, len(Gt_file_l)))
    filename = '{}/{}'.format(directory,file)
    (_, time, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, ur, ut, up,
        Br, Bt, Bp, _) = parodyload(filename)
    timet[n_s] = time
    #
    #-- resymmetrize the data if minc > 1
    if ( l_resym and minc !=1 ):
        print('Re-symmetrizing the data for file n°{}, t={}'.format(n_s+1, time))
        ur = symmetrize(ur,ms=minc); Br = symmetrize(Br,ms=minc)
        ut = symmetrize(ut,ms=minc); Bt = symmetrize(Bt,ms=minc)
        up = symmetrize(up,ms=minc); Bp = symmetrize(Bp,ms=minc)
    #
    #-- Adding back the background magnetic field B0 to B
    if ( l_back_b0 ):
        # NOTE: no u0 for velocity so no need to add anything for u
        print('Adding background field for sample n°{}/{}'.format(n_s+1, n_samples))
        Br = Br + B0r
        Bt = Bt + B0t
        Bp = Bp + B0p
    #
    #-- Removing the phi-avg for better waves spooting
    #-- f_fluct = f - f_phi-Avg, if needed
    #-- Vp = up_m=0 = \overline{up} = axisymmetric or zonal azimuthal velocity
    urf = np.zeros_like(ur); Brf = np.zeros_like(Br)
    utf = np.zeros_like(ut); Btf = np.zeros_like(Bt)
    upf = np.zeros_like(up); Bpf = np.zeros_like(Bp)
    urpm = ur.mean(axis=0)
    utpm = ut.mean(axis=0)
    Vp = up.mean(axis=0)
    Brpm = Br.mean(axis=0)
    Btpm = Bt.mean(axis=0)
    Bppm = Bp.mean(axis=0)
    for n_p in range(nphir):
        #print('Removing mean field for phi {}/{}'.format(n_p+1, nphir))
        urf[n_p] = ur[n_p] - urpm
        utf[n_p] = ut[n_p] - utpm
        upf[n_p] = up[n_p] - Vp#
        Brf[n_p] = Br[n_p] - Brpm
        Btf[n_p] = Bt[n_p] - Btpm
        Bpf[n_p] = Bp[n_p] - Bppm
    #
    #-- Select equatorial plane and save the field
    Ekin[n_s] = 0.5*(ur**2 + ut**2 + up**2)[phi_sample]#
    Ekinm[n_s] = 0.5*(ur**2 + ut**2 + up**2).mean(axis=0)#[phi_sample]#
    Eflu[n_s] = 0.5*(urf**2 + utf**2 + upf**2)[phi_sample]#
    Eflum[n_s] = 0.5*(urf**2 + utf**2 + upf**2).mean(axis=0)#
    Ezon[n_s] = 0.5*Vp**2#(urpm**2 + utpm**2 + Vp**2)#-- Already phi-Avg!
    Emag[n_s] = 0.5*Bdim*(Br**2 + Bt**2 + Bp**2)[phi_sample]#
    Emagm[n_s] = 0.5*Bdim*(Br**2 + Bt**2 + Bp**2).mean(axis=0)#
    Emaf[n_s] = 0.5*Bdim*(Brf**2 + Btf**2 + Bpf**2)[phi_sample]#
    Emafm[n_s] = 0.5*Bdim*(Brf**2 + Btf**2 + Bpf**2).mean(axis=0)#
    Emaz[n_s] = 0.5*Bdim*Bppm**2#(Brpm**2 + Btpm**2 + Bppm**2)#-- Already phi-Avg!
    #
#-- end loop reading and computing Energies

#------------------------------------------------------------------------------
#%% Additional processing if needed

#-- Compute z-Avg of Energies if needed
if ( l_zavg ): 
    from vizuals import zavg
    #
    rrad = radius[::-1] # zavg() expect reversed radius
    #nsmax = 309 #--> to get nsmax ~ nsmaxja
    nsmax = int(nr*1.5 + 17) # to maintain nsmax ~ nrmax outside of TC (but will give nsmax > nrmax)
    #
    #-- z-Avg quantities
    Ekincol = np.zeros((n_samples, nsmax),) #, 617),) #401),) #401 = length of r_s for case at Ek=1e-7 (can not be really known before z-Avg or a preliminary test)
    Eflucol = np.zeros_like(Ekincol)
    Ezoncol = np.zeros_like(Ekincol)
    Emagcol = np.zeros_like(Ekincol)
    Emafcol = np.zeros_like(Ekincol)
    Emazcol = np.zeros_like(Ekincol)
    Ekinmcol = np.zeros_like(Ekincol)
    Eflumcol = np.zeros_like(Ekincol)
    Emagmcol = np.zeros_like(Ekincol)
    Emafmcol = np.zeros_like(Ekincol)
    #
    #-- Compute z-Avg force balances
    # NOTE: WARNING!:: zavg() or zavgpy() expect reversed radius as in pizza/MagIC (r_cmb = r[0]) SO the arrays MUST BE REVSERED as well
    print('Computing Fortran z-avg of the Forces for all {} files'.format(n_samples))
    hs, rs, ekincols = zavg(Ekin[... ,::-1], rrad, nsmax, colat=theta, minc=1, normed=True, save=False)
    hs, rs, ekinmcols = zavg(Ekinm[... ,::-1], rrad, nsmax, colat=theta, minc=1, normed=True, save=False)
    hs, rs, eflucols = zavg(Eflu[... ,::-1], rrad, nsmax, colat=theta, minc=1, normed=True, save=False)
    hs, rs, eflumcols = zavg(Eflum[... ,::-1], rrad, nsmax, colat=theta, minc=1, normed=True, save=False)
    hs, rs, ezoncols = zavg(Ezon[... ,::-1], rrad, nsmax, colat=theta, minc=1, normed=True, save=False)
    hs, rs, emagcols = zavg(Emag[... ,::-1], rrad, nsmax, colat=theta, minc=1, normed=True, save=False)
    hs, rs, emagmcols = zavg(Emagm[... ,::-1], rrad, nsmax, colat=theta, minc=1, normed=True, save=False)
    hs, rs, emafcols = zavg(Emaf[... ,::-1], rrad, nsmax, colat=theta, minc=1, normed=True, save=False)
    hs, rs, emafmcols = zavg(Emafm[... ,::-1], rrad, nsmax, colat=theta, minc=1, normed=True, save=False)
    hs, rs, emazcols = zavg(Emaz[... ,::-1], rrad, nsmax, colat=theta, minc=1, normed=True, save=False)
    #-- exluding IC
    if ( l_noIC ):
        Ekincol = ekincols[:,rs>=radius[0]]; Emagcol = emagcols[:,rs>=radius[0]]
        Eflucol = eflucols[:,rs>=radius[0]]; Emafcol = emafcols[:,rs>=radius[0]]
        Ezoncol = ezoncols[:,rs>=radius[0]]; Emazcol = emazcols[:,rs>=radius[0]]
        Ekinmcol = ekinmcols[:,rs>=radius[0]]; Emagmcol = emagmcols[:,rs>=radius[0]]
        Eflumcol = eflumcols[:,rs>=radius[0]]; Emafmcol = emafmcols[:,rs>=radius[0]]
        sr = rs[rs>=radius[0]]
    else: #-- with IC
        Ekincol = ekincols[:,:]; Emagcol = emagcols[:,:]
        Eflucol = eflucols[:,:]; Emafcol = emafcols[:,:]
        Ezoncol = ezoncols[:,:]; Emazcol = emazcols[:,:]
        Ekinmcol = ekinmcols[:,:]; Emagmcol = emagmcols[:,:]
        Eflumcol = eflumcols[:,:]; Emafmcol = emafmcols[:,:]
        sr = rs[:]
    del(ekincols, ekinmcols, eflucols, eflumcols, ezoncols)
    del(emagcols, emagmcols, emafcols, emafmcols, emazcols)
    #
#-- end loop z-Avg of Energies

#-- Redim time
tdim = timet*tredim#/(Ek/Lehnert)

#------------------------------------------------------------------------------
#%% Compute trajectories along Aflven speeds

dirva = '/Users/obarrois/Desktop/stella/Work/IC_Impulse/Waves'+run_Ek+'/'
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
    dirva = '/Users/obarrois/Desktop/stella/Work/IC_Impulse/Waves'+run_Ek+'/'
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
    advtime = 1.e-2
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

#-- Palette
WA_Dar1_lightBlue = '#5bbcd6'
WA_Ziss_Yellow = '#ebcc2a'
WA_BRoc_Dark = '#0d1606'
Personal_Cyan = '#1cfaf7'

#-- Adjusting colorsfor trajs #CB Barrois and Aubert 2024
trajcolor = WA_BRoc_Dark
trajcolor2= WA_BRoc_Dark#Personal_Cyan#WA_Dar1_lightBlue#WA_BRoc_Yellow

#------------------------------------------------------------------------------
#%% Main plot section

#plt.rcParams['text.usetex'] = True
#-- Ekin Tot
ffield = Ekincol#Ekin[:,ntheta//2,:]
Mratio = np.log(np.sqrt(ffield))
fig = plt.figure(figsize=(16, 12.5))
cmax = (Mratio[:,:]).max()*0.9
llevels=np.linspace(0.,cmax,64)
clevels=np.linspace(0.,cmax,5)
ax = plt.subplot(111)
#cf = ax.contourf(sr,tdim,Mratio[:,:],levels=llevels,extend='both',cmap=cmo.thermal)
cf = ax.contourf(sr,tdim,Mratio[:,:],levels=64,extend='both',cmap=cmo.ice)#cmo.haline)#cmo.thermal)
plt.xlabel(r'Cylindrical radius, $s$', fontsize=36)#
plt.ylabel(r'Time, $t$', fontsize=36)#
cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.05, orientation='vertical')#, ticks=clevels, format=r'${x:.2f}$')
plt.gca().xaxis.set_tick_params(which='major', labelsize=32)
plt.gca().xaxis.set_tick_params(which='minor', labelsize=32)
plt.gca().yaxis.set_tick_params(which='major', labelsize=32)
plt.gca().yaxis.set_tick_params(which='minor', labelsize=32)
cb.set_label(label=r'$E_{kin} = \frac{1}{2} \lbrace {\bf u}^2 \rbrace_{3D}$',size=36)#, weight='bold')
cb.ax.tick_params(labelsize=32)
#
plt.tight_layout()

#-- Ekin_fluct
ffield = Eflucol#Ekin[:,ntheta//2,:]
Mratio = np.log(np.sqrt(ffield))
fig = plt.figure(figsize=(16, 12.5))
cmax = (Mratio[:,:]).max()*0.9
llevels=np.linspace(0.,cmax,64)
clevels=np.linspace(0.,cmax,5)
ax = plt.subplot(111)
#cf = ax.contourf(sr,tdim,Mratio[:,:],levels=llevels,extend='both',cmap=cmo.thermal)
cf = ax.contourf(sr,tdim,Mratio[:,:],levels=64,extend='both',cmap=cmo.ice)#cmo.haline)#cmo.thermal)
plt.xlabel(r'Cylindrical radius, $s$', fontsize=36)#
plt.ylabel(r'Time, $t$', fontsize=36)#
cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.05, orientation='vertical')#, ticks=clevels, format=r'${x:.2f}$')
plt.gca().xaxis.set_tick_params(which='major', labelsize=32)
plt.gca().xaxis.set_tick_params(which='minor', labelsize=32)
plt.gca().yaxis.set_tick_params(which='major', labelsize=32)
plt.gca().yaxis.set_tick_params(which='minor', labelsize=32)
cb.set_label(label=r'$E_{kin-fluct} = \frac{1}{2} \lbrace \tilde{\bf u}^2 \rbrace_{3D}$',size=36)#, weight='bold')
cb.ax.tick_params(labelsize=32)
#
plt.tight_layout()

#-- Emag Tot
ffield = Emagcol#Emag[:,ntheta//2,:]
Mratio = np.log(np.sqrt(ffield))
fig = plt.figure(figsize=(16, 12.5))
cmax = (Mratio[:,:]).max()*0.9
llevels=np.linspace(0.,cmax,64)
clevels=np.linspace(0.,cmax,5)
ax = plt.subplot(111)
#cf = ax.contourf(sr,tdim,Mratio[:,:],levels=llevels,extend='both',cmap=cmo.thermal)
cf = ax.contourf(sr,tdim,Mratio[:,:],levels=64,extend='both',cmap=cmo.solar)#cmo.thermal)#
plt.xlabel(r'Cylindrical radius, $s$', fontsize=36)#
plt.ylabel(r'Time, $t$', fontsize=36)#
cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.05, orientation='vertical')#, ticks=clevels, format=r'${x:.2f}$')
plt.gca().xaxis.set_tick_params(which='major', labelsize=32)
plt.gca().xaxis.set_tick_params(which='minor', labelsize=32)
plt.gca().yaxis.set_tick_params(which='major', labelsize=32)
plt.gca().yaxis.set_tick_params(which='minor', labelsize=32)
cb.set_label(label=r'$E_{mag} =\frac{1}{Pm\,\lambda} \frac{1}{2} \lbrace {\bf b}^2 \rbrace_{3D}$',size=36)#, weight='bold')
cb.ax.tick_params(labelsize=32)
#
plt.tight_layout()
if ( not l_save ):  plt.show()

#-- Save True Fields analysis
if ( l_save ):
    plt.figure(1)
    plt.savefig(savePlot+'Radial_time_Ekin_Tot-Col_phi-Avg.png')
    plt.figure(2)
    plt.savefig(savePlot+'Radial_time_Ekin_Fluct-Col_phi-Avg.png')
    plt.figure(3)
    plt.savefig(savePlot+'Radial_time_Emag_Fluct-Col_phi-Avg.png')
    plt.close('all')

#plt.rcParams['text.usetex'] = True
#-- Emag/Ekin -- Try -- #
#-- Emag/Ekin Tot
epss=0.#1e-15
if ( l_plot_mean ):
    ffield = (3.*Emagmcol+epss)/(Ekinmcol+epss)#(Emag[:,ntheta//2,:]+epss)/(Ekin[:,ntheta//2,:]+epss)#
else:
    ffield = (3.*Emagcol+epss)/(Ekincol+epss)#+epss)#(Emagcol+epss)/(Ekincol+epss)#(Emag[:,ntheta//2,:]+epss)/(Ekin[:,ntheta//2,:]+epss)#
Mratio = np.log((ffield))
if ( np.isnan(Mratio.sum()) ): Mratio = Smooth_8N(Mratio)
#
fig = plt.figure(figsize=(16, 12.5))
cmax = (Mratio[:,:]).max()*0.9
cmin = -cmax#(Mratio[:,:]).min()*0.9#
llevels=np.linspace(cmin,cmax,64)
clevels=np.linspace(cmin,cmax,5)
ax = plt.subplot(111)
cf = ax.contourf(sr,tdim,Mratio[:,:],levels=llevels,extend='both',cmap='BrBG_r')#'RdGy_r')#cmo.balance)#cmo.thermal)
ax.plot(rtraj, ttraj, color=trajcolor, ls='-',marker='', lw='6.4', alpha=0.9) # Va_mean
plt.ylim(tdim[0],tdim[-2])#1.76)#
plt.xlabel(r'Cylindrical radius, $s$', fontsize=36)#
plt.ylabel(r'Time, $t$', fontsize=36)#
cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.05, orientation='vertical', ticks=clevels, format=r'${x:.2f}$')
plt.gca().xaxis.set_tick_params(which='major', labelsize=32)
plt.gca().xaxis.set_tick_params(which='minor', labelsize=32)
plt.gca().yaxis.set_tick_params(which='major', labelsize=32)
plt.gca().yaxis.set_tick_params(which='minor', labelsize=32)
if ( l_plot_mean ):
    cb.set_label(label=r'log$\left(\left< 3\,\overline{E}_\mathrm{mag}/\overline{E}_\mathrm{kin} \right>\right)$',size=36)#, weight='bold')
else:
    cb.set_label(label=r'log$\left(\left< 3\,E_\mathrm{mag}/E_\mathrm{kin} \right>\right)$',size=36)#, weight='bold')
cb.ax.tick_params(labelsize=32)
#
plt.tight_layout()
transAx = mtransforms.ScaledTranslation(13.95+10/72, -45/72, fig.dpi_scale_trans)
ax.text(0.0, 1.0, r'$(a)$', transform=ax.transAxes + transAx,
        fontsize=36, va='bottom', fontfamily='serif')

#-- Emag_fluct/Ekin_fluct
if ( l_plot_mean ):
    ffield = (3.*Emafmcol+epss)/(Eflumcol+epss)#(Emagmcol+epss)/(Eflumcol+epss)#
else:
    ffield = (3.*Emafcol+epss)/(Eflucol+epss)#(Emagcol+epss)/(Eflucol+epss)#
Mratio = np.log((ffield))
if ( np.isnan(Mratio.sum()) ): Mratio = Smooth_8N(Mratio)
#
fig = plt.figure(figsize=(16, 12.5))
cmax = (Mratio[:,:]).max()*0.9
llevels=np.linspace(-cmax,cmax,64)
clevels=np.linspace(-cmax,cmax,5)
ax = plt.subplot(111)
cf = ax.contourf(sr,tdim,Mratio[:,:],levels=llevels,extend='both',cmap='BrBG_r')#'RdGy_r')#cmo.balance)#cmo.thermal)
ax.plot(rtraj2, ttraj2, color=trajcolor2, ls='-.',marker='', lw='6.4', alpha=0.9) # Va_fast
plt.ylim(tdim[0],tdim[-2])#1.76)#
plt.xlabel(r'Cylindrical radius, $s$', fontsize=36)#
plt.ylabel(r'Time, $t$', fontsize=36)#
cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.05, orientation='vertical', ticks=clevels, format=r'${x:.2f}$')
plt.gca().xaxis.set_tick_params(which='major', labelsize=32)
plt.gca().xaxis.set_tick_params(which='minor', labelsize=32)
plt.gca().yaxis.set_tick_params(which='major', labelsize=32)
plt.gca().yaxis.set_tick_params(which='minor', labelsize=32)
if ( l_plot_mean ):
    cb.set_label(label=r'log$\left(\left< 3\,\overline{\tilde{E}}_\mathrm{mag}/\overline{\tilde{E}}_\mathrm{kin} \right>\right)$',size=36)#, weight='bold')
else:
    cb.set_label(label=r'log$\left(\left< 3\,\tilde{E}_\mathrm{mag}/\tilde{E}_\mathrm{kin} \right>\right)$',size=36)#, weight='bold')
cb.ax.tick_params(labelsize=32)
#
plt.tight_layout()
transAx = mtransforms.ScaledTranslation(13.95+10/72, -45/72, fig.dpi_scale_trans)
ax.text(0.0, 1.0, r'$(b)$', transform=ax.transAxes + transAx,
        fontsize=36, va='bottom', fontfamily='serif')

#-- Emag_zon/Ekin_zon Tot
ffield = (3.*Emazcol+epss)/(Ezoncol+epss)#+epss)#(Emagcol+epss)/(Ekincol+epss)#(Emaz[:,ntheta//2,:]+epss)/(Ezon[:,ntheta//2,:]+epss)#
Mratio = np.log((ffield))
if ( np.isnan(Mratio.sum()) ): Mratio = Smooth_8N(Mratio)
#
fig = plt.figure(figsize=(16, 12.5))
cmax = (Mratio[:,:]).max()*0.9
llevels=np.linspace(-cmax,cmax,64)
clevels=np.linspace(-cmax,cmax,5)
ax = plt.subplot(111)
cf = ax.contourf(sr,tdim,Mratio[:,:],levels=llevels,extend='both',cmap='BrBG_r')#'RdGy_r')#cmo.balance)#cmo.thermal)
ax.plot(rtraj, ttraj, color=trajcolor, ls='-',marker='', lw='6.4', alpha=0.9) # Va_mean
plt.ylim(tdim[0],tdim[-2])#1.76)#
plt.xlabel(r'Cylindrical radius, $s$', fontsize=36)#
plt.ylabel(r'Time, $t$', fontsize=36)#
cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.05, orientation='vertical', ticks=clevels, format=r'${x:.2f}$')
plt.gca().xaxis.set_tick_params(which='major', labelsize=32)
plt.gca().xaxis.set_tick_params(which='minor', labelsize=32)
plt.gca().yaxis.set_tick_params(which='major', labelsize=32)
plt.gca().yaxis.set_tick_params(which='minor', labelsize=32)
cb.set_label(label=r'log$\left(\left< 3\,\overline{E}_\mathrm{mag}/\overline{E}_\mathrm{kin} \right>\right)$',size=36)#, weight='bold')
cb.ax.tick_params(labelsize=32)
#
plt.tight_layout()
transAx = mtransforms.ScaledTranslation(13.95+10/72, -45/72, fig.dpi_scale_trans)
ax.text(0.0, 1.0, r'$(a)$', transform=ax.transAxes + transAx,
        fontsize=36, va='bottom', fontfamily='serif')
if ( not l_save ):  plt.show()

#-- Save True Fields analysis
if ( l_save ):
    if ( l_plot_mean ):
        pTitle = '_phi-Avg'
    else:
        pTitle = ''
    plt.figure(1)
    plt.savefig(savePlot+'Radial_time_EmaglEkin_Tot-Col'+pTitle+'.png')
    plt.figure(2)
    plt.savefig(savePlot+'Radial_time_EmaglEkin_Fluct-Col'+pTitle+'.png')
    plt.figure(3)
    plt.savefig(savePlot+'Radial_time_EmaglEkin_Zon-Col.png')
    plt.close('all')

#-- End Script
