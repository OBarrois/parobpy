#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon June 22 2023
@author: obarrois

Python script to loads graphics file with either curl balances of coriolis, lorentz or vorticity.
Then extract signal information and plot FFT or Hankel transforms to get 'FK' (frequency vs k_s number) dispersion plots

Script to ONLY plot the Dispersion diagrams (one needs to use an other script to compute fields needed for it)!
"""

from parobpy.load_parody import parodyload, load_dimensionless, load_basefield
import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cmo

#----------------------------------------------------------------------------%%
#-- INPUT PARAMETERS

#-- Lehnert number, \lambda = B/{sqrt(rho mu) * Omega*d}.
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '3e7', 'Pm0o25', 1.46e-3, 1.0, 'b3', '0.000280800' # 3e-7 S=1214.1
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Lund8e2', 1.1e-3, 0.5, 'b4-5', '0.000288000' # 2e-7
run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1', 1.1e-3, 1.0, 'b4', '0.000040320' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'PathLund3o2e3', 5.53e-4, 1.0, 'b4o6', '0.000015600' # 6.3e-9 (1e7 grid)
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e8', 'PathLund3o2e3', 5.53e-4, 1.0, 'b4o6-2', '0.000008800' # 6.3e-9 (1e8 grid)
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e8', 'PathLund3o2e3BIS', 5.53e-4, 1.0, 'b4o6-2', '0.000009080' # 6.3e-9 (1e8 grid)BIS
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e8', 'Pm0o46e-1', 6.2e-4, 1.0, 'b4o5', '0.000004800' # 1e-8
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '6e9', 'Pm0o36e-1', 5.53e-4, 1.0, 'b4o63', '0.000015560' # 6.3e-9
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '3e10', '', 2.6e-4, 1.0, 'b5', '0.000001456' # 3e-10

#saveDir = '/gpfs/users/obarrois/Parodys/Outputs/Data/' # if you use python directly from cluster
#directory = '/gpfs/users/obarrois/Work/Waves1e7/' # if you use python directly from cluster
#directory = '/Users/obarrois/Desktop/dfgnas3/Waves/Data_3e10/' # run at 3e-10 on jaubert disk
directory = '/Users/obarrois/Desktop/stella/Work/Waves'+run_Ek+'/'+run_Pm+'/' # if you mount stella on own desk

#-- phi-Avg of field - phi-Avg gives only noise. Otherwise the rest is fine.
t_srate = 1 # sampling rate of Gt files if needed. Should be <= ntime
phi_sample = 3 # longitude to extract for phi sampling

rank = 0 # default proc number when l_mpi == False
l_mpi = True # use MPI to speed up horizontal gradients for the curl computation or not

l_field = 'Lorz' #field to be curled, transformed and plotted: can be either 'omz', 'Corz' or 'Lorz'
l_redim = True # re-dimensionalisation of the quantities
l_tau_A = True # re-dimensionalisation on the Alfven or on the Viscous time-scales
l_build_wave_disp = True # compute the theoretical wave-dispersion relations from (Gillet et al. 2022)

l_spectral = True # using spectral derivatives in theta and phi for curl? (advised:  = True)
l_trunc = False # truncation for SH?
l_check_all = False # display all check plot figures

l_noIC = True # Excluding z-avg fields inside of the TC?
l_resym = True # re-symmetrizing the data if minc !=1
l_lambda = False # Use the wavelength or compute the wave numbers to plot FK diagrams?
l_snorm = False # Normalise fields by the maximum at each s-radius or not
l_deminc = False # Plot fields including azimuthal symmetries or not (if minc !=1)

l_txtd = False # Extending time up to 2pi for FFT?
l_all_phi = True # Transform and Sum for all phi or not? #NOTE: Could be extremely slow if set to True
l_sqdht = False # Using Hankel transform (if True) for radius axis OR DCT (if False)

#-- I/O; NOTE: does not seem to replace files if they exist even if l_save*!
saveDir = '/Users/obarrois/Desktop/Parodys/Outputs/Waves'+run_Ek+'/Balances/Data/' # path to read colmunar balance files from
l_save = 0 # save main figures?
l_read = 1 # read zAvg-balances?
l_read_allphi = 1 # read zAvg-balances with all phi?
l_read_Va = 1 # read theoretical Alfven wave speed?
tagRead = '_-phi-Avg_n500'#_-phi-Avg_n600'#_-phi-Avg_n500_lowLehn'#_-phi-Avg_n500_lowerPm'# #  # tag at the end of the forces files if needed
savePlot = '/Users/obarrois/Desktop/Parodys/Outputs/Waves'+run_Ek+'/Signal/Dispersion/' # path to save main figures
#saveDir = '/Users/obarrois/Desktop/stella/Parodys/Outputs/Data/' # if you use python from perso to remote data directly from cluster

#------------------------------------------------------------------------------
#%% Initialisation

if ( l_read ):
    import h5py

#-- set up MPI proc ranks if l_mpi
try: #-- Import anyway if the script is called with mpiexec but l_mpi = False to avoid print and plot repetitions
    import mpi4py.MPI as mpi
    if ( l_mpi ): l_mpi = True # Do not erase l_mpi if the script user did not want to use mpi
    comm = mpi.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print('MPI set up for rank = {}, l_mpi = {}'.format(rank, l_mpi))
except:# ModuleNotFoundError:
    print("mpi4py: Module Not Found")
    l_mpi = False

Gt_file = 'Gt={}.{}'.format(timestamp,run_ID)
filename = directory + Gt_file

n_samples = 1
(version, time, DeltaU, Coriolis, Lorentz, Buoyancy, ForcingU,
            DeltaT, ForcingT, DeltaB, ForcingB, Ek, Ra, Pm, Pr,
            nr, ntheta, nphi, minc, radius, theta, phi, _, _, _,
            _, _, _, _) = parodyload(filename)

if ( run_Pm == 'PathBase' ):
    basedir = '/Users/obarrois/Desktop/stella/Work/' # basefield copied on stella
    basename = basedir+'basefield_path.mat' # if you use python directly from cluster
else:
    basedir = '/Users/obarrois/Desktop/stella/Work/Waves'+run_Ek+'/' # basefield copied on stella
    basename = basedir+'basefield.mat' # if you use python directly from cluster
B0r, B0t, B0p, j0r, j0t, j0p = load_basefield(basename,nr,ntheta,nphi)

NR, Ek, Ra, Pr, Pm, minc, mcl, fi, rf = load_dimensionless(run_ID, directory)

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

#-- grid mesh helpers
grid = [radius, theta, phir]
r3D = np.zeros((nphir, ntheta, nr),)
for i in range(nr):
    r3D[:,:,i] = radius[i]

th3D = np.zeros_like(r3D)
for j in range(ntheta):
    th3D[:,j,:] = theta[j]

sint3D = np.sin(th3D)
cost3D = np.cos(th3D)
s3D = r3D*sint3D
grid_help = [r3D, sint3D, s3D]

#-- prepare sh_struct for Spectral transforms
print('call get curl, l_spectral = ', l_spectral)
if ( l_spectral ):
    import shtns
    if ( l_trunc ):
        l_max = 16
        m_max = 16
    else: # NOTE: WARNING!!: need to get the l_max/m_max of the simulation
        l_max = 133# nphir//3
        m_max = 48# l_max# nphi//3#
        if ( l_resym and minc !=1 ):
            m_max = nphir//3#--> 48
        else:
            m_max = nphi//3#--> 16
        #m_max = 133
    #
    sh = shtns.sht(l_max, m_max)
    nlat, nlon = sh.set_grid(nphi=nphir, nlat=ntheta)
else:
    sh = None

#-- Colorbar and s-sampling rate for plots
if ( run_Ek == '1e8' ):
    rprate = 2
else:
    rprate = 1

#------------------------------------------------------------------------------
#%% Read Data

#-- Time and Forces Redimensionalisation, if needed
#-- read the columnar averaged forces directly from a save file or re-compute them
if ( l_read ):
    #-- Time and Forces Redimensionalisation, if needed
    if ( l_tau_A ):
        tredim = 1./(Ek/Lehnert)
        Lordim = 1./(Pm*Lehnert)
        Cordim = 1./(Lehnert)
    else:
        tredim = 1.#/(Ek/Lehnert)*1/(Lehnert/Ek)
        Lordim = 1./(Pm*Ek)#/(Pm*Lehnert)*(Lehnert/Ek)
        Cordim = 1./Ek#/(Lehnert)*(Lehnert/Ek)
    if ( l_read_allphi ):
        #import h5py
        if ( l_field == 'vortz' ):
            fztitle = 'omzcol'
        elif ( l_field == 'Corz' ):
            fztitle = 'Corcol'
        elif ( l_field == 'Lorz' ):
            fztitle = 'Lorcol'
        f = h5py.File(saveDir+'Zavg_bal_m'+tagRead+'.hdf5', 'r')
        print('Reading columnar balances with Fields: %s' % f.keys())
        fkeys = list(f.keys())
        sr = np.array(f['srad'])
        timet = np.array(f['time'])
        upcol = np.array(f['upcol'])
        fieldzcol = np.array(f[fztitle])
        f.close()
        #
        if ( l_noIC ):
            #-- exluding IC
            upcol = upcol[... ,sr>=radius[0]]
            fieldzcol = fieldzcol[... ,sr>=radius[0]]
            sr = sr[sr>=radius[0]]
        if ( l_redim ):
            if ( l_field == 'Lorz' ):   fieldzcol=fieldzcol*Lordim#/(Pm*Lehnert)
            elif ( l_field == 'Corz' ):    fieldzcol=fieldzcol*Cordim#/(Lehnert)
    else:
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
        f.close()
        if ( l_redim ):
            Lorcol=Lorcol*Lordim#/(Pm*Lehnert)
            Corcol=Corcol*Cordim#/(Lehnert)
            #-- New standard from March 10 2023:
            #--     d . /dt computed using timet (adimensionalised as in Parody): has to be re-dimensionaliased here as well
            dVpcol=dVpcol/tredim
            domzcol=domzcol/tredim
    #
    nsmax = len(sr)
    tdim = timet*tredim#/(Ek/Lehnert)

#-- read Alfven speeds for wave speed comparison
if ( l_read_Va ):
    srja=np.linspace(radius[0],radius[-1],len(sr[sr>=radius[0]]))#201) #--> rebuild radius like in curlbal_QGA_4_JA.m
    #-- Normal Va
    dirva = '/Users/obarrois/Desktop/stella/Work/Waves'+run_Ek+'/'
    try:
        Valfven = np.loadtxt(dirva+'Va')
        l_plot_Va = True
    except FileNotFoundError:
        print("Alfven speed: File Not Found")
        Valfven = np.zeros((nr),)
        l_plot_Va = False
    trajcolor='#32CD32' # Lime # '#99CC32' # Jaune-Vert # '#FF7F00' # Orange
    #
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
        if ( rtraj[istep] > 0.99*radius[-1] ): #if rtraj(step)>0.99*r(nr);
            advtime = 0.
        ttraj[istep] = ttraj[istep-1] + advtime #ttraj(step)=ttraj(step-1)+advtime;
    #
    #-- Adimensionalising the axis because plt.imshow is not entirely satisfiying!
    tjplot = (len(tdim[::-1])-1)-(ttraj - ttraj[0])/ tdim.max()*len(tdim[::-1])
    rjplot = (rtraj - rtraj[0])/(rtraj - rtraj[0]).max()*(len(srja[::rprate])-1)
    #
    #-- Fast Va
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
    #
    n_Asteps = 25000
    rtraj = np.zeros((n_Asteps),); ttraj = np.zeros_like(rtraj)
    #rtraj[0] = srja[0]; ttraj[0] = tdim[0]
    rtraj[0] = 0.54; ttraj[0] = 0.
    advtime = 1.e-2
    for istep in range(1,n_Asteps):
        i = np.sum(radius<=rtraj[istep-1]) #i=sum(r<=rtraj(step-1));
        j = np.sum(tdim<=ttraj[istep-1]) #j=sum(tt<=ttraj(step-1));
        Val = Valfven[i]
        #
        #print(istep, i, j, Val, rtraj[istep-1])
        rtraj[istep] = rtraj[istep-1] + Val*advtime #rtraj(step)=rtraj(step-1)+Val*advtime;
        if ( rtraj[istep] > 0.99*radius[-1] ): #if rtraj(step)>0.99*r(nr);
            advtime = 0.
        ttraj[istep] = ttraj[istep-1] + advtime #ttraj(step)=ttraj(step-1)+advtime;
    #
    #-- Adimensionalising the axis because plt.imshow is not entirely satisfiying!
    tjplot2 = (len(tdim[::-1])-1)-(ttraj - ttraj[0])/ tdim.max()*len(tdim[::-1])
    rjplot2 = (rtraj - rtraj[0])/(rtraj - rtraj[0]).max()*(len(srja[::rprate])-1)

#-- Plot s-z-avg-profiles of different force balances as a function of time
#if ( l_check_all and rank==0 ):
if ( l_read_allphi ):
    analys = upcol[:,phi_sample,:].copy()# fieldzcol[:,phi_sample,:].copy()# Vpcol[:,:].copy()#
else:
    analys = upcol[:,:].copy()# fieldzcol[:,:].copy()# Vpcol[:,:].copy()#
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
#-- Field
pmax = np.amax(abs(analys[:,::n_steps]))/2.
plt.figure()
plt.imshow(analys[::-1,::n_steps], vmin=-pmax, vmax=pmax, cmap=cmo.balance)
plt.title('z-Avg uphi')
plt.ylabel('Time')
plt.xlabel('Cylindrical radius')
plt.xticks(x_post,x_axis)
plt.yticks(y_post,y_axis)
plt.colorbar(shrink=0.5, orientation='vertical')
if ( l_save ):
    plt.close('all')
else:
    plt.show()

#-- Plot azimuthal-z-avg-profiles of different force balances as a function of time
#if ( l_check_all and rank==0 ):
if ( l_read_allphi ):
    rplot=nr//4#*3//4#
    analys = upcol[:,:,rplot].copy()#fieldzcol[:,:,rplot].copy()#
    #analys = upcol.mean(axis=-1)#fieldzcol.mean(axis=-1)#
    #analys = -simps(upcol *hsr*sr, sr)#-simps(fieldzcol *hsr*sr, sr)#
    nysteps=2
    ny_labels = 6
    ny = tdim[::nysteps].shape[0]
    y_steps = int(ny /(ny_labels - 1))
    y_post = np.arange(0,ny,y_steps)
    y_axis = np.round(tdim[::nysteps][::-y_steps], 4) # to get time going up
    nxsteps=1
    nx_labels = 5
    nx = phir[::nxsteps].shape[0]
    x_steps = int(nx/(nx_labels - 1))
    x_post = np.arange(0,nx,x_steps)
    x_axis = np.round(phir[::nxsteps][::x_steps]*180./np.pi,1)
    #-- Field
    pmax = np.amax(abs(analys[:,::nxsteps]))/2.
    plt.figure()
    plt.imshow(analys[::-nysteps,::nxsteps], vmin=-pmax, vmax=pmax, cmap=cmo.balance)
    #plt.title('z-Avg '+fztitle)
    plt.title('z-Avg uphi')
    plt.ylabel('Time')
    plt.xlabel('Azimuth')
    plt.xticks(x_post,x_axis)
    plt.yticks(y_post,y_axis)
    plt.colorbar(shrink=0.5, orientation='vertical')
    if ( not l_save ):  plt.show()

#------------------------------------------------------------------------------
#%% Additional computations: wa

#-- compute the theoretical wave-dispersion relations from (Gillet et al. 2022)
if ( l_build_wave_disp ):
    from parobpy.core_properties import earth_radius
    #
    #-- Prepare velocity, height and Omega
    srja=np.linspace(radius[0],radius[-1],len(sr[sr>=radius[0]]))#201) #--> rebuild radius like in makeB.m
    Valfmean = np.loadtxt(dirva+'Va') # NOTE: Val defined with radius from parody: not practical to use sr or cylindrical radius
    #hs = np.sqrt(radius[-1]**2 - (radius*np.sin(theta[ntheta//2]))**2) #np.sqrt(sr[0]**2 - sr[1:]**2)
    hs = np.sqrt(srja[-1]**2 - srja[1:]**2)
    Omega=Valfmean/(hs*Lehnert)#1./Lehnert#
    #
    #-- Alfven waves --> actually mixed between Aflven and MC waves
    k_wave = 32; m_wave = 6
    omVa = Valfmean*k_wave + m_wave*Omega/(k_wave**2 * hs**2)
    omMiC= m_wave*Omega/(k_wave**2 * hs**2) + np.sqrt( (Valfmean*k_wave)**2 + (m_wave*Omega/(k_wave**2 * hs**2))**2 )
    k_max = 71; m_max = 48
    k_0 = np.zeros((m_max,len(Valfmean)),)
    omMiT = np.zeros((m_max,k_max,len(Valfmean)),); omMiT2 = np.zeros_like(omMiT)
    for m_wave  in range(m_max):
        k_0[m_wave] = (m_wave*Valfmean/(hs*Lehnert)/(Valfmean*hs**2))**(1/3)
        for k_wave  in range(k_max):
            omMiT[m_wave,k_wave] = m_wave*Omega/(k_wave**2 * hs**2) + np.sqrt( (Valfmean*k_wave)**2 + (m_wave*Omega/(k_wave**2 * hs**2))**2 )
            omMiT2[m_wave,k_wave] = Valfmean*k_wave*((k_0[m_wave]/k_wave)**3 + np.sqrt( 1. + (k_0[m_wave]/k_wave)**6 )) #Eq.20 = rephrasing of Eq.19
    #
    if ( l_check_all ):
        m_wave = 6
        pulse_rescale = 1.#/(2.*np.pi)
        plt.figure()
        plt.plot(np.arange(1,k_max), omMiT[m_wave,1:,nr//4]*pulse_rescale, 'g-o', lw=2.1, label='Mixed pulsation at radius = '+str(np.round(radius[nr//4], 3)))
        plt.plot(np.arange(1,k_max), omMiT2[m_wave,1:,nr//4]*pulse_rescale, 'r--o', lw=2.1,alpha=0.6, label='Other pulsation at radius = '+str(np.round(radius[nr//4], 3)))
        plt.plot(np.arange(1,k_max), omMiT[m_wave,1:,nr//2]*pulse_rescale, 'g-*', lw=2.1, label='Mixed pulsation at radius = '+str(np.round(radius[nr//2], 3)))
        plt.plot(np.arange(1,k_max), omMiT2[m_wave,1:,nr//2]*pulse_rescale, 'r--*', lw=2.1,alpha=0.6, label='Other pulsation at radius = '+str(np.round(radius[nr//2], 3)))
        plt.plot(np.arange(1,k_max), omMiT[m_wave,1:,nr*3//4]*pulse_rescale, 'g-d', lw=2.1, label='Mixed pulsation at radius = '+str(np.round(radius[nr*3//4], 3)))
        plt.plot(np.arange(1,k_max), omMiT2[m_wave,1:,nr*3//4]*pulse_rescale, 'r--d', lw=2.1,alpha=0.6, label='Other pulsation at radius = '+str(np.round(radius[nr*3//4], 3)))
        plt.xlabel('radial wave number')
        plt.ylabel('Wave pulsation')
        plt.xlim(1,k_max-1)
        plt.ylim(-100.,100.)
        plt.grid(ls=':')
        plt.legend()
        #
        rad_wave = nr//2
        plt.figure()
        plt.contourf(np.arange(1,k_max), np.arange(1,m_max), omMiT[1:,1:,rad_wave]*pulse_rescale, cm=cmo.thermal)
        plt.xlabel('radial wave number')
        plt.ylabel('azimuthal wave number')
        #plt.show()
        #
        #-- Save Theoretical Dispersion relations
        if ( l_save and rank==0 ):
            size_plot = (10.2, 10.8)
            plt.figure(1)
            plt.gcf().set_size_inches(size_plot, forward=False)
            plt.savefig(savePlot+'QGMC-Waves_Pulsation-vs-radial-wave-number.png')
            plt.close('all')


#-------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------
#%% Main plot and computations section

#-- Import packages
from pyhank import qdht #NOTE: Warning; if using Hankel Transform: better to use the full shell z-avg (stored in 'Zavg_bal_m'+tagRead+'.hdf5')
from vizuals import spat_spec
from scipy.fftpack import fft, dct
from scipy.special import jv
from scipy.integrate import simps

#------------------------------------------------------------------------------
#%% Extracting dispersion relation using FFT and DCT or Hankel !!! Plot 'FK' (frequency versus k-number) diagrams

#-- Extand time up to 2pi if needed
rtxtd = 0; ntpi=0
while ( rtxtd<1 ):
    ntpi+=1
    rtxtd = ntpi*2.*np.pi/tdim[-1]
#
ntxtd = int(len(tdim)*rtxtd)+1
ntend = len(tdim)+(ntxtd - len(tdim))//2
tdxtd = np.linspace(0,ntpi*2.*np.pi,ntxtd)

nlim = 1; nslim = 1
analys = upcol.copy()#fieldzcol.copy()#
if ( l_read_allphi ):
    readfield = analys.copy()
else: #phi-dimension is missing in the array and that will cause problems in the following
    readfield = np.zeros((len(tdim),nphir,len(sr)),)
    readfield[:,phi_sample,:] = analys.copy()
if ( l_txtd ):
    tana = tdxtd.copy()
    fieldxtd = np.zeros((ntxtd, readfield.shape[1], readfield.shape[2]),)
    fieldxtd[ntend-len(tdim):ntend] = readfield.copy()
else:
    tana = tdim.copy()
    fieldxtd = readfield.copy()
ntheb = fieldxtd.shape[0]*2//3; nmmax = fieldxtd.shape[1]//3; ncheb = len(sr)*2//3 #NOTE: cut of the DCT seems important for the final power spectrum!
#ntheb = fieldxtd.shape[0]; nmmax = nphir; ncheb = len(sr)
wllin = np.arange(1,ntheb+1); mllin = np.arange(1,nmmax+1); kllin = np.arange(1,ncheb+1)

#-- Compute wave number or use wavelenght for axis instead?
if ( l_lambda ):
    wlin = wllin
    mlin = mllin
    klin = kllin
else:
    if ( l_sqdht ):
        klin = 2.*np.pi/((sr[nslim] - sr[-nslim])/kr)
        #klin = 1./((sr[nlim] - sr[-nlim])/kr)
    else:
        klin = 2.*np.pi/((sr[nslim] - sr[-nslim])/np.arange(1,ncheb+1))
    wlin = 2.*np.pi/((tana[-nlim] - tana[nlim])/np.arange(1,ntheb+1))
    mlin = 2.*np.pi/((phir[-1] - phir[0])/np.arange(1,nmmax+1))

#-- True Full field!!
#-- All times and all s AND all PHI
if ( l_all_phi ):
    field_T = np.zeros((fieldxtd.shape[0],nphir,len(sr)),dtype=np.complex128)
    field_M = np.zeros_like(field_T)
    field_C = np.zeros_like(field_T)
    field_MC = np.zeros_like(field_T)
    field_TMC = np.zeros_like(field_T)
    for i in range(nslim,len(sr)-nslim):
        for n_phi in range(nphir):
            field_T[:ntheb,n_phi,i] = spat_spec(fieldxtd[:,n_phi,i],ntheb)
            field_TMC[:ntheb,n_phi,i] = spat_spec(fieldxtd[:,n_phi,i],ntheb)
        #
        for n_t in range(nlim,len(tana)-nlim):
            field_M[n_t,:nmmax,i] = spat_spec(fieldxtd[n_t,:,i],nmmax)
            field_MC[n_t,:nmmax,i] = spat_spec(fieldxtd[n_t,:,i],nmmax)
            field_TMC[n_t,:nmmax,i] = spat_spec(field_TMC[n_t,:,i],nmmax)
    #
    for n_phi in range(nphir):
        for n_t in range(nlim,len(tana)-nlim): #NOTE: Warning: FFT does not work properly on s (non-periodic + boundaries)
            if ( l_sqdht ):
                kr, field_C[n_t,n_phi,:] = qdht(sr, fieldxtd[n_t,n_phi,:])
                kr, field_MC[n_t,n_phi,:] = qdht(sr, field_MC[n_t,n_phi,:])
                kr, field_TMC[n_t,n_phi,:] = qdht(sr, field_TMC[n_t,n_phi,:])
            else:
                field_C[n_t,n_phi,:ncheb] = dct(fieldxtd[n_t,n_phi,:],type=1,n=ncheb)#spat_spec(analys[j,:],ncheb)#
                field_MC[n_t,n_phi,:ncheb] = dct(field_MC[n_t,n_phi,:],type=1,n=ncheb)#spat_spec(analys[j,:],ncheb)#
                field_TMC[n_t,n_phi,:ncheb] = dct(field_TMC[n_t,n_phi,:],type=1,n=ncheb)#spat_spec(analys[j,:],ncheb)#
            #
    #
    field_T2 = (field_T*field_T.conjugate()).real
    field_M2 = (field_M*field_M.conjugate()).real
    field_C2 = (field_C*field_C.conjugate()).real
    field_MC2 = (field_MC*field_MC.conjugate()).real
    field_TMC2 = (field_TMC*field_TMC.conjugate()).real
    #
    #--Integrations (time and phi are simple averages)
    #-- Simps integ in s --> -1 sign from integration in radial direction
    hsr = np.sqrt(sr[0]**2 - sr**2)
    spec_T2 = -simps(field_T2.mean(axis=1)*2.*np.pi *hsr*sr, sr)
    spec_M2 = -simps(field_T2.mean(axis=0) *hsr*sr, sr)
    spec_C2 = (field_C2.mean(axis=0)).mean(axis=0)*2.*np.pi
    #
    #-- Convolution product of Energies to reconstruct 2D spectrum?
    vomk = np.zeros_like(field_T2)
    for i in range(1,len(sr)):
        for m in range(1,len(phir)):
            for j in range(1,len(tana)):
                vomk[j,m,i] = spec_T2[j]*spec_M2[m]*spec_C2[i] #NOTE: Smooth a lot compared to several spectra in a row
else:
    field_T = np.zeros((fieldxtd.shape[0],len(sr)),dtype=np.complex128)
    field_C = np.zeros_like(field_T)
    field_TC = np.zeros_like(field_T)
    for i in range(nslim,len(sr)-nslim):
        field_T[:ntheb,i] = spat_spec(fieldxtd[:,phi_sample,i],ntheb)
        field_TC[:ntheb,i] = spat_spec(fieldxtd[:,phi_sample,i],ntheb)
    #
    for n_t in range(nlim,len(tana)-nlim): #NOTE: Warning: FFT does not work properly on s (non-periodic + boundaries)
        if ( l_sqdht ):
            kr, field_C[n_t,:] = qdht(sr, fieldxtd[n_t,phi_sample,:])
            kr, field_TC[n_t,:] = qdht(sr, field_TC[n_t,:])
        else:
            field_C[n_t,:ncheb] = dct(fieldxtd[n_t,phi_sample,:],type=1,n=ncheb)#spat_spec(analys[j,:],ncheb)#
            field_TC[n_t,:ncheb] = dct(field_TC[n_t,:],type=1,n=ncheb)#spat_spec(analys[j,:],ncheb)#
        #
    #
    field_T2 = (field_T*field_T.conjugate()).real
    field_C2 = (field_C*field_C.conjugate()).real
    field_TC2 = (field_TC*field_TC.conjugate()).real
    #
    #--Integrations (time and phi are simple averages)
    #-- Simps integ in s --> -1 sign from integration in radial direction
    hsr = np.sqrt(sr[0]**2 - sr**2)
    spec_T2 = -simps(field_T2 *hsr*sr, sr)#field_T2.sum(axis=1)#
    spec_C2 = field_C2.mean(axis=0)#field_C2.sum(axis=0)#
    #
    #-- Convolution product of Energies to reconstruct 2D spectrum?
    vomk = np.zeros_like(field_T2)
    for i in range(1,len(sr)):
        for j in range(1,len(tana)):
            vomk[j,i] = spec_T2[j]*spec_C2[i] #NOTE: Smooth a lot compared to several spectra in a row

#------------------------------------------------------------------------------
#%% Main plot section

plt.rcParams['text.usetex'] = True
#-- Fig. Barrois and Aubert 2024
#-- Plot 'FK' (frequency versus k-number) diagram with theoretical dispersion relation
m_wave = 6
plim = 50#100#400#
cmax = abs(field_TMC2.sum(axis=1)[:plim,:plim]).max()*0.92
llevels=np.linspace(0,cmax,64)#16)
clevels=np.linspace(0,cmax,7)
#
fig = plt.figure(figsize=(16, 12.5))
ax = plt.subplot(111)
cf = ax.contourf(klin[:plim],wlin[:plim],field_TMC2.sum(axis=1)[:plim,:plim], levels=llevels,cmap='CMRmap')#cmo.thermal)# #NOTE: Plot "_Integ"
if ( l_build_wave_disp and not l_lambda ):
    ax.plot(np.arange(20,70), omMiT[m_wave][20:70,nr//4], 'r--*', lw=1.9, alpha=0.8,label=r'QG-A-MC pulsation (eq.~18) at $s = {:.3f}$ '.format(radius[nr//4]))
    ax.plot(np.arange(20,70), omMiT[m_wave][20:70,nr//2], 'c--*', lw=1.9, alpha=0.8,label=r'... at $s = {:.3f}$ '.format(radius[nr//2]))
    ax.plot(np.arange(20,70), omMiT[m_wave][20:70,nr*3//4], 'b--*', lw=1.9, alpha=0.8,label=r'... at $s = {:.3f}$ '.format(radius[nr*3//4]))
if ( l_lambda ):
    plt.xlabel('lambda_s', fontsize=36)#
    plt.ylabel('lambda_t', fontsize=36)#
else:
    plt.xlabel(r'Cylindrically radial wavenumber, $k$', fontsize=36)#
    plt.ylabel(r'Wave period, $\omega$', fontsize=36)#
if ( l_build_wave_disp and not l_lambda ):   plt.legend(fontsize=32)
plt.xscale('log')
plt.yscale('log')
plt.gca().xaxis.set_tick_params(which='major', labelsize=32)
plt.gca().xaxis.set_tick_params(which='minor', labelsize=32)
plt.gca().yaxis.set_tick_params(which='major', labelsize=32)
plt.gca().yaxis.set_tick_params(which='minor', labelsize=32)
cb = fig.colorbar(cf, ax=ax, fraction=0.04, pad=0.05, orientation='vertical', ticks=clevels)#, format=r'${x:.1e}$')
#cb.set_label(label=r'Spectrum density, $\sum_m {\tt DCT}_s \lbrace {\tt FFT}_\phi \left[ {\tt FFT}_t \left( \left< \tilde{u_\phi} \right>^2 \right) \right] \rbrace$',size=36)#, weight='bold')
cb.set_label(label=r'Spectrum modulus',size=36)#, weight='bold')
cb.ax.tick_params(labelsize=32)
plt.tight_layout()
if ( not l_save ):  plt.show()

#-- Save True Fields analysis
if ( l_save and rank==0 ):
    plt.figure(1)
    plt.savefig(savePlot+'QGMC-Waves-Diagram-SumPhi_Pulsation-vs-radial-wave-number_uphicol_FFT+DCT_Integ_ncheb-ntheb=lower.png')
    #plt.close('all')

#------------------------------------------------------------------------------
#%% Aggregated plots for paper

#-- Fig. Barrois and Aubert 2024
#-- Aggregated plot m-vs-s at different times
if ( l_all_phi ):
    eps = 1.e-3
    nplots = 4
    t_pick = np.array([99, 199, 299, 399, 498])
    xlevels=(0.6, 0.9, 1.2, 1.5)
    #
    k=-1
    #fig = plt.figure(figsize=(21, 7.4)) #NOTE: cannot unpack non-iterable Figure object
    fig, axs = plt.subplots(1, 5, figsize=(21, 7.4))#figsize=(22.1, 7.4))#, sharex=True)#, layout='constrained')
    for n_t in t_pick:
        cmax = 1. #abs(np.sqrt(field_M2[n_t,:nmmax,:])).max()*0.98#
        llevels=np.linspace(0.,cmax,64)
        clevels=np.linspace(0.,cmax,5)
        k+=1
        ax = plt.subplot2grid((1,nplots+1), (0,k))
        axs[k] = plt.subplot2grid((1,nplots+1), (0,k))
        if ( not l_snorm ):
            cf = axs[k].contourf(sr,mllin,np.sqrt(field_M2[n_t,:nmmax,:]),levels=llevels, extend='both',cmap='CMRmap')#cmap=cmo.solar)#
        else:
            cf = ax.contourf(sr,mllin,np.sqrt((field_M2[n_t,:nmmax,:]/field_M2[n_t,:nmmax,:].max(axis=0))),levels=15)
        plt.xlabel(r'Cylindrical radius, $s$', fontsize=36)
        if ( not l_snorm ):
            if (k==0):  plt.ylabel(r'Azimuthal wavenumber, $m$', fontsize=36)
        else:
            if (k==0):  plt.ylabel(r'M fft - $s$-Normalised')
        plt.xticks(xlevels)
        plt.gca().xaxis.set_tick_params(labelsize=32)
        plt.gca().yaxis.set_tick_params(labelsize=32)
        plt.title(r'time $t = {:.2f}$'.format(tana[n_t]), fontsize=32)
        cb = fig.colorbar(cf, ax=axs[k], fraction=0.05, pad=0.16, orientation='horizontal', ticks=clevels, format=r'${x:.1f}$')
        cb.ax.tick_params(labelsize=32)
    plt.tight_layout()
    plt.show()
    #
    k=-1
    #fig = plt.figure(figsize=(20, 6.5))
    fig = plt.figure(figsize=(21, 7.4))
    for n_t in t_pick:
        cmax = 4. #abs(fieldxtd[n_t,:,:]).max()*0.92
        llevels=np.linspace(-cmax,cmax,64)
        clevels=np.linspace(-cmax,cmax,5)
        k+=1
        ax = plt.subplot2grid((1,nplots+1), (0,k))
        if ( not l_snorm ):
            if ( l_deminc ):
                cf = ax.contourf(sr,phir,fieldxtd[n_t,:,:],levels=llevels,extend='both',cmap=cmo.balance)#'seismic')#cmperso)#
            else:
                cf = ax.contourf(sr,phi,fieldxtd[n_t,:nmmax,:],levels=llevels,extend='both',cmap='seismic')#cmo.balance)#
        else:
            if ( l_deminc ):
                cf = ax.contourf(sr,phir,(fieldxtd[n_t,:,:]/abs(fieldxtd[n_t,:,:]).max(axis=0)),vmin=-1.,vmax=1.02,levels=32,cmap='seismic')#cmo.balance)#
            else:
                cf = ax.contourf(sr,phi,(fieldxtd[n_t,:nmmax,:]/abs(fieldxtd[n_t,:nmmax,:]).max(axis=0)),vmin=-1.,vmax=1.02,levels=32,cmap='seismic')#cmo.balance)#
        plt.xlabel(r'Cylindrical radius, $s$', fontsize=36)
        if ( not l_snorm ):
            if (k==0):  plt.ylabel(r'Azimuth, $\phi$', fontsize=36)
        else:
            if (k==0):  plt.ylabel('azimuth $\phi$ - $s$-Normalised')
        plt.xticks(xlevels)
        plt.gca().xaxis.set_tick_params(labelsize=32)
        plt.gca().yaxis.set_tick_params(labelsize=32)
        plt.title(r'time $t = {:.2f}$'.format(tana[n_t]), fontsize=32)
        cb = fig.colorbar(cf, ax=ax, fraction=0.05, pad=0.16, orientation='horizontal', ticks=clevels, format=r'${x:.0f}$')
        cb.ax.tick_params(labelsize=32)
    plt.tight_layout()
    plt.show()
    #
    #-- Save True Fields analysis
    if ( l_save and rank==0 ):
        plt.figure(1)
        plt.savefig(savePlot+'Azimuth-Radius-Diagram_M-FFT_uphi-Col.png')
        plt.figure(2)
        plt.savefig(savePlot+'Azimuth-Radius-Diagram_Minc_uphi-Col.png')
        plt.close('all')

#-- End Script
