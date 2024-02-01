#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon June 22 2023
@author: obarrois

Python script to compute curl balances of either coriolis, lorentz or vorticity.
Then zAvg the quantities and store them as time/cylindrical radii (full disk) matrices.

NOTE: Possible to use mpi to speed up curl calculations but one has to call the script as:
    $ mpiexec -n X python3 Compute_Zavg_Allphi.py 
"""

from vizuals import zavg
from parobpy.parob_lib import curl_sph
from parobpy.load_parody import parodyload, load_dimensionless, load_basefield, list_Gt_files
import numpy as np

#----------------------------------------------------------------------------%%
#-- INPUT PARAMETERS

#-- Lehnert number, \lambda = B/{sqrt(rho mu) * Omega*d}.
run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '3e7', 'Pm0o25', 1.46e-3, 1.0, 'b3', '0.000280800' # 3e-7 S=1214.1
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1', 1.1e-3, 1.0, 'b4', '0.000040320' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1Long', 1.1e-3, 1.0, 'b4-long', '0.000614400' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'PathLund3o2e3', 5.53e-4, 1.0, 'b4o6', '0.000015600' # 6.3e-9 (1e7 grid)
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Bench_Pulse-Time', 1.1e-3, 1.0, 'b4-bis', '0.000080000' # 1e-7 #, 'Pm1o44e-1_Bench'
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e8', 'Pm0o46e-1', 6.2e-4, 1.0, 'b4o5', '0.000021840' # 1e-8
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e8', 'Pm0o46e-1BIS', 6.2e-4, 1.0, 'b4o5', '0.000024896' # 1e-8
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '6e9', 'Pm0o36e-1', 5.53e-4, 1.0, 'b4o63', '0.000015560' # 6.3e-9
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '6e9', 'Pm0o36e-1BIS', 5.53e-4, 1.0, 'b4o63B', '0.000019760' # 6.3e-9
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '3e10', '', 2.6e-4, 1.0, 'b5', '0.000001456' # 3e-10

#saveDir = '/gpfs/users/obarrois/Parodys/Outputs/Data/' # if you use python directly from cluster
#directory = '/gpfs/users/obarrois/Work/Waves1e7/' # if you use python directly from cluster
#directory = '/Users/obarrois/Desktop/dfgnas3/Waves/Data_3e10/' # run at 3e-10 on jaubert disk
directory = '/Users/obarrois/Desktop/stella/Work/Waves'+run_Ek+'/'+run_Pm+'/' # if you mount stella on own desk

#-- phi-Avg of field - phi-Avg gives only noise. Otherwise the rest is fine.
tstart = 0 # time rank of the first Gt file read. Should be <= ntime and can be mofidify to scan != samples with t_srate
t_srate = 1 # sampling rate of Gt files if needed. Should be <= ntime
phi_sampling = 1 # longitude to extract for time sampling

rank = 0 # default proc number when l_mpi == False
l_mpi = True # use MPI to speed up horizontal gradients for the curl computation or not

l_field = 'Lorz' #field to be curled, transformed and plotted: can be either 'omz', 'Corz' or 'Lorz'
l_redim = True # re-dimensionalisation of the quantities
l_rebuild = False # rebuild field from sinusogramm (Radon transform output)

l_spectral = True # using spectral derivatives in theta and phi for curl? (advised:  = True)
l_trunc = False # truncation for SH?

l_noIC = False # Excluding z-avg fields inside of the TC?
l_resym = True # re-symmetrizing the data if minc !=1
l_back_b0 = False # add back the background magnetic field b0

#-- I/O; NOTE: does not seem to replace files if they exist even if l_save*!
l_save = 1 # save zAvg-balances?
l_zavg = 1 # using python method to compute z-avg of the forces
saveDir = '/Users/obarrois/Desktop/Parodys/Outputs/Waves'+run_Ek+'/Balances/Data/' # path to save colmunar balance files
#saveDir = '/Users/obarrois/Desktop/stella/Parodys/Outputs/Data/' # if you use python from perso to remote data directly from cluster

#------------------------------------------------------------------------------
#%% Initialisation

if ( l_save ):
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
    if ( False ):#if ( not l_resym and minc !=1 ): # Test to get tab2 like in curlbal_QGA_4_JA.m
        sh.imtab = sh.m.copy()
        for lm in range(sh.nlm):
            if ( np.mod(lm,3) != 0 ):
                sh.imtab[lm] = 0
        sh.m = sh.imtab
else:
    sh = None

#------------------------------------------------------------------------------
#%% Loop over time samples to compute curls

#-- former implementation with reading everything before curling
#from parobpy.load_parody import load_parody_serie
#if ( l_sample ):
#    _, _, _, timet, urt, utt, upt, Brt, Btt, Bpt, _ = load_parody_serie(run_ID, directory,t_srate) # sample some of the Gt files (with a rate of t_srate)
#else:
#    _, _, _, timet, urt, utt, upt, Brt, Btt, Bpt, _ = load_parody_serie(run_ID, directory) # load all Gt files --> can a be a lot of files
#n_samples = len(timet)

Gt_file_l = list_Gt_files(run_ID,directory) # Find all Gt_no in folder
n_samples = len(Gt_file_l[tstart::t_srate])

rrad = radius[::-1] # zavg() expect reversed radius
#nsmax = 309 #--> to get nsmax ~ nsmaxja
nsmax = int(nr*1.5 + 17) # to maintain nsmax ~ nrmax outside of TC (but will give nsmax > nrmax)
#Vp = np.zeros((n_samples, nphir//phi_sampling, ntheta, nr),)
#ups = np.zeros_like(Vp)
#fieldzs = np.zeros_like(ups)
upcol = np.zeros((n_samples, nphir//phi_sampling, nsmax),) #, 617),) #401),) #401 = length of r_s for case at Ek=1e-7 (can not be really known before z-Avg or a preliminary test)
fieldzcol = np.zeros_like(upcol)
timet = np.zeros((n_samples),)#-- Better to do all the redimensionalisation in the plot script: new standard from March 10 2023

n_s=-1
#-- Start loop
for file in Gt_file_l[tstart::t_srate]:#[249:250]:#
    n_s+=1
    #------------------------------------------------------------------------------
    #-- loading pointtime data
    print('Loading {} (({}/{})/{})'.format(file, n_s+1, n_samples, len(Gt_file_l)))
    filename = '{}/{}'.format(directory,file)
    (_, time, _, _, _, _, _,
        _, _, _, _, _, _, _, _,
        _, _, _, _, _, _, _, ur, ut, up,
        Br, Bt, Bp, _) = parodyload(filename)
    timet[n_s] = time
    #------------------------------------------------------------------------------
    #-- post processing if needed
    #
    #-- resymmetrize the data if minc > 1
    if ( l_resym and minc !=1 ):
        print('Re-symmetrizing the data for file n°{}, t={}'.format(n_s+1, time))
        Br = symmetrize(Br,ms=minc); ur = symmetrize(ur,ms=minc)
        Bt = symmetrize(Bt,ms=minc); ut = symmetrize(ut,ms=minc)
        Bp = symmetrize(Bp,ms=minc); up = symmetrize(up,ms=minc)
    #
    #-- Adding back the background magnetic field B0 to B
    if ( l_back_b0 ):
        # NOTE: no u0 for velocity so no need to add anything for u
        print('Adding background field for sample n°{}/{}'.format(n_s+1, n_samples))
        Br = Br + B0r
        Bt = Bt + B0t
        Bp = Bp + B0p
    #
    #------------------------------------------------------------------------------
    #-- Compute curls
    #
    #-- Computation of point time Forces
    print('Computing Force bal for file n°{}/{}, t={}, rank={}'.format(n_s+1, n_samples, time, rank))
    #
    if ( l_field == 'vortz' ):
        fztitle = 'omzcol'
        #-- Vorticity: Vxu
        fr = ur.copy(); ft = ut.copy(); fp = up.copy()
    elif ( l_field == 'Corz' ):
        fztitle = 'Corcol'
        #-- Coriolis: -2*u x Omega
        fr = 2.*-sint3D*up#up #corr=2*-mmsint.*Vp;
        ft = 2.*-cost3D*up#up #cort=2*-mmcost.*Vp;
        fp = 2.*(sint3D*ur + cost3D*ut) #corp=2*(mmcost.*Vt+mmsint.*Vr); #(sint3D*ur + cost3D*ut)
    elif ( l_field == 'Lorz' ):
        fztitle = 'Lorcol'
        #-- Curl B: j = VxB
        jr, jt, jp = curl_sph(grid_help, Br, Bt, Bp, l_mpi=l_mpi, l_spectral=l_spectral, sh_struct=sh)
        if ( np.sum(np.isnan(jr)) + np.sum(np.isnan(jt)) + np.sum(np.isnan(jp)) != 0 ): print('Warning!:: nan in curl B computations!') # to check if there are any problems
        #-- Curl B cross B (magnetic stress?): jxB
        fr = jt*B0p - jp*B0t + j0t*Bp - j0p*Bt # jt*Bp - jp*Bt + jt*Bp - jp*Bt #
        ft = jp*B0r - jr*B0p + j0p*Br - j0r*Bp # jp*Br - jr*Bp + jp*Br - jr*Bp #
        fp = jr*B0t - jt*B0r + j0r*Bt - j0t*Br # jr*Bt - jt*Br + jr*Bt - jt*Br #
        del(jr,jt,jp)
        #-- Lorentz force: VxjxB
    #--
    fieldr, fieldt, _ = curl_sph(grid_help, fr, ft, fp, l_mpi=l_mpi, l_spectral=l_spectral, sh_struct=sh)
    if ( np.sum(np.isnan(fieldr)) + np.sum(np.isnan(fieldt)) != 0 ): print('Warning!:: nan in curl u computations!') # to check if there are any problems #+ np.sum(np.isnan(omp)) 
    fieldz = cost3D*fieldr - sint3D*fieldt
    del(fieldr,fieldt)#,fieldp)
    del(fr,ft,fp)
    #
    #-- Compute phi average from fields
    Vp = up.mean(axis=0) #-- Zonal velocity
    fzmm = fieldz.mean(axis=0)
    #-- phi sampling: take only one longitude to save space and computation time
    up  = up[::phi_sampling] - Vp
    fieldz  = fieldz[::phi_sampling] - fzmm
    #
    #-- Compute z-Avg force balances
    if ( l_zavg ):
        # NOTE: WARNING!:: zavg() or zavgpy() expect reversed radius as in pizza/MagIC (r_cmb = r[0]) SO the arrays MUST BE REVSERED as well
        print('Computing Fortran z-avg of the Forces for all {} files'.format(n_samples))
        #hs, rs, Vpcols = zavg(Vp[..., ::-1], rrad, nsmax, colat=theta, minc=1, normed=True, save=False) #zavgpy(Vp[..., ::-1], rrad, nsmax, minc=1, normed=True)
        hs, rs, upcols = zavg(up[..., ::-1], rrad, nsmax, colat=theta, minc=1, normed=True, save=False)
        hs, rs, fieldzcols = zavg(fieldz[..., ::-1], rrad, nsmax, colat=theta, minc=1, normed=True, save=False)
        #-- exluding IC
        if ( l_noIC ):
            upcol[n_s] = upcols[:,rs>=radius[0]]; fieldzcol[n_s] = fieldzcols[:,rs>=radius[0]]
            sr = rs[rs>=radius[0]]
        else: #-- with IC
            upcol[n_s] = upcols[:,:]; fieldzcol[n_s] = fieldzcols[:,:]
            sr = rs[:]
        #
    #-- end z-Avg of forces
#-- end loop reading and computing forces

#-- save forces in a data file?
if( l_save and rank==0 ):
    with h5py.File('{}/Zavg_bal_m.hdf5'.format(saveDir), 'a') as f:
        f.create_dataset('time', data=timet)
        f.create_dataset('srad', data=sr)
        f.create_dataset('upcol', data=upcol)
        f.create_dataset(fztitle, data=fieldzcol)

#-- End Script
