#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 2023
@author: obarrois

Python version of PARODY-JA4.56-Base/Base_Field Matlab file 'Matlab/curlbal_QGA_4.m'.
Loads graphics file and compute curl balances of coriolis, lorentz and vorticity.

Script to compute ONLY the force balances (one needs to use an other script to plot them)!
NOTE: Possible to use mpi to speed up curl calculations but has to call the script as:
    $ mpiexec -n X python3 Compute_Curl_Bal.py 
"""

#import os
#import sys
from vizuals import zavg
#from parobpy.core_properties import icb_radius, cmb_radius
from parobpy.parob_lib import curl_sph#, zavgpy# get_curl, 
from parobpy.load_parody import parodyload, load_dimensionless, load_basefield, list_Gt_files
import numpy as np
#import shtns

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
l_sample = True # sample Gt time files instead of computing for every t=... (advised: could be really slow otherwise)
tstart = 0 # time rank of the first Gt file read. Should be <= ntime and can be mofidify to scan != samples with t_srate
t_srate = 1 # sampling rate of Gt files if needed. Should be <= ntime
l_phi_avg = False # phi avg or phi sampling of the fields
phi_sample = 3 # longitude to extract for time sampling
l_minus_mean = False # remove mean value or phi avg from fields

rank = 0 # default proc number when l_mpi == False
l_mpi = True # use MPI to speed up horizontal gradients for the curl computation or not

l_spectral = True # using spectral derivatives in theta and phi for curl? (advised:  = True)
l_trunc = False # truncation for SH?

l_resym = True # re-symmetrizing the data if minc !=1
l_back_b0 = False # add back the background magnetic field b0

#-- I/O; NOTE: does not seem to replace files if they exist even if l_save*!
l_save_bal = 1 # save computed balances?
l_zavg = 1 # using python method to compute z-avg of the forces
l_zavg_ja = 1 # using JA method to compute z-avg and save the related quantities in an hdf5 file
l_save_balzavg = 1 # save zAvg-balances?
l_binary = False # in a binary or in a hd5f format?
saveDir = '/Users/obarrois/Desktop/Parodys/Outputs/Waves'+run_Ek+'/Balances/Data/' # path to save colmunar balance files
#saveDir = '/Users/obarrois/Desktop/stella/Parodys/Outputs/Data/' # if you use python from perso to remote data directly from cluster

#------------------------------------------------------------------------------
#%% Initialisation

if ( l_save_bal or l_save_balzavg ):
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

Vp = np.zeros((n_samples, ntheta, nr),)
ups = np.zeros_like(Vp)
omzs = np.zeros_like(ups)
Lorzs = np.zeros_like(omzs)
Corzs = np.zeros_like(omzs)
timet = np.zeros((n_samples),)

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
    if ( False ): #-- testing
        #-- Broadcast data to avoid reading X times data files
        if ( l_mpi and size>1 ): comm.bcast(n_samples, root=0); comm.bcast(nphi, root=0)
        if ( l_mpi and size>1 and rank!=0 ):
            timet = np.empty((n_samples),)
            Br = np.empty((n_samples, nphi, ntheta, nr),)
            Bt = np.empty_like(Br); Bp = np.empty_like(Br)
            ur = np.empty_like(Br); ut = np.empty_like(ur); up = np.empty_like(ur)
        if ( l_mpi and size>1 ):
            comm.Bcast(timet, root=0)
            comm.Bcast(Br, root=0); comm.Bcast(Bt, root=0); comm.Bcast(Bp, root=0)
            comm.Bcast(ur, root=0); comm.Bcast(ut, root=0); comm.Bcast(up, root=0)
    #
    #------------------------------------------------------------------------------
    #-- Compute curls
    #
    #-- Computation of point time Forces
    print('Computing Forces for file n°{}/{}, t={}, rank={}'.format(n_s+1, n_samples, time, rank))
    #
    #-- Vorticity: Vxu
    omr, omt, _ = curl_sph(grid_help, ur, ut, up, l_mpi=l_mpi, l_spectral=l_spectral, sh_struct=sh)
    if ( np.sum(np.isnan(omr)) + np.sum(np.isnan(omt)) != 0 ): print('Warning!:: nan in curl u computations!') # to check if there are any problems #+ np.sum(np.isnan(omp)) 
    omz = cost3D*omr - sint3D*omt
    del(omr,omt)#,omp)
    #
    #-- Coriolis: -2*u x Omega
    corr = 2.*-sint3D*up#up #corr=2*-mmsint.*Vp;
    cort = 2.*-cost3D*up#up #cort=2*-mmcost.*Vp;
    corp = 2.*(sint3D*ur + cost3D*ut) #corp=2*(mmcost.*Vt+mmsint.*Vr); #(sint3D*ur + cost3D*ut)
    ccorr, ccort, _ = curl_sph(grid_help, corr, cort, corp, l_mpi=l_mpi, l_spectral=l_spectral, sh_struct=sh)
    #pccorr, pccort, _ = curl_sph(grid_help, corr, cort, corp, l_mpi=False, l_spectral=False, sh_struct=None) # double check if needed
    if ( np.sum(np.isnan(ccorr)) + np.sum(np.isnan(ccort)) != 0 ): print('Warning!:: nan in curl Cor computations!') # to check if there are any problems #+ np.sum(np.isnan(ccorp))
    Corz = cost3D*ccorr - sint3D*ccort #ccorz=mmcost.*ccorr-mmsint.*ccort;
    del(ccorr,ccort)#,ccorp)
    del(corr,cort,corp)
    #
    #-- Curl B: j = VxB
    jr, jt, jp = curl_sph(grid_help, Br, Bt, Bp, l_mpi=l_mpi, l_spectral=l_spectral, sh_struct=sh)
    if ( np.sum(np.isnan(jr)) + np.sum(np.isnan(jt)) + np.sum(np.isnan(jp)) != 0 ): print('Warning!:: nan in curl B computations!') # to check if there are any problems
    #-- Curl B cross B (magnetic stress?): jxB
    jxBr = jt*B0p - jp*B0t + j0t*Bp - j0p*Bt # jt*Bp - jp*Bt + jt*Bp - jp*Bt #
    jxBt = jp*B0r - jr*B0p + j0p*Br - j0r*Bp # jp*Br - jr*Bp + jp*Br - jr*Bp #
    jxBp = jr*B0t - jt*B0r + j0r*Bt - j0t*Br # jr*Bt - jt*Br + jr*Bt - jt*Br #
    del(jr,jt,jp)
    #-- Lorentz force: VxjxB
    VxjxBr, VxjxBt, _ = curl_sph(grid_help, jxBr, jxBt, jxBp, l_mpi=l_mpi, l_spectral=l_spectral, sh_struct=sh)
    if ( np.sum(np.isnan(VxjxBr)) + np.sum(np.isnan(VxjxBt)) != 0 ): print('Warning!:: nan in curl jxB computations!') # to check if there are any problems #+ np.sum(np.isnan(VxjxBp)) 
    Lorz = cost3D*VxjxBr - sint3D*VxjxBt
    del(VxjxBr,VxjxBt)#,VxjxBp)
    del(jxBr,jxBt,jxBp)
    #
    #-- remove phi average or mean value from fields
    Vp[n_s] = up.mean(axis=0) #-- Zonal velocity
    if ( l_minus_mean ):
        upmm = np.mean(up)#omz.mean(axis=0)
        omzmm = np.mean(omz)#omz.mean(axis=0)
        Lorzmm = np.mean(Lorz)#Lorz.mean(axis=0)
        Corzmm = np.mean(Corz)#Corz.mean(axis=0)
    else:
        upmm = Vp[n_s].copy()#up.mean(axis=0)#
        omzmm = omz.mean(axis=0)
        Lorzmm = Lorz.mean(axis=0)
        Corzmm = Corz.mean(axis=0)
    #--   Corzs(incr,:,:)=squeeze(ccorz(4,:,:))-squeeze(mean(ccorz)); # NOTE: mean() in matlab only average over one dimension (here the first)
    #-- remove mean value from fields and then phi averaged or phi-sampling if needed
    if ( l_phi_avg ):
        uprm = np.zeros((nphi,ntheta,nr),)
        omzrm = np.zeros((nphi,ntheta,nr),)
        Lorzrm = np.zeros((nphi,ntheta,nr),)
        Corzrm = np.zeros((nphi,ntheta,nr),)
        for n_p in range(nphi):
            print('Removing mean field for phi {}/{}'.format(n_p+1, nphi))
            uprm[n_p]  = up[n_p] - upmm#Vp
            omzrm[n_p]  = omz[n_p] - omzmm
            Lorzrm[n_p] = Lorz[n_p] - Lorzmm
            Corzrm[n_p] = Corz[n_p] - Corzmm
        ups[n_s] = uprm.mean(axis=0)
        omzs[n_s] = omzrm.mean(axis=0)
        Lorzs[n_s] = Lorzrm.mean(axis=0)
        Corzs[n_s] = Corzrm.mean(axis=0)
        titleTimeAvg = 'phi-time-Avg'
        del(uprm,omzrm,Lorzrm,Corzrm)
    else: #-- phi sampling: take only one longitude to save space and computation time
        ups[n_s]  = up[phi_sample] - upmm#Vp
        omzs[n_s]  = omz[phi_sample] - omzmm #omzs(incr,:,:)=squeeze(omz(4,:,:))-squeeze(mean(omz));
        Lorzs[n_s] = Lorz[phi_sample] - Lorzmm #Lorzs(incr,:,:)=squeeze(ccbz(4,:,:))-squeeze(mean(ccbz));
        Corzs[n_s] = Corz[phi_sample] - Corzmm #Corzs(incr,:,:)=squeeze(ccorz(4,:,:))-squeeze(mean(ccorz));
        titleTimeAvg = 'slice-time-Avg'
    #-- cleaning extra fields because it can be RAM extensive
    del(upmm,omzmm,Lorzmm,Corzmm)
    #print('end loop n°={}'.format(n_s))
    #del(up, omz, Lorz, Corz)
#-- end loop reading and computing forces

#-- Time Redimensionalisation
#-- Better to do all the redimensionalisation in the plot script: new standard from March 10 2023
tdim = timet.copy()#/(Ek/Lehnert)

#-- save forces in a data file?
if( l_save_bal and rank==0 ):
    if ( l_binary ): #-- save a binary file using native python open and tofile routines
        f = open('{}/balances.bi'.format(saveDir), 'wb')
        timet.tofile(f)
        radius.tofile(f)
        theta.tofile(f)
        phir.tofile(f)
        Vp.tofile(f)
        ups.tofile(f)
        omzs.tofile(f)
        Lorzs.tofile(f)
        Corzs.tofile(f)
        f.close()
    else: #-- save balances using h5py
        with h5py.File('{}/balances.hdf5'.format(saveDir), 'a') as f:
            f.create_dataset('time', data=timet)
            f.create_dataset('radius', data=radius)
            f.create_dataset('theta', data=theta)
            f.create_dataset('phi', data=phir)
            f.create_dataset('VpZonal', data=Vp)
            f.create_dataset('upSlice', data=ups)
            f.create_dataset('omzSlice', data=omzs)
            f.create_dataset('LorzSlice', data=Lorzs)
            f.create_dataset('CorzSlice', data=Corzs)

#------------------------------------------------------------------------------
#%% Compute z-Avg force balances

#-- Compute z-avg
if ( l_zavg ):
    rrad = radius[::-1] # zavg() expect reversed radius
    #nsmax = 309 #--> to get nsmax ~ nsmaxja
    nsmax = int(nr*1.5 + 17) # to maintain nsmax ~ nrmax outside of TC (but will give nsmax > nrmax)
    # NOTE: WARNING!:: zavg() or zavgpy() expect reversed radius as in pizza/MagIC (r_cmb = r[0]) SO the arrays MUST BE REVSERED as well
    print('Computing Fortran z-avg of the Forces for all {} files'.format(n_samples))
    hs, rs, Vpcol = zavg(Vp[..., ::-1], rrad, nsmax, colat=theta, minc=1, normed=True, save=False) #zavgpy(Vp[..., ::-1], rrad, nsmax, minc=1, normed=True)
    hs, rs, upcol = zavg(ups[..., ::-1], rrad, nsmax, colat=theta, minc=1, normed=True, save=False)
    hs, rs, omzcol = zavg(omzs[..., ::-1], rrad, nsmax, colat=theta, minc=1, normed=True, save=False)
    hs, rs, Lorcol = zavg(Lorzs[..., ::-1], rrad, nsmax, colat=theta, minc=1, normed=True, save=False)
    hs, rs, Corcol = zavg(Corzs[..., ::-1], rrad, nsmax, colat=theta, minc=1, normed=True, save=False)
    #-- exluding IC
    Vpcol = Vpcol[:,rs>=radius[0]]; upcol = upcol[:,rs>=radius[0]]
    omzcol = omzcol[:,rs>=radius[0]]; Lorcol = Lorcol[:,rs>=radius[0]]; Corcol = Corcol[:,rs>=radius[0]]
    sr = rs[rs>=radius[0]]
    #-- Would probably be better to re-compute these d . /dt fields in the plot script (where the re-dimensionalisation is done properly)
    #-- Vp and domzdt balances
    updiff = upcol#Vpcol
    dVpcol = np.zeros_like(omzcol)
    domzcol = np.zeros_like(omzcol)
    dVpcol[0] = (updiff[1] - updiff[0])/(tdim[1] - tdim[0])#(upcol[1] - upcol[0])/(tdim[1] - tdim[0])
    domzcol[0] = (omzcol[1] - omzcol[0])/(tdim[1] - tdim[0])
    o2dt = 0.5/np.diff(tdim.astype(np.float32)).mean() #dt=mean(diff(tt));
    for n_s in range(1,n_samples-1):
        #o2dt = 0.5/(tdim[n_s+1] - tdim[n_s-1])
        dVpcol[n_s,:] = (updiff[n_s+1,:] - updiff[n_s-1,:])*o2dt#(upcol[n_s+1,:] - upcol[n_s-1,:])*o2dt
        domzcol[n_s,:] = (omzcol[n_s+1,:] - omzcol[n_s-1,:])*o2dt
    dVpcol[-1] = (updiff[-1] - updiff[-2])/(tdim[-1] - tdim[-2])#(upcol[-1] - upcol[-2])/(tdim[-1] - tdim[-2])
    domzcol[-1] = (omzcol[-1] - omzcol[-2])/(tdim[-1] - tdim[-2])
    del(updiff)
    #
    #-- Remaining of the balance: should be equivalent to the Ageostrophic Forces!
    #AgeosCorcol=Lorcol-dVpcol
    #Residuals = Lorcol-Corcol-domzcol
    #
    #-- save forces in a data file?
    if( l_save_balzavg and rank==0 ):
        if ( l_binary ): #-- save a binary file using native python open and tofile routines
            f = open('{}/Zavg_bal.bi'.format(saveDir), 'wb')
            timet.tofile(f)
            sr.tofile(f)
            Vpcol.tofile(f)
            upcol.tofile(f)
            omzcol.tofile(f)
            Lorcol.tofile(f)
            Corcol.tofile(f)
            dVpcol.tofile(f)
            domzcol.tofile(f)
            #AgeosCorcol.tofile(f) # only a by-product of Lorcol and dVpcol
            f.close()
        else: #-- save balances using h5py
            with h5py.File('{}/Zavg_bal.hdf5'.format(saveDir), 'a') as f:
                f.create_dataset('time', data=timet)
                f.create_dataset('srad', data=sr)
                f.create_dataset('Vpcol', data=Vpcol)
                f.create_dataset('upcol', data=upcol)
                f.create_dataset('omzcol', data=omzcol)
                f.create_dataset('Lorcol', data=Lorcol)
                f.create_dataset('Corcol', data=Corcol)
                f.create_dataset('dVpcol', data=dVpcol)
                f.create_dataset('domzcol', data=domzcol)
                #f.create_dataset('AgeosCorcol', data=AgeosCorcol) # only a by-product of Lorcol and dVpcol


#-- Testing filter z-avg as Julien Aubert
if ( l_zavg_ja ):
    #-- equally spaced radius
    #ss = np.linspace(radius[0],radius[-1],200)#s=[r(1):1/200:r(nr)];
    #from parobpy.parob_lib import rderavg, thetaderavg
    #sc2a = abs(r3D[0]*cost3D[0]) #--> for benchmark purposes
    srja=np.linspace(radius[0],radius[-1],len(sr))#201) #--> rebuild radius like in curlbal_QGA_4_JA.m
    nsmaxja = len(srja)
    #
    #-- building filter function
    # NOTE: numerical gradient! = spacing between points --> dx != physical gradient --> d./dx
    gradr = np.zeros_like(r3D[0]) #mdr=gradient(r)*ones(1,nt); 
    gradt = np.zeros_like(r3D[0]) #mdt=ones(nr,1)*gradient(theta');
    for j in range(ntheta):
        gradr[j,:] = np.gradient(radius)
    for i in range(nr):
        gradt[:,i] = np.gradient(theta)
    dS = gradr*r3D[0]*gradt #mdr.*mr.*mdt;
    Ls = np.zeros((nsmaxja),)
    zslice = np.zeros((ntheta,nr,nsmaxja),) #slice=zeros(nr,nt,sr);
    for i in range(nsmaxja): #slice(:,:,ir)=exp(-(r*sint'-s(ir)).^2/5e-4);
        zslice[:,:,i] = np.exp(-(r3D[0]*sint3D[0]-srja[i])**2/5.e-4) # NOTE: WARNING!:: radius should Increase here!
        Ls[i] = 1./np.sum(zslice[:,:,i]*dS[:,:]) #L(ir)=1./sum(sum(slice(:,:,ir).*dS));
        zslice[:,:,i] = zslice[:,:,i]*Ls[i]*dS[:,:] #slice(:,:,ir)=slice(:,:,ir)*L(ir).*dS;
    #-- actual z-avg
    #sc2=squeeze(omzs(incr,:,:))';
    #for ir=1:sr
    #   omzcol(incr,ir)=sum(sum(sc2.*squeeze(slice(:,:,ir))));
    omzcolja = np.zeros((n_samples,nsmaxja),)
    Lorcolja = np.zeros_like(omzcolja)
    Corcolja = np.zeros_like(omzcolja)
    for n_s in range(n_samples):
        print('Computing z-avg of the Forces (with a filter) for file n°{}/{}, t={}'.format(n_s+1, n_samples, timet[n_s]))
        for i in range(nsmaxja):
            #testja[i] = np.sum(sc2a*zslice[:,:,i])
            omzcolja[n_s,i] = np.sum(omzs[n_s]*zslice[:,:,i]) #sum(sum(sc2.*squeeze(slice(:,:,ir))));
            Lorcolja[n_s,i] = np.sum(Lorzs[n_s]*zslice[:,:,i]) # NOTE: np.sum() sums all components so no need to repeat it!
            Corcolja[n_s,i] = np.sum(Corzs[n_s]*zslice[:,:,i])
    #
    #-- Would probably be better to re-compute these d . /dt fields in the plot script (where the re-dimensionalisation is done properly)
    #-- Vp and domzdt balances
    domzcolja = np.zeros_like(omzcolja)
    o2dt = 0.5/np.diff(tdim.astype(np.float32)).mean() #dt=mean(diff(tt));
    for n_s in range(1,n_samples-1):
        domzcolja[n_s,:] = (omzcolja[n_s+1,:] - omzcolja[n_s-1,:])*o2dt #domzcol(incr,:)=(omzcol(incr+1,:)-omzcol(incr-1,:))/(2*dt);
    #
    if( l_save_balzavg and rank==0 ):
        with h5py.File('{}/Zavg_bal_ja.hdf5'.format(saveDir), 'a') as f:
            f.create_dataset('radiusja', data=srja)
            f.create_dataset('omzcolja', data=omzcolja)
            f.create_dataset('Lorcolja', data=Lorcolja)
            f.create_dataset('Corcolja', data=Corcolja)
            f.create_dataset('domzcolja', data=domzcolja)

#-- End Script
