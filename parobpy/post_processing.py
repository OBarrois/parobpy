#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 16:23:22 2021
@author: wongj

modified by: obarrois 01 Feb 2023
"""
import numpy as np
try:
    import chaosmagpy as cp
    l_chaos_magp = True
except:# ModuleNotFoundError:
    print("chaosmagpy: Module Not Found")
    l_chaos_magp = False
try:
    import shtns
    l_shtns = True
except:# ModuleNotFoundError:
    print("shtns: Module Not Found")
    l_shtns = False
try:
    import h5py
    l_h5py = True
except:# ModuleNotFoundError:
    print("h5py: Module Not Found")
    l_h5py = False

from scipy.integrate import trapz
from parobpy.core_properties import icb_radius, cmb_radius, earth_radius
from parobpy.load_parody import parodyload, list_Gt_files, list_St_files, surfaceload

shell_gap = cmb_radius - icb_radius
ro = cmb_radius/shell_gap

def sim_time(data):
    '''Total simulation time (viscous) from diagnostic data'''
    time = data.time.iloc[-1]-data.time.iloc[0]
    
    return time

def grav_torque(mantle_data):
    '''Mean and maximum gravitational torque'''
    gamma=np.mean(mantle_data.gravitational_torque_on_mantle)
    gamma_max = max(np.abs(mantle_data.gravitational_torque_on_mantle))
    
    return gamma,gamma_max

def meridional_timeavg(run_ID, directory):
    '''Time average of azimuthally averaged Gt files'''
    Gt_file = list_Gt_files(run_ID,directory) # Find all Gt_no in folder
    n = len(Gt_file)

    # Loop over time
    Vt_m, Vp_m, Vr_m, Br_m, Bt_m, Bp_m, T_m = [
        [] for _ in range(7)]
    i=0
    for file in Gt_file:
        print('Loading {} ({}/{})'.format(file, i+1, n))
        filename = '{}/{}'.format(directory,file)
        (_, _, _, _, _, _, _,
            _, _, _, _, _, _, _, _,
            _, _, _, _, radius, theta, phi, Vr, Vt, Vp,
            Br, Bt, Bp, T) = parodyload(filename)
        # Zonal averages
        Vr_m.append(np.mean(Vr,axis=0))
        Vt_m.append(np.mean(Vt, axis=0))
        Vp_m.append(np.mean(Vp,axis=0))
        Br_m.append(np.mean(Br,axis=0))
        Bt_m.append(np.mean(Bt, axis=0))
        Bp_m.append(np.mean(Bp,axis=0))
        T_m.append(np.mean(T,axis=0))
        i += 1
    # Time average (should really divide by dt but n is good enough according to JA)
    Vr_out = sum(Vr_m)/n
    Vt_out = sum(Vt_m)/n
    Vp_out = sum(Vp_m)/n
    Br_out = sum(Br_m)/n
    Bt_out = sum(Bt_m)/n
    Bp_out = sum(Bp_m)/n
    T_out = sum(T_m)/n
    
    # Save
    if ( l_h5py ):
        with h5py.File('{}/meridional_timeavg'.format(directory), 'a') as f:
            f.create_dataset('radius', data=radius)
            f.create_dataset('theta', data=theta)
            f.create_dataset('phi', data=phi)
            f.create_dataset('Vr', data=Vr_out)
            f.create_dataset('Vt', data=Vt_out)
            f.create_dataset('Vp', data=Vp_out)
            f.create_dataset('Br', data=Br_out)
            f.create_dataset('Bt', data=Bt_out)
            f.create_dataset('Bp', data=Bp_out)
            f.create_dataset('T', data=T_out)

    return (radius, theta, phi, Vr_out, Vt_out, Vp_out, Br_out, Bt_out, Bp_out, T_out)

def surface_timeavg(run_ID, directory):
    '''
    Time average surface fields
    '''
    St_file = list_St_files(run_ID,directory) # Find all Gt_no in folder
    n = len(St_file)

    # Loop over time
    Vt_s, Vp_s, Br_s, dtBr_s = [[] for _ in range(4)]
    i=0
    for file in St_file:
        print('Loading {} ({}/{})'.format(file, i+1, n))
        filename = '{}/{}'.format(directory,file)

        (_, _, _, _, _, _, _,
          _, _, _, _, _, _, _, _,
            _, _, _, _, _, theta, phi, Vt, Vp, Br,
            dtBr) = surfaceload(filename)
        # Append
        Vt_s.append(Vt)
        Vp_s.append(Vp)
        Br_s.append(Br)
        dtBr_s.append(dtBr)
        i += 1
    # Time average (should really divide by dt but n is good enough)
    Vt_out = sum(Vt_s)/n
    Vp_out = sum(Vp_s)/n
    Br_out = sum(Br_s)/n
    dtBr_out = sum(dtBr_s)/n

    # Save 
    if ( l_h5py ):
        with h5py.File('{}/surface_timeavg'.format(directory), 'a') as f:
            f.create_dataset('theta', data=theta)
            f.create_dataset('phi', data=phi)
            f.create_dataset('Vt', data=Vt_out)
            f.create_dataset('Vp', data=Vp_out)
            f.create_dataset('Br', data=Br_out)
            f.create_dataset('dtBr', data=dtBr_out)
    print('{}/surface_timeavg saved'.format(directory))

    return (theta, phi, Vt_out, Vp_out, Br_out, dtBr_out)

def surface_phiavg_timeavg(run_ID, directory):
    '''
    Time average phi averaged surface fields
    '''
    St_file = list_St_files(run_ID, directory)  # Find all St_no in folder
    n = len(St_file)

    # Loop over time
    Vt_s, Vp_s, Br_s, dtBr_s = [[] for _ in range(4)]
    i = 0
    for file in St_file:
        print('Loading {} ({}/{})'.format(file, i+1, n))
        filename = '{}/{}'.format(directory, file)

        (_, _, _, _, _, _, _,
         _, _, _, _, _, _, _, _,
            _, _, _, _, _, theta, phi, Vt, Vp, Br,
            dtBr) = surfaceload(filename)
        # Append
        Vt_s.append(np.mean(Vt,axis=0))
        Vp_s.append(np.mean(Vp,axis=0))
        Br_s.append(np.mean(Br,axis=0))
        dtBr_s.append(np.mean(dtBr,axis=0))
        i += 1
    # Time average (should really divide by dt but n is good enough)
    Vt_out = sum(Vt_s)/n
    Vp_out = sum(Vp_s)/n
    Br_out = sum(Br_s)/n
    dtBr_out = sum(dtBr_s)/n

    # Save
    if ( l_h5py ):
        with h5py.File('{}/surface_phiavg_timeavg'.format(directory), 'w') as f:
            f.create_dataset('theta', data=theta)
            f.create_dataset('phi', data=phi)
            f.create_dataset('Vt', data=Vt_out)
            f.create_dataset('Vp', data=Vp_out)
            f.create_dataset('Br', data=Br_out)
            f.create_dataset('dtBr', data=dtBr_out)
    print('{}/surface_phiavg_timeavg saved'.format(directory))

    return (theta, phi, Vt_out, Vp_out, Br_out, dtBr_out)

if ( l_chaos_magp ):
    def CHAOS_phiavg_timeavg(nphi, ntheta, lmax, yearStart, yearEnd, n, radius=3485):
        model = cp.load_CHAOS_matfile('../data/CHAOS-7.7.mat')
        time = np.linspace(yearStart, yearEnd, n)
        time_chaos = np.array([cp.data_utils.dyear_to_mjd(x) for x in time]) # modified Julian date
        # create grid
        latC = np.linspace(0.01, 179.99, num=ntheta)  # colatitude in degrees
        lonC = np.linspace(-179.99, 179.99, num=nphi)  # longitude in degrees
        Br_avg = []
        for t in time_chaos:
            print('Loading CHAOS time {:.2f}'.format(t))
            # ylm = model.synth_coeffs_tdep(t)
            # dtglm = model.synth_coeffs_tdep(t, deriv=1)  # Secular Variation
            Br, _, _ = model.synth_values_tdep(
                t, radius, latC, lonC, nmax=lmax, deriv=0, grid=True)  # (nT)
            BrC = trapz(Br, lonC, axis=1) # phi average
            Br_avg.append(BrC)
        Br_out = sum(Br_avg)/n
        latC = 90-latC # NOTE: 0<theta<180 to read data but -90<theta<90 to plot

        return (latC, lonC, Br_out)

def convective_power_timeavg(run_ID, directory):
    Gt_file = list_Gt_files(run_ID, directory)  # Find all Gt_no in folder
    n = len(Gt_file)

    # Loop over time
    I_avg = []
    i = 0
    for file in Gt_file:
        print('Loading {} ({}/{})'.format(file, i+1, n))
        filename = '{}/{}'.format(directory, file)
        (_, _, _, _, _, _, _,
            _, _, _, _, _, _, _, _,
            nr, ntheta, nphi, _, radius, theta, phi, Vr, _, _,
            _, _, _, T) = parodyload(filename)
        # Integrate over phi and theta
        mr, mdr, msint, mcost, mdt, mdp = initialise_dV(
            nphi, ntheta, nr, phi, theta, radius)
        dV = mr**2*mdr*mdt*msint*mdp
        A = mr*Vr*T
        I = np.sum(np.sum(A*dV, axis=1), axis=0) / \
            np.sum(np.sum(dV, axis=0), axis=1)
        I_avg.append(I)
        i += 1
    # Time average (should really divide by dt but n is good enough)
    I_out = sum(I_avg)/n
    # Save
    if ( l_h5py ):
        with h5py.File('{}/convective_power'.format(directory), 'a') as f:
            f.create_dataset('radius', data=radius)
            f.create_dataset('I', data=I_out)

    return radius, I_out

def gauss_ratios_timeavg(run_ID, directory, l_trunc=14):
    St_file = list_St_files(run_ID, directory)  # Find all St_no in folder
    n = len(St_file)

    # Loop over time
    g10_avg, g20_avg, g30_avg, G2_avg, G3_avg = [[] for _ in range(5)]
    i = 0
    for file in St_file:
        print('Loading {} ({}/{})'.format(file, i+1, n))
        filename = '{}/{}'.format(directory, file)
        (_, _, _, _, _, _, _,
            _, _, _, _, _, _, _, _,
            _, ntheta, nphi, _, _, theta, phi, _, _, Br,
            _) = surfaceload(filename)
        # Gauss coefficients
        glm, _ = gauss_coeffs(Br, nphi, ntheta, l_trunc)
        G2 = glm[2]/glm[1]
        G3 = glm[3]/glm[1]
        g10_avg.append(glm[1])
        g20_avg.append(glm[2])
        g30_avg.append(glm[3])
        G2_avg.append(G2)
        G3_avg.append(G3)
        i += 1
    # Time average (should really divide by dt but n is good enough)
    g10_out = sum(g10_avg)/n
    g20_out = sum(g20_avg)/n
    g30_out = sum(g30_avg)/n
    G2_out = sum(G2_avg)/n
    G3_out = sum(G3_avg)/n
    # Save
    if ( l_h5py ):
        with h5py.File('{}/gauss_ratios_timeavg'.format(directory), 'a') as f:
            f.create_dataset('g10', data=g10_out)
            f.create_dataset('g20', data=g20_out)
            f.create_dataset('g30', data=g30_out)
            f.create_dataset('G2', data=G2_out)
            f.create_dataset('G3', data=G3_out)
    
    return g10_out, g20_out, g30_avg, G2_out, G3_out

def ur_avg(run_ID, directory, rf):
    '''
    Time series of theta and phi averaged radial velocity in the F-layer
    '''
    Gt_file = list_Gt_files(run_ID, directory)  # Find all Gt_no in folder
    n = len(Gt_file)

    # Loop over time
    ur_avg = []
    i = 0
    for file in Gt_file:
        print('Loading {} ({}/{})'.format(file, i+1, n))
        filename = '{}/{}'.format(directory, file)
        (_, _, _, _, _, _, _,
            _, _, _, _, _, _, _, _,
            nr, ntheta, nphi, _, radius, theta, phi, Vr, _, _,
            _, _, _, T) = parodyload(filename)
        # Phi and theta average
        ur = Vr.mean(axis=0).mean(axis=0)
        # Truncate to F-layer
        if i == 0:
            idx = np.argwhere(radius < rf)[-1][0]
        ur = ur[:idx]
        # Average ur in F-layer
        ur_avg.append(ur.mean())
        i += 1
    # Save
    if ( l_h5py ):
        with h5py.File('{}/flayer_ur'.format(directory), 'a') as f:
            f.create_dataset('ur', data=ur_avg)

    return ur_avg

def initialise_dV(nphi,ntheta,nr,phi,theta,r):
    '''
    Prepare grid for integration in spherical coordinates
    '''
    mr = np.ones([nphi, ntheta, nr])
    mdr = np.ones([nphi, ntheta, nr])
    gr = np.gradient(r)
    for i in range(nr):
        mr[: , : , i] = r[i]
        mdr[: , : , i] = gr[i]

    gt = np.gradient(theta)
    msint = np.ones([nphi, ntheta, nr])
    mcost = np.ones([nphi, ntheta, nr])
    mdt = np.ones([nphi, ntheta, nr])
    sint = np.array([np.sin(x) for x in theta])
    cost = np.array([np.cos(x) for x in theta])
    for i in range(ntheta):
        msint[: , i, : ] = sint[i]
        mcost[: , i, : ] = cost[i]
        mdt[: , i, : ] = gt[i]

    mdp = np.ones([nphi, ntheta, nr])
    gp = np.gradient(phi)
    for i in range(nphi):
        mdp[i, : , : ] = gp[i]

    return mr, mdr, msint, mcost, mdt, mdp

def ref_codensity(r,rf,fi):
    '''
    Compute reference codensity
    '''
    nr = r.size
    shell_gap = cmb_radius-icb_radius
    ro = cmb_radius/shell_gap
    ri = icb_radius/shell_gap
    idx_f = np.where(r<rf)
    idx_o = np.where(r>=rf)

    if fi<0:
        So = -3/(ro**3-rf**3)
        Sf = 3*(1-fi)/(rf**3-ri**3)
        dCodr = So*(ro**3 - r[idx_o]**3)/(3*r[idx_o]**2)
        dCfdr = -Sf*(r[idx_f]**3-ri**3)/(3*r[idx_f]**2) - fi/r[idx_f]**2

        alpha = -(ro**3/rf+rf**2/2)/(ro**3 - rf**3) - \
            (1-fi)*(rf**2/2 + ri**3/rf)/(rf**3-ri**3) + fi/rf
        Co = (ro**3/r[idx_o] + r[idx_o]**2/2)/(ro**3 - rf**3) + alpha
        Cf = - (1-fi)*(r[idx_f]**2/2 + ri**3/r[idx_f])/(rf**3 - ri**3) + fi/r[idx_f]
        C = np.concatenate([Cf, Co])

    return C

if ( l_shtns ):
    def filter_field(Br,nphi,ntheta,l_trunc,m_max=None):
        '''Use SHTns to truncate field to degree l_trunc'''
        if ( m_max == None ): m_max = l_trunc
        sh = shtns.sht(l_trunc, m_max) # by default 'flag = sht_quick_init' uses gaussian grid
        nlat, nlon = sh.set_grid(nphi=nphi, nlat=ntheta) # NOTE: array has to be dtype='float64' and not 'float32'
        vr = Br.T.astype('float64')
        ylm = sh.analys(vr)  # spatial to spectral
        Br_f = sh.synth(ylm)  # spectral to spatial
        return Br_f

    def power_spectrum(Br, nphi, ntheta, l_trunc, r=ro):
        '''
        Obtain power spectrum of core surface field to degree l_trunc (default at the CMB)
        '''
        # SH transform from spatial to spectral space
        m_max = l_trunc 
        # by default 'flag = sht_quick_init' uses gaussian grid
        sh = shtns.sht(l_trunc, m_max)
        nlat, nlon = sh.set_grid(nphi=nphi, nlat=ntheta)
        vr = Br.T.astype('float64') # NOTE: array has to be dtype='float64' and not 'float32'
        clm = sh.analys(vr)  # spatial to spectral
        # Construct Gauss coefficients
        glm = np.zeros(sh.nlm)
        hlm = np.zeros(sh.nlm)
        for l in range(l_trunc+1):
            fac = 1/(l+1)  # cmb_radius/(l+1) # from Br potential to full potential
            for m in range(l+1):
                glm[sh.idx(l, m)] = fac*np.real(clm[sh.idx(l, m)])
                hlm[sh.idx(l, m)] = fac*np.imag(clm[sh.idx(l, m)])
        # Power spectrum
        degrees = np.arange(1, l_trunc+1, 1)
        spectrum = np.zeros(l_trunc+1)
        for l in range(l_trunc+1):
            fac = (l+1)*(earth_radius/cmb_radius)**(2*l+4) # TODO: check factors using time average
            sum_m = 0
            for m in range(l+1):
                sum_m += glm[sh.idx(l,m)]**2+hlm[sh.idx(l, m)]**2
            spectrum[l] = fac*sum_m
        spectrum = spectrum[1:]  # ignore g00

        return degrees, spectrum, clm, glm, hlm

    def gauss_coeffs(Br, nphi, ntheta, l_trunc):
        '''
        Obtain Gauss coefficients from core surface field
        '''
        # SH transform from spatial to spectral space
        m_max = l_trunc 
        # by default 'flag = sht_quick_init' uses gaussian grid
        sh = shtns.sht(l_trunc, m_max)
        nlat, nlon = sh.set_grid(nphi=nphi, nlat=ntheta)
        vr = Br.T.astype('float64') # NOTE: array has to be dtype='float64' and not 'float32'
        clm = sh.analys(vr)  # spatial to spectral
        # Construct Gauss coefficients
        glm = np.zeros(sh.nlm)
        hlm = np.zeros(sh.nlm)
        for l in range(l_trunc+1):
            fac = 1/(l+1)  # cmb_radius/(l+1) # from Br potential to full potential
            for m in range(l+1):
                glm[sh.idx(l, m)] = fac*np.real(clm[sh.idx(l, m)])
                hlm[sh.idx(l, m)] = fac*np.imag(clm[sh.idx(l, m)])

        return glm, hlm

    def fcf(run_ID, directory, l_trunc = 8):
        '''
        Flux concentration factor used in measure of semblance (Christensen et al. 2010). Note that we truncate at degree and order 8
        '''
        St_file = list_St_files(run_ID, directory)  # Find all St_no in folder
        n = len(St_file)

        # Loop over time
        FCF_s = []
        i = 0
        for file in St_file:
            print('Loading {} ({}/{})'.format(file, i+1, n))
            filename = '{}/{}'.format(directory, file)

            (_, _, _, _, _, _, _,
            _, _, _, _, _, _, _, _,
                _, ntheta, nphi, _, _, theta, phi, _, _, Br,
                _) = surfaceload(filename)

            # Truncate degree 8
            m_max = l_trunc
            #sh = shtns.sht(l_trunc,m_max)
            sh = shtns.sht(l_trunc, m_max)
            nlat, nlon = sh.set_grid(nphi=nphi, nlat=ntheta)
            # NOTE: array has to be dtype='float64' and not 'float32'
            vr = Br.T.astype('float64') # NOTE: need to give theta dimension first
            clm = sh.analys(vr)  # spatial to spectral
            Br_f = sh.synth(clm) # spectral to spatial
            Br_4 = (Br_f**4).mean()
            Br_2 = (Br_f**2).mean()
            FCF = (Br_4 - Br_2**2)/Br_2**2

            # Append
            FCF_s.append(np.mean(FCF))
            i += 1
        # Time average (should really divide by dt but n is good enough)
        FCF_out = sum(FCF_s)/n

        # Save
        if ( l_h5py ):
            with h5py.File('{}/fcf'.format(directory), 'w') as f:
                f.create_dataset('FCF_out', data=FCF_out)
        print('{}/fcf saved'.format(directory))

        return FCF_out

def gauss_coeffs0(ylm, lmax):
    '''
    Construct Gauss coefficients glm & hlm from complex coefficients ylm
    '''
    k = 0
    j = 0
    m = 0
    glm = np.zeros(int(lmax*(lmax+3)/2))
    hlm = np.zeros(int(lmax*(lmax+3)/2))
    for l in range(1, lmax+1):
        for i in range(2*l+1):
            if i == 0:
                glm[j] = ylm[k]
                hlm[j] = 0.0
                j += 1
                m += 1
            else:
                if np.mod(i, 2) == 1:
                    glm[j] = ylm[k]
                    j += 1
                else:
                    hlm[m] = ylm[k]
                    m += 1
            k += 1

    return glm, hlm
