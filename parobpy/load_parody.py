#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 16:10:19 2021
@author: wongj

modified by: obarrois 01 Feb 2023
"""

try:
    import pandas as pand
    l_panda = True
except:# ModuleNotFoundError:
    print("pandas: Module Not Found")
    l_panda = False
import numpy as np
import os
import re

from parobpy.core_properties import icb_radius, cmb_radius
#------------------------------------------------------------------------------
# Diagnostics
if ( l_panda ):
    def load_kinetic(run_ID,directory):
        '''Load e_kin.run_ID diagnostic data'''
        filename="e_kin." + run_ID.lower()
        data=pand.read_csv('{}/{}'.format(directory,filename),header=None,delim_whitespace=True)
        data=data.replace({'D':'E'},regex=True) # replace Fortran float D to E notation
        data=data.astype(dtype='float') # convert string to float
        data.columns=["time","ke_per_unit_vol","poloidal_ke","toroidal_ke","axisymmetric_poloidal_ke","axisymmetric_toroidal_ke"]
        return (data)

    def load_magnetic(run_ID,directory):
        '''Load e_mag.run_ID diagnostic data'''
        filename="e_mag." + run_ID.lower()
        data=pand.read_csv('{}/{}'.format(directory,filename),header=None,delim_whitespace=True)
        data=data.replace({'D':'E'},regex=True) # replace Fortran float D to E notation
        data=data.astype(dtype='float') # convert string to float
        data.columns=["time","me_per_unit_vol","poloidal_me","toroidal_me","axisymmetric_poloidal_me","axisymmetric_toroidal_me"]
        return (data)

    def load_nusselt(run_ID,directory):
        '''Load Nuss.run_ID diagnostic data'''
        filename="Nuss." + run_ID.lower()
        data=pand.read_csv('{}/{}'.format(directory,filename),header=None,delim_whitespace=True)
        data=data.replace({'D':'E'},regex=True) # replace Fortran float D to E notation
        # data=data.astype(dtype='float') # convert string to float
        data.columns=["time","nu1","nu2"]
        return (data)

    def load_dipole(run_ID,directory):
        '''Load dipole.run_ID diagnostic data'''
        filename="dipole." + run_ID.lower()
        data=pand.read_csv('{}/{}'.format(directory,filename),header=None,delim_whitespace=True)
        data=data.replace({'D':'E'},regex=True) # replace Fortran float D to E notation
        data=data.astype(dtype='float') # convert string to float
        data.columns=["time","dipole_colatitude","g10","g11","surface_rms_B","dipole_rms_B", "surface_rms_B_deg12"]
        return (data)

    def load_power(run_ID,directory):
        '''Load power.run_ID diagnostic data'''
        filename="power." + run_ID.lower()
        data=pand.read_csv('{}/{}'.format(directory,filename),header=None,delim_whitespace=True)
        data=data.replace({'D':'E'},regex=True) # replace Fortran float D to E notation
        data=data.astype(dtype='float') # convert string to float
        data.columns=["time","available_convective_power_per_unit_vol","viscous","magnetic"]
        return (data)

    def load_scales(run_ID,directory):
        '''Load scales.run_ID diagnostic data'''
        filename="scales." + run_ID.lower()
        data=pand.read_csv('{}/{}'.format(directory,filename),header=None,delim_whitespace=True)
        data=data.replace({'D':'E'},regex=True) # replace Fortran float D to E notation
        data=data.astype(dtype='float') # convert string to float
        data.columns=["time","mean_l_velocity","mean_l_field","mean_l_temperature", \
                      "median_l_velocity","median_l_field","median_l_temperature", \
                      "dissipation_scale_velocity","dissipation_scale_field"]
        return (data)

    def load_spec_l(run_ID,directory):
        '''Load spec_l.run_ID diagnostic data'''
        filename="spec_l." + run_ID.lower()
        data=pand.read_csv('{}/{}'.format(directory,filename),header=None,delim_whitespace=True)
        data=data.replace({'D':'E'},regex=True) # replace Fortran float D to E notation
        data=data.astype(dtype='float') # convert string to float
        data.columns=["sh_degree","instant_velocity","timeavg_velocity", \
                      "instant_field", "timeavg_field"]
        return (data)

    def load_spec_m(run_ID,directory):
        '''Load spec_m.run_ID diagnostic data'''
        filename="spec_m." + run_ID.lower()
        data=pand.read_csv('{}/{}'.format(directory,filename),header=None,delim_whitespace=True)
        data=data.replace({'D':'E'},regex=True) # replace Fortran float D to E notation
        data=data.astype(dtype='float') # convert string to float
        data.columns=["sh_order","instant_velocity","timeavg_velocity", \
                      "instant_field", "timeavg_field"]
        return (data)

    def load_mantle(run_ID,directory):
        '''Load mantle.run_ID diagnostic data'''
        filename="mantle." + run_ID.lower()
        data=pand.read_csv('{}/{}'.format(directory,filename),header=None,delim_whitespace=True)
        data=data.replace({'D':'E'},regex=True) # replace Fortran float D to E notation
        data=data.astype(dtype='float') # convert string to float
        data.columns=["time","mantle_rotation_rate","rotation_rate_on_CMB_fluid_side", \
                      "magnetic_torque_on_mantle","gravitational_torque_on_mantle", \
                      "total_angular_momentum_ic+oc+m"]
        return (data)

    def load_innercore(run_ID,directory):
        '''Load innercore.run_ID diagnostic data'''
        filename="innercore." + run_ID.lower()
        data=pand.read_csv('{}/{}'.format(directory,filename),header=None,delim_whitespace=True)
        data=data.replace({'D':'E'},regex=True) # replace Fortran float D to E notation
        data=data.astype(dtype='float') # convert string to float
        data.columns=["time","me_ic","poloidal","toroidal","rotation_sic", \
                      "rotation_fic","viscous_torque_ic", "magnetic_torque_ic", \
                      "gravity_torque_ic","total_angular_momentum_ic+oc+m"]
        return (data)

    def load_compliance(run_ID,directory):
        '''Load compliance.run_ID diagnostic data'''
        filename="compliance." + run_ID.lower()
        data=pand.read_csv('{}/{}'.format(directory,filename),header=None,delim_whitespace=True)
        data=data.replace({'D':'E'},regex=True) # replace Fortran float D to E notation
        data=data.astype(dtype='float') # convert string to float
        data.columns=["time","ADNAD", "OE", "ZNZ"]

        return data

#------------------------------------------------------------------------------
# Graphics and surface data
def fread(fid, nelements, dtype):
     '''Matlab fread equivalent'''
     if dtype is np.str_:
         dt = np.uint8  # WARNING: assuming 8-bit ASCII for np.str
     else:
         dt = dtype
     data_array = np.fromfile(fid, dt, nelements)
     data_array.shape = (nelements, 1)

     return data_array

def parodyload(filename):
    '''PARODY-JA4.3/Matlab/parodyload_v4.m equivalent'''
    fid = open(filename, 'rb')

    version = fread(fid, 1, np.int16)
    version = version[0][0]
    dummy = fread(fid, 1, np.int16)
    phypar = fread(fid, 10, np.float64)
    # Simulation parameters
    time = phypar[0][0]
    DeltaU = phypar[1][0]
    Coriolis = phypar[2][0]
    Lorentz = phypar[3][0]
    Buoyancy = phypar[4][0]
    ForcingU = phypar[5][0]
    DeltaT = phypar[6][0]
    ForcingT = phypar[7][0]
    DeltaB = phypar[8][0]
    ForcingB = phypar[9][0]

    Ek = 1/Coriolis
    Ra = Buoyancy*Ek
    Pm = 1/DeltaB
    Pr = 1/DeltaT

    # Grid parameters
    gridpar = fread(fid, 8, np.int16)
    nr = int(gridpar[0][0])
    ntheta = int(gridpar[2][0])
    nphi = int(gridpar[4][0])
    azsym = int(gridpar[6][0])

    radius = fread(fid, nr, np.float64)
    radius = radius.transpose()[0]
    theta = fread(fid, ntheta, np.float64)
    theta = theta.transpose()[0]
    phi = np.arange(1,nphi+1)*2*np.pi/(nphi*azsym)

    # Output fields
    Vr = np.zeros((nphi,ntheta,nr))
    Vt = np.zeros((nphi,ntheta,nr))
    Vp = np.zeros((nphi,ntheta,nr))
    Br = np.zeros((nphi,ntheta,nr))
    Bt = np.zeros((nphi,ntheta,nr))
    Bp = np.zeros((nphi,ntheta,nr))
    T = np.zeros((nphi,ntheta,nr))

    a = fread(fid, nr*ntheta*nphi, np.float32)
    Vr = np.reshape(a,(nphi,ntheta,nr),order='F')
    a = fread(fid, nr*ntheta*nphi, np.float32)
    Vt = np.reshape(a,(nphi,ntheta,nr),order='F')
    a = fread(fid, nr*ntheta*nphi, np.float32)
    Vp = np.reshape(a,(nphi,ntheta,nr),order='F')
    a = fread(fid, nr*ntheta*nphi, np.float32)
    Br = np.reshape(a,(nphi,ntheta,nr),order='F')
    a = fread(fid, nr*ntheta*nphi, np.float32)
    Bt = np.reshape(a,(nphi,ntheta,nr),order='F')
    a = fread(fid, nr*ntheta*nphi, np.float32)
    Bp = np.reshape(a,(nphi,ntheta,nr),order='F')
    a = fread(fid, nr*ntheta*nphi, np.float32)
    T = np.reshape(a,(nphi,ntheta,nr),order='F')

    fid.close()

    return (version, time, DeltaU, Coriolis, Lorentz, Buoyancy, ForcingU,
            DeltaT, ForcingT, DeltaB, ForcingB, Ek, Ra, Pm, Pr,
            nr, ntheta, nphi, azsym, radius, theta, phi, Vr, Vt, Vp,
            Br, Bt, Bp, T)

def surfaceload(filename):
    '''PARODY-JA4.3/Matlab/surfaceload_v4.m equivalent'''
    fid = open(filename, 'rb')

    version = fread(fid, 1, np.int16)
    version = version[0][0]
    dummy = fread(fid, 1, np.int16)
    phypar = fread(fid, 11, np.float64)
    # Simulation parameters
    time = phypar[0][0]
    dt = phypar[1][0]
    DeltaU = phypar[2][0]
    Coriolis = phypar[3][0]
    Lorentz = phypar[4][0]
    Buoyancy = phypar[5][0]
    ForcingU = phypar[6][0]
    DeltaT = phypar[7][0]
    ForcingT = phypar[8][0]
    DeltaB = phypar[9][0]
    ForcingB = phypar[10][0]

    Ek = 1/Coriolis
    Ra = Buoyancy*Ek
    Pm = 1/DeltaB
    if ( DeltaT != 0. ):
        Pr = 1/DeltaT
    else:
        Pr=0.

    # Grid parameters
    gridpar = fread(fid, 8, np.int16)
    nr = int(gridpar[0][0])
    ntheta = int(gridpar[2][0])
    nphi = int(gridpar[4][0])
    azsym = int(gridpar[6][0])

    radius = fread(fid, nr, np.float64)
    radius = radius.transpose()[0]
    theta = fread(fid, ntheta, np.float64)
    theta = theta.transpose()[0]
    phi = np.arange(1,nphi+1)*2*np.pi/(nphi*azsym)

    Vt = np.zeros((nphi,ntheta,nr))
    Vp = np.zeros((nphi,ntheta,nr))
    Br = np.zeros((nphi,ntheta,nr))
    dtBr = np.zeros((nphi,ntheta,nr))

    a = fread(fid, ntheta*nphi, np.float32)
    Vt = np.reshape(a,(nphi,ntheta),order='F')
    a = fread(fid, ntheta*nphi, np.float32)
    Vp = np.reshape(a,(nphi,ntheta),order='F')
    a = fread(fid, ntheta*nphi, np.float32)
    Br = np.reshape(a,(nphi,ntheta),order='F')
    a = fread(fid, ntheta*nphi, np.float32)
    dtBr = np.reshape(a,(nphi,ntheta),order='F')

    fid.close()

    return (version, time, DeltaU, Coriolis, Lorentz, Buoyancy, ForcingU,
            DeltaT, ForcingT, DeltaB, ForcingB, Ek, Ra, Pm, Pr,
            nr, ntheta, nphi, azsym, radius, theta, phi, Vt, Vp, Br,
            dtBr)

def load_parody_serie(run_ID, directory, tstart=0, t_sampling=1):
    '''Serie of Gt files
    This function reads several Gt files at different times
    with a sampling rate of t_sampling.

    Inputs:
        - filename: name of the file to read ( = directory+Gt=time+run_ID)
        - tstart: time of the first Gt file read
        - t_sampling: sampling rate of Gt files (should be >=1)
    Outputs:
        - All components of V, B and T for all times[::t_sampling]
    '''
    Gt_file_l = list_Gt_files(run_ID,directory) # Find all Gt_no in folder
    ntimes = len(Gt_file_l)

    #-- Loop over time
    time_t, Vt_t, Vp_t, Vr_t, Br_t, Bt_t, Bp_t, T_t = [
        [] for _ in range(8)]
    i=0
    for file in Gt_file_l[tstart::t_sampling]:
        print('Loading {} (({}/{})/{})'.format(file, i+1, ntimes//t_sampling, ntimes))
        filename = '{}/{}'.format(directory,file)
        (_, time, _, _, _, _, _,
            _, _, _, _, _, _, _, _,
            _, _, _, _, radius, theta, phi, Vr, Vt, Vp,
            Br, Bt, Bp, T) = parodyload(filename)
        #-- Append Time series of Graph files
        time_t.append(time)
        Vr_t.append(Vr)
        Vt_t.append(Vt)
        Vp_t.append(Vp)
        Br_t.append(Br)
        Bt_t.append(Bt)
        Bp_t.append(Bp)
        T_t.append(T)
        i += 1
    
    #-- return np.ndarray instead of a list
    time_t = np.array(time_t); T_t = np.array(T_t)
    Vr_t = np.array(Vr_t); Vt_t = np.array(Vt_t); Vp_t = np.array(Vp_t)
    Br_t = np.array(Br_t); Bt_t = np.array(Bt_t); Bp_t = np.array(Bp_t)

    return (radius, theta, phi, time_t, Vr_t, Vt_t, Vp_t, Br_t, Bt_t, Bp_t, T_t)


def load_surface_serie(run_ID, directory, tstart=0, t_sampling=1):
    '''Serie of St files
    This function reads several St files at different times
    with a sampling rate of t_sampling.

    Inputs:
        - filename: name of the file to read ( = directory+St=time+run_ID)
        - tstart: time of the first Gt file read
        - t_sampling: sampling rate of St files (should be >=1)
    Outputs:
        - Surface fields Vt, Vp, Br and dtBr for all times[::t_sampling]
    '''
    St_file_l = list_St_files(run_ID,directory) # Find all Gt_no in folder
    ntimes = len(St_file_l)

    #-- Loop over time
    time_t, Vt_st, Vp_st, Br_st, dtBr_st = [[] for _ in range(5)]
    i=0
    for file in St_file_l[tstart::t_sampling]:
        print('Loading {} (({}/{})/{})'.format(file, i+1, ntimes//t_sampling, ntimes))
        filename = '{}/{}'.format(directory,file)
        (_, time, _, _, _, _, _,
          _, _, _, _, _, _, _, _,
            _, _, _, _, _, theta, phi, Vt, Vp, Br,
            dtBr) = surfaceload(filename)
        #-- Append Time series of Surface files
        time_t.append(time)
        Vt_st.append(Vt)
        Vp_st.append(Vp)
        Br_st.append(Br)
        dtBr_st.append(dtBr)
        i += 1
    
    #-- return np.ndarray instead of a list
    time_t = np.array(time_t)
    Vt_st = np.array(Vt_st); Vp_st = np.array(Vp_st)
    Br_st = np.array(Br_st); dtBr_st = np.array(dtBr_st)

    return (theta, phi, time_t, Vt_st, Vp_st, Br_st, dtBr_st)

def load_basefield(filename,nr,ntheta,nphi):
    '''PARODY-JA4.56-Base/Base_Field/load_basfield_v4.m equivalent
    This function reads base fields created by makebasefield.m and
    stored in basefield.mat files.
    Need to rely on h5py to read file.mat files.

    Inputs:
        - filename: name of the file to read (=basefield.mat)
        - grid parameters: nr, ntheta, nphi
    Outputs:
        - 3D components of B0 and 3D components of j0
    '''
    import h5py
    f = h5py.File(filename, 'r')
    print('Reading base field with Fields: %s' % f.keys())
    fkeys = list(f.keys())

    B0r = np.transpose(f['Brbase'])
    B0t = np.transpose(f['Btbase'])
    B0p = np.transpose(f['Bpbase'])
    j0r = np.transpose(f['Jrbase'])
    j0t = np.transpose(f['Jtbase'])
    j0p = np.transpose(f['Jpbase'])

    f.close()

    return (B0r, B0t, B0p, j0r, j0t, j0p)

#------------------------------------------------------------------------------
def list_Gt_files(run_ID,directory):
    '''Make a list of all Gt file names'''
    Gt_file = []
    for files in os.walk(directory+"/"):
        for file in files[2]:
            if file.startswith('Gt=') and file.endswith('.{}'.format(run_ID)):
                Gt_file.append(file)
    Gt_file = sorted(Gt_file)

    return Gt_file

def list_St_files(run_ID, directory):
    '''Make a list of all St file names'''
    St_file = []
    for files in os.walk(directory+"/"):
        for file in files[2]:
            if file.startswith('St=') and file.endswith('.{}'.format(run_ID)):
                St_file.append(file)
    St_file = sorted(St_file)

    return St_file

def list_Dt_files(run_ID,directory):
    '''Make a list of all Dt file names'''
    Dt_file = []
    for files in os.walk(directory+"/"):
        for file in files[2]:
            if file.startswith('Gt=') and file.endswith('.{}'.format(run_ID)):
                Dt_file.append(file)
    Dt_file = sorted(Dt_file)

    return Dt_file

def load_dimensionless(run_ID, directory):
    '''Load Important parameters from a log.TAG file'''
    filename = '{}/log.{}'.format(directory, run_ID)

    f = open(filename, "r")
    #lines = [29, 50, 54, 55, 59, 73, 75]
    lines = [30, 42, 48, 51, 52, 57, 67, 68, 74]
    i = 0
    final_list = []
    for line in f:
        if i in lines:
            match_number = re.compile(
                '-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
            final_list.append([float(x)
                              for x in re.findall(match_number, line)])
        i = i+1
    f.close()

    ms = int(final_list[1][0]) # azimuthal symmetries
    NR = final_list[0][0]
    Ra = final_list[2][0]
    Ek = final_list[3][0]
    try:
        Pr = final_list[4][0]
    except IndexError: # for runs without Heat Equation (with B_0)
        Pr = 0.
    Pm = final_list[5][0]
    try:
        mcouple = final_list[6][0]
    except IndexError:  # for runs with no couple
        mcouple = 0
    try:
        rstrat = final_list[7][0]
        fi = final_list[8][0]/100  # divide percentage by 100
    except IndexError:  # for runs with no stratification
        shell_gap = cmb_radius - icb_radius
        rstrat = icb_radius/shell_gap
        fi = 0

    return (NR, Ek, Ra, Pr, Pm, ms, mcouple, fi, rstrat)
