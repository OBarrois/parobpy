#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 2023
@author: obarrois

Loads graphics file of the zonal velocity and plots the Torsional waves Width and max. Amplitude.
To be checked against (Jault, 2008) theory: \delta = S^{-1/4}; max-Amp = S^{1/2}.

Script to ONLY plot the TW width and Amplitude (one needs to use an other script to compute the TW t-s diagrams used here)!
"""

#import os
#import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import cmocean.cm as cmo
import h5py

l_show = True # allow Figures to be displayed or not 
if ( not l_show ): #-- Recommended if running the script only to compute and store Figures!
    import matplotlib
    matplotlib.use('Agg')  # backend for no display if needed

#----------------------------------------------------------------------------%%
#-- INPUT PARAMETERS

#-- Lehnert number, \lambda = B/{sqrt(rho mu) * Omega*d}.
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '3e7', 'Pm0o25', 1.46e-3, 1.0, 'b3', '0.000280800' # 3e-7 S=1214.1
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Lund8e2', 1.1e-3, 0.5, 'b4-5', '0.000288000' # 2e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1', 1.1e-3, 1.0, 'b4', '0.000040320' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'PathLund3o2e3', 5.53e-4, 1.0, 'b4o6', '0.000015600' # 6.3e-9 (1e7 grid)
run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e8', 'PathLund3o2e3', 5.53e-4, 1.0, 'b4o6-2', '0.000008800' # 6.3e-9 (1e8 grid)
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e8', 'PathLund3o2e3BIS', 5.53e-4, 1.0, 'b4o6-2', '0.000009080' # 6.3e-9 (1e8 grid)BIS
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e8', 'Pm0o46e-1', 6.2e-4, 1.0, 'b4o5', '0.000004800' # 1e-8
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '3e10', '', 2.6e-4, 1.0, 'b5', '0.000001456' # 3e-10

#saveDir = '/gpfs/users/obarrois/Parodys/Outputs/Data/' # if you use python directly from cluster
#directory = '/gpfs/users/obarrois/Work/IC_Impulse/Waves1e7/' # if you use python directly from cluster
directory = '/Users/obarrois/Desktop/stella/Work/IC_Impulse/Waves'+run_Ek+'/'+run_Pm+'/' # if you mount stella on own desk

rank = 0 # default proc number when l_mpi == False or None

l_check_all = False # display all check plot figures
l_resym = True # re-symmetrizing the data if minc !=1
l_redim = True # re-dimensionalisation of the quantities --> has to be True here to compare all Runs
l_tau_A = True # re-dimensionalisation on the Alfven or on the Viscous time-scales

#-- I/O; NOTE: does not seem to replace files if they exist even if l_save*!
saveDir = '/Users/obarrois/Desktop/Parodys/Outputs/IC_Impulse/Waves'+run_Ek+'/Balances/Data/'#Old_Computations/ # path to save colmunar balance files
tagRead = '_-phi-Avg_n500_highLund'#_-phi-Avg_n500'#_-phi-Avg_n500'#_-phi-Avg_n500_lowLund'#_-phi-Avg_n600'#_-phi-Avg_n500_lowLehn'#_-phi-Avg_n500_lowerPm'# #  # tag at the end of the forces files if needed
l_save = 0 # save main figures?
savePlot = '/Users/obarrois/Desktop/Parodys/Outputs/' # path to save main figures

#------------------------------------------------------------------------------
#%% Extract TW Width and max Amplitude

#-- Setting up all different S cases (6 different cases)
ExtraDir='Images/Lund-Modified_fortran-zAvg_-phi-Avg/'#PathLund3o2e3 _-phi-Avg_n500_pathLund #+'PathLund3o2e3BIS_TimeStep-Diff'
#Number =   [ 0,                 1,                 2,                          3,                          4,                              5,                              6,                  7,                          8,                  9]#,               10]#]
S_Ek =      [3e-7,              1e-7,              2e-7,                       6.3e-9,                     5e-8,                           2e-8,                           1e-8,               6.3e-9,                     6.3e-9,             3e-10,            1e-7]#]
S_run_Ek =  ['3e7',             '1e7',             '1e7',                      '1e7',                      '1e7',                          '1e7',                          '1e8',              '1e8',                      '6e9',              '3e10',           '1e7']#]
S_extra =   ['',                '',                ExtraDir+'Lund8o0e2',       ExtraDir+'PathLund3o2e3',   ExtraDir+'Lund3o2e3',           ExtraDir+'Lund1o6e4',           '',                 ExtraDir+'PathLund3o2e3',   '',                 '',               'Images/fortran-zAvg_fixed-zAvg_-phi-Avg/Benchmark_Hyperdiffusion/lh60_qh1o05']#]
S_run_Pm =  ['Pm0o25',          'Pm1o44e-1',       'Lund8e2'                   'PathLund3o2e3',            'Lund3o2e3',                    'Lund1o6e4',                    'Pm0o46e-1',        'PathLund3o2e3',            'Pm0o36e-1',        '',               'Pm1o44e-1']#]
S_Lehnert = [1.46e-3,           1.1e-3,            1.1e-3,                     5.53e-4,                    1.1e-3,                         1.1e-3,                         6.2e-4,             5.53e-4,                    5.53e-4,            2.6e-4,           1.1e-3]#]
S_Lundquist=[1.21e3,            1.6e3,             8.e2,                       3.2e3,                      3.2e3,                          8.e3,                           2.85e3,             3.2e3,                      3.2e3,              6.8e3,            1.6e3]#]
S_tagRead = ['_-phi-Avg_n500',  '_-phi-Avg_n500',  '_-phi-Avg_n500_lowLund',   '_-phi-Avg_n500_pathLund',  '_-phi-Avg_n500_lowLund3o2e3',  '_-phi-Avg_n500_lowLund1o6e4',  '_-phi-Avg_n500',   '_-phi-Avg_n500_highLund',  '_-phi-Avg_n500',   '_-phi-Avg_n600', '_-phi-Avg_n500_lh60qh1o05']#]
#-- Reading through all different S cases:
SsrS = np.zeros((len(S_tagRead), 1248),) #dimensions has to be the max of the runs (600, 1248) for Ek=3e-10
StdmS = np.zeros((len(S_tagRead), 600),) #dimensions has to be the max of the runs (600, 1248) for Ek=3e-10
VpcolS = np.zeros((len(S_tagRead), 600, 1248),) #dimensions has to be the max of the runs (600, 1248) for Ek=3e-10
for i in range(len(S_tagRead)):
    SsaveDir = '/Users/obarrois/Desktop/Parodys/Outputs/IC_Impulse/Waves'+S_run_Ek[i]+'/Balances/'+S_extra[i]+'/Data/'#Old_Computations/ # path to save colmunar balance files
    #-- Actually reading Vpcol field
    f = h5py.File(SsaveDir+'Zavg_bal'+S_tagRead[i]+'.hdf5', 'r')
    print('Reading columnar balances with Fields: %s' % f.keys())
    fkeys = list(f.keys())
    Ssr = np.array(f['srad'])
    Stimet = np.array(f['time'])
    Vpcolr = np.array(f['Vpcol'])
    #
    SsrS[i,:len(Ssr)] = Ssr.copy()
    StdmS[i,:len(Stimet)] = Stimet.copy()*1./(S_Ek[i]/S_Lehnert[i])#--> redim time
    VpcolS[i,:len(Stimet),:len(Ssr)] = Vpcolr.copy()

#-- Dispatch fields
sr = SsrS[1,:401].copy()
tdimp = Stimet.copy()*1./(3.e-10/2.6e-4)#1./(S_run_Ek[i]/S_Lehnert[i])# last one = Ek=3e10; S=6.8e3
Vpcol0 = VpcolS[0,:500,:323].copy(); td0 = StdmS[0,:500].copy(); sr0 = SsrS[0,:323].copy() # Ek=3e7; S=1.21e3
Vpcol1 = VpcolS[1,:500,:401].copy(); td1 = StdmS[1,:500].copy(); sr1 = SsrS[1,:401].copy() # Ek=1e7; S=1.6e3
Vpcol10 = VpcolS[2,:500,:401].copy(); td10 = StdmS[2,:500].copy(); sr10 = SsrS[2,:401].copy() # Ek=2e7; S=8e2
Vpcol211 = VpcolS[3,:500,:401].copy(); td211 = StdmS[3,:500].copy(); sr211 = SsrS[3,:401].copy() # Ek=6.3e-9; S=3.2e3 (1e7 grid)
Vpcol21 = VpcolS[4,:500,:401].copy(); td21 = StdmS[4,:500].copy(); sr21 = SsrS[4,:401].copy() # Ek=5e8; S=3.2e3
Vpcol24 = VpcolS[5,:500,:401].copy(); td24 = StdmS[5,:500].copy(); sr24 = SsrS[5,:401].copy() # Ek=2e8; S=8e3
Vpcol2 = VpcolS[6,:500,:669].copy(); td2 = StdmS[6,:500].copy(); sr2 = SsrS[6,:669].copy() # Ek=1e8; S=2.85e3
Vpcol212 = VpcolS[7,:500,:669].copy(); td212 = StdmS[7,:500].copy(); sr212 = SsrS[7,:669].copy() # Ek=6.3e-9; S=3.2e3 (1e8 grid)
Vpcol23 = VpcolS[8,:500,:961].copy(); td23 = StdmS[8,:500].copy(); sr23 = SsrS[8,:961].copy() # Ek=6.3e-9; S=3.2e3
Vpcol3 = VpcolS[9,:,:].copy(); td3 = StdmS[9,:].copy(); sr3 = SsrS[9,:].copy() # Ek=3e10; S=6.8e3
VpcolEx = VpcolS[10,:500,:401].copy(); tdEx = StdmS[10,:500].copy(); srEx = SsrS[10,:401].copy() # Ek=2e7; S=1.6e3 (1e7 grid); # Ek=7e8; S=1.6e3 (1e7 grid); 

#-- Plot TW width for observation purposes
if ( l_check_all ):
    stim = 300
    tdim_plot = td1[stim]
    plt.figure()
    plt.plot()
    stim = 312#301#
    plt.plot(sr0, Vpcol0[stim, :]/Vpcol0[stim, :].max(), 'b-*', alpha=0.5, label='Normalised TO, Ek = {}; Lund (low) = {}'.format('3e-7', '1o21e3'))
    #stim = 303
    #plt.plot(sr10, Vpcol10[stim, :]/Vpcol10[stim, :].max(), 'b--x', alpha=0.5, label='Normalised TO, Ek = {}; Lund (low) = {}'.format('2e-7', '8e2'))
    stim = 300
    plt.plot(sr1, Vpcol1[stim, :]/Vpcol1[stim, :].max(), 'k-*', alpha=0.8, label='Normalised TO, Ek = {}; Lund (start) = {}'.format('1e-7', '1o6e3'))
    #stim = 306#301
    #plt.plot(sr211, Vpcol211[stim, :]/Vpcol211[stim, :].max(), 'k:o', alpha=0.8, label='Normalised TO, Ek = {}; Lund (high, grid=1e7) = {}'.format('6.3e-9', '3o2e3'))
    #plt.plot(sr21, Vpcol21[298, :]/Vpcol21[298, :].max(), 'y--x', alpha=0.5, label='Normalised TO, Ek = {}; Lund (high) = {}'.format('5e-8', '3o2e3'))
    #plt.plot(sr24, Vpcol24[298, :]/Vpcol24[298, :].max(), 'r--x', alpha=0.5, label='Normalised TO, Ek = {}; Lund (highest) = {}'.format('2e-8', '8e3'))
    stim = 265#266#303#301# #-- to get the same dimensionalised time!
    plt.plot(sr2, Vpcol2[stim, :]/Vpcol2[stim, :].max(), 'm-*', alpha=0.8, label='Normalised TO, Ek = {}; Lund (high) = {}'.format('1e-8', '2o85e3'))
    #stim = 306#301
    #plt.plot(sr212, Vpcol212[stim, :]/Vpcol212[stim, :].max(), 'y:o', alpha=0.8, label='Normalised TO, Ek = {}; Lund (high, grid=1e8) = {}'.format('6.3e-9', '3o2e3'))
    stim = 306#301#287 #-- to get the same dimensionalised time!
    plt.plot(sr23, Vpcol23[stim, :]/Vpcol23[stim, :].max(), 'y-*', alpha=0.8, label='Normalised TO, Ek = {}; Lund (high) = {}'.format('6.3e-9', '3o2e3'))
    stim = 305 #-- to get the same dimensionalised time!
    plt.plot(sr3, Vpcol3[int(stim), :]/Vpcol3[stim, :].max(), 'c-*', alpha=0.8, label='Normalised TO, Ek = {}; Lund (highest) = {}'.format('3e-10', '6o8e3'))
    #plt.plot(sr3, Vpcol3[int(stim/500*600), :]/Vpcol3[stim, :].max(), 'c-*', alpha=0.8, label='Normalised TO, Ek = {}; Lund (highest) = {}'.format('3e-10', '6o8e3'))
    plt.plot(srEx, VpcolEx[300, :]/VpcolEx[300, :].max(), 'r--+', alpha=0.8, label='Normalised TO, Ek = {}; Lund (variable) = {}'.format('variable', 'variable'))
    plt.plot(sr, np.zeros((len(sr)),), 'k-')
    plt.xlim(sr[-1],sr[0])
    plt.ylim(-0.5,1)
    plt.xlabel('cylindrical radius')
    plt.ylabel('Normalised zAvg field')
    plt.title('Normalised Vpcol at different Lundquist and time = {}'.format(np.round(tdim_plot,2)))
    plt.grid(ls=':')
    plt.legend()
    plt.tight_layout()
    plt.show()

#-- witdh (NOTE: Observed WITH SAME TDIM!!)
l0 = 1.21e3;    lh0 = 1.46e-3;  em0 = 3e-7/0.25;        w0= 1.238871 - 0.780668;    m0 = Vpcol0.max()     #-- S=1.21e3 (3.e-7 grid)
l10 = 8e2;      lh10 = 1.1e-3;  em10 = 2e-7/0.144;      w10= 1.246824 - 0.784147;   m10 = Vpcol10.max()     #-- S=8.0e2 (1.e-7 grid)
l1 = 1.6e3;     lh1 = 1.1e-3;   em1 = 1e-7/0.144;       w1 = 1.232416 - 0.808648;   m1 = Vpcol1.max()
l211 = 3.2e3;   lh211 = 5.53e-4;em211 = 6.3e-9/0.0365;  w211 = 1.200834 - 0.840648; m211 = Vpcol211.max()     #-- S=3.2e3 (1.e-7 grid)
l2 = 2.85e3;    lh2 = 6.2e-4;   em2 = 1e-8/0.046;       w2 = 1.208074 - 0.836204;   m2 = Vpcol2.max()      #--  S=2.85e3 (1.e-8) --> Pm0o46e-1 Original TimeStep
#l2 = 2.85e3;    lh2 = 6.2e-4;   em2 = 1e-8/0.046;       w2 = 1.200444 - 0.828097;   m2 = Vpcol2.max()       #--  S=2.85e3 (1.e-8) --> BIS =TimeStep-Diff!
l212 = 3.2e3;   lh212 = 5.53e-4;em212 = 6.3e-9/0.0365;  w212 = 1.200023 - 0.842116; m212 = Vpcol212.max()     #-- S=3.2e3 (1.e-8 grid)
#l212 = 3.2e3;   lh212 = 5.53e-4;em212 = 6.3e-9/0.0365;  w212 = 1.200083 - 0.842047; m212 = Vpcol212.max()     #-- S=3.2e3 (1.e-8 grid) --> BIS =TimeStep-Diff!
l23 = 3.2e3;    lh23 = 5.53e-4; em23 = 6.3e-9/0.0365;   w23= 1.199429 - 0.844158;   m23 = Vpcol23.max()     #-- S=3.2e3 (6.3e-9 grid)
#l23 = 3.2e3;    lh23 = 5.53e-4; em23 = 6.3e-9/0.0365;   w23= 1.199427 - 0.844157;   m23 = Vpcol23.max()     #-- S=3.2e3 (6.3e-9 grid) --> BIS =TimeStep-Diff!
l3 = 6.8e3;     lh3 = 2.6e-4;   em3 = 3e-10/7.9e-3;     w3 = 1.181878 - 0.880915;   m3 = Vpcol3.max()

#-- witdh (NOTE: Observed WITH SAME TDIM + ALIGNED!!)
l0 = 1214;    lh0 = 1.46e-3;  em0 = 3e-7/0.25;        w0= 1.263274 - 0.802899;    m0 = Vpcol0[312].max()     #-- S=1.21e3 (3.e-7 grid)
l10 = 798;      lh10 = 1.11e-3;  em10 = 2e-7/0.144;      w10= 1.255575 - 0.791360;   m10 = Vpcol10[303].max()     #-- S=8.0e2 (Ek=2.e-7, 1.e-7 grid)
l1 = 1596;     lh1 = 1.11e-3;   em1 = 1e-7/0.144;       w1 = 1.232406 - 0.808645;   m1 = Vpcol1[300].max()
l1h1 = 1596;     lh1h1 = 1.11e-3;   em1h1 = 1e-7/0.144;       w1h1 = 1.232404 - 0.808647;   m1h1 = 9.995237      #-- Testing Hyperdiffusion: 1.e-7 ; lh30 qh1o025
l1h2 = 1596;     lh1h2 = 1.11e-3;   em1h2 = 1e-7/0.144;       w1h2 = 1.232818 - 0.807409;   m1h2 = 9.996328      #-- Testing Hyperdiffusion: 1.e-7 ; lh60 qh1o05
l1h3 = 1596;     lh1h3 = 1.11e-3;   em1h3 = 1e-7/0.144;       w1h3 = 1.232404 - 0.808646;   m1h3 = 9.986944      #-- Testing Hyperdiffusion: 1.e-7 ; lh100 qh1o03
l211 = 3201;   lh211 = 5.53e-4;em211 = 6.3e-9/0.0365;  w211 = 1.213551 - 0.849143; m211 = Vpcol211[306].max()     #-- S=3.2e3 (6.3e-9, 1.e-7 grid)
l2 = 2853;    lh2 = 6.20e-4;   em2 = 1e-8/0.046;       w2 = 1.202635 - 0.833419;   m2 = Vpcol2[265].max()      #--  S=2.85e3 (1.e-8) --> Pm0o46e-1 Original TimeStep
#l2 = 2853;    lh2 = 6.20e-4;   em2 = 1e-8/0.046;       w2 = 1.205635 - 0.833423;   m2 = Vpcol2.max()       #--  S=2.85e3 (1.e-8) --> BIS =TimeStep-Diff!
l212 = 3201;   lh212 = 5.53e-4;em212 = 6.3e-9/0.0365;  w212 = 1.213618 - 0.850781; m212 = Vpcol212[306].max()     #-- S=3.2e3 (6.3e-9, 1.e-8 grid)
#l212 = 3201;   lh212 = 5.53e-4;em212 = 6.3e-9/0.0365;  w212 = 1.213705 - 0.850965; m212 = Vpcol212.max()     #-- S=3.2e3 (1.e-8 grid) --> BIS =TimeStep-Diff!
l23 = 3201;    lh23 = 5.53e-4; em23 = 6.3e-9/0.0365;   w23= 1.212938 - 0.852902;   m23 = Vpcol23[306].max()     #-- S=3.2e3 (6.3e-9 grid)
#l23 = 3201;    lh23 = 5.53e-4; em23 = 6.3e-9/0.0365;   w23= 1.212941 - 0.852899;   m23 = Vpcol23.max()     #-- S=3.2e3 (6.3e-9 grid) --> BIS =TimeStep-Diff!
l3 = 6825;     lh3 = 2.59e-4;   em3 = 3e-10/7.9e-3;     w3 = 1.181874 - 0.880917;   m3 = Vpcol3[305].max()
#
l111 = 1596; lh111 = 2.22e-3; em111 = 2e-7/0.144;     w111 = 1.274104 - 0.790351;     m111 =  8.489377         #m111 = VpcolEx[306].max() #-- S=1.6e3 (Ek=2.e-7, 1.e-7 grid) --> #Pulsation Okay (S and Pm unchanged!)
l112 = 1596; lh112 = 5.54e-4; em112 = 5e-8/0.144;     w112 = 1.229892 - 0.836394;     m112 = 11.472847         #m112 = VpcolEx[296].max() #-- S=1.6e3 (Ek=5.e-8, 1.e-7 grid) --> #Pulsation Okay (S and Pm unchanged!)
l113 = 1596; lh113 = 2.22e-4; em113 = 2e-8/0.144;     w113 = 1.202922 - 0.834559;     m113 = 12.710137         #m113 = VpcolEx[300].max() #-- S=1.6e3 (Ek=2.e-8, 1.e-7 grid) --> #Pulsation Okay (S and Pm unchanged!)
l114 = 2257; lh114 = 7.84e-4; em114 = 5e-8/0.144;     w114 = 1.225784 - 0.827716;     m114 = 16.256321         #m114 = VpcolEx[157].max() #-- S=2.3e3 (Ek=5.e-8, 1.e-7 grid) --> #W=Wrong-Pulsation #--Lehn2=Lehn7o7e-4
l100 =  505; lh100 = 3.50e-3; em100 = 1e-7/0.0144;    w100 = 1.312785 - 0.751625;     m100 = 17.760969         #m100 = VpcolEx[207].max() #-- S=5.0e2 (Ek=1.e-7, 1.e-7 grid) --> #W=Wrong-Pulsation #--Pm1o44e-2 #NOTE: WARNING!!: Spurious point (outlier)!
l110 = 1129; lh110 = 1.57e-3; em110 = 2e-7/0.144;     w110 = 1.273338 - 0.804302;     m110 = 6.2060032         #m110 = VpcolEx[144].max() #-- S=1.1e3 (Ek=2.e-7, 1.e-7 grid) --> #W=Wrong-Pulsation #--Lund2=Lund1o1e2
l120 = 1129; lh120 = 1.57e-3; em120 = 1e-7/0.072;     w120 = 1.272958 - 0.801678;     m120 = 12.295158         #m120 = VpcolEx[221].max() #-- S=1.1e3 (Ek=1.e-7, 1.e-7 grid) --> #W=Wrong-Pulsation by a factor 2.0 (Scale as Pm/S) #--PmHalf
l130 = 1009; lh130 = 1.40e-3; em130 = 1e-7/0.072;     w130 = 1.256672 - 0.792556;     m130 = 9.5671835         #m130 = VpcolEx[270].max() #-- S=1.0e3 (Ek=1.e-7, 1.e-7 grid) --> #W=Wrong-Pulsation #--ElssHalf
l121 = 1590; lh121 = 1.11e-3; em121 = 7e-8/0.1;       w121 = 1.234046 - 0.807999;     m121 = 14.226600         #m121 = VpcolEx[283].max() #-- S=1.6e3 (Ek=7.e-8, 1.e-7 grid) --> #W=Wrong-Pulsation by a factor 1.44 (Scale as Pm/S) #--Pm1o0e-1
l122 = 1596; lh122 = 1.11e-3; em122 = 2e-7/0.288;     w122 = 1.232064 - 0.810267;     m122 = 5.0382891         #m122 = VpcolEx[295].max() #-- S=1.6e3 (Ek=2.e-7, 1.e-7 grid) --> #W=Wrong-Pulsation by a factor 2.0 (Scale as Pm/S) #--Pm2o88e-1
l21 = 3192;  lh21 = 1.11e-3;  em21 = 5e-8/0.144;      w21 = 1.221470 - 0.819780;      m21 = Vpcol21[298].max() #-- S=3.2e3 (Ek=5.e-8, 1.e-7 grid) --> #W=Wrong-Pulsation
l24 = 7980;  lh24 = 1.11e-3;  em24 = 2e-8/0.144;      w24 = 1.219919 - 0.836011;      m24 = Vpcol24[298].max() #-- S=8.0e3 (Ek=2.e-8, 1.e-7 grid) --> #W=Wrong-Pulsation

#- All Data. NOTE: Warning!:: Low/high lund/lehn and 'other runs' have to be re-done to scale tau^*
ekm = [em0, em10, em110, em120, em111, em1, em1h1, em1h2, em1h3, em121, em122, em112, em113, em114, em130, em2, em21, em211, em212, em23, em3, em24,  em100]#
wid = [w0,  w10,  w110,  w120,  w111,  w1,  w1h1,  w1h2,  w1h3,  w121,  w122,  w112,  w113,  w114,  w130,  w2,  w21,  w211,  w212,  w23, w3,  w24,  w100]#
lun = [l0,  l10,  l110,  l120,  l111,  l1,  l1h1,  l1h2,  l1h3,  l121,  l122,  l112,  l113,  l114,  l130,  l2,  l21,  l211,  l212,  l23, l3,  l24,  l100]#
leh = [lh0, lh10, lh110, lh120, lh111, lh1, lh1h1, lh1h2, lh1h3, lh121, lh122, lh112, lh113, lh114, lh130, lh2, lh21, lh211, lh212, lh23, lh3, lh24,  lh100]#
mid = [m0,  m10,  m110,  m120,  m111,  m1,  m1h1,  m1h2,  m1h3,  m121,  m122,  m112,  m113,  m114,  m130,  m2,  m21,  m211,  m212,  m23,  m3,  m24,  m100]#
#
#- Selected Data: Path only re-observed data at same tdim!
ekmp = [em0, em1, em1h1, em1h2, em1h3, em2, em23, em3]
widp = [w0, w1,  w1h1,  w1h2,  w1h3, w2, w23, w3]
lunp = [l0, l1,  l1h1,  l1h2,  l1h3, l2, l23, l3]
lehp = [lh0, lh1, lh1h1, lh1h2, lh1h3, lh2, lh23, lh3]
midp = [m0, m1,  m1h1,  m1h2,  m1h3, m2, m23, m3]

#------------------------------------------------------------------------------
#%% Plot TW Width and max Amplitude change with Multi-parameters!

eps = 1.e-3
plt.rcParams['text.usetex'] = True
#-- S scaling
lin1 = np.linspace(3e2,2e4,50)
law1 = 2.75*lin1**(-1/4)
#-- Figure
fig = plt.figure(figsize=(22, 6.5))
ax = plt.subplot(111)
ax.loglog(lun,wid,'bo', lw=2.1, ms=11, label=r'Observed Width')
ax.loglog(lin1,law1,'k--', lw=1.7,alpha=0.8, label=r'$S^{-1/4}$')#label=r'$3\,S^{-1/4}$')#
plt.xlim(min(lun)-eps,max(lun)-eps)
plt.xlabel(r'Lundquist, $S$', fontsize=36)
plt.ylabel(r'Width, $\delta_\textnormal{\small TW}$', fontsize=36)
plt.gca().xaxis.set_tick_params(which='major', labelsize=32)
plt.gca().xaxis.set_tick_params(which='minor', labelsize=32)
plt.gca().yaxis.set_tick_params(which='major', labelsize=32)
plt.gca().yaxis.set_tick_params(which='minor', labelsize=32)
plt.grid(ls=':')
plt.legend(fontsize=32)
transAx = mtransforms.ScaledTranslation(-1.95-10/72, -35/72, fig.dpi_scale_trans)
ax.text(0.0, 1.0, r'$(a)$', transform=ax.transAxes + transAx,
        fontsize=36, va='bottom', fontfamily='serif')
plt.tight_layout()
if ( l_save ):  plt.savefig(savePlot+'Scaling_TO_Width-vs-Lundquist_Path-Allp-Aligned.pdf')
#plt.show()

#-- EkM scaling
lin1 = np.linspace(3e-8,8e-5,50)
#law1 = 20.*lin1**(1/4)
law1 = 2.55*lin1**(1/8)
#-- Figure
fig = plt.figure(figsize=(22, 6.5))
ax = plt.subplot(111)
ax.loglog(ekm,wid,'yo', lw=2.1, ms=11, label=r'Observed Width')
ax.loglog(lin1,law1,'k--', lw=1.7,alpha=0.8, label=r'$(Ek_M)^{1/4}$')#label=r'$3\,S^{-1/4}$')#
plt.xlim(min(ekm),max(ekm))
plt.xlabel(r'Magnetic Ekman, $Ek_M$', fontsize=36)
plt.ylabel(r'Width, $\delta_\textnormal{\small TW}$', fontsize=36)
plt.gca().xaxis.set_tick_params(which='major', labelsize=32)
plt.gca().xaxis.set_tick_params(which='minor', labelsize=32)
plt.gca().yaxis.set_tick_params(which='major', labelsize=32)
plt.gca().yaxis.set_tick_params(which='minor', labelsize=32)
plt.grid(ls=':')
plt.legend(fontsize=32)
transAx = mtransforms.ScaledTranslation(-1.95-10/72, -35/72, fig.dpi_scale_trans)
ax.text(0.0, 1.0, r'$(b)$', transform=ax.transAxes + transAx,
        fontsize=36, va='bottom', fontfamily='serif')
plt.tight_layout()
if ( l_save ):  plt.savefig(savePlot+'Scaling_TO_Width-vs-EkMag_Path-Allp-Aligned.pdf')
plt.show()

#-- End Script
