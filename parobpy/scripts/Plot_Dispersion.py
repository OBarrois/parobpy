#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 07 2025
@author: obarrois

Python script to plot the evolution of the Dipersion relation for QG-MC and QG-Alfven waves
Rebuild or read data from csv field from simulations computed over a non-axisymmetric magnetic field background

Script to ONLY plot the resulting wave numbers from computations.
"""

#from parobpy.load_parody import parodyload, load_dimensionless, load_basefield
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.transforms as mtransforms
#import cmocean.cm as cmo

#-- Functions
#matplotlib.use('Agg')  # backend for no display if needed

#----------------------------------------------------------------------------%%
#-- INPUT PARAMETERS

#-- I/O; NOTE: does not seem to replace files if they exist even if l_save*!
l_save = 1 # save zAvg-balances?
savePlot = '/Users/obarrois/Desktop/Parodys/Outputs/Waves1e7/' # path to save files

l_phydim = True # physically redimensionning quantities?

#-- Constants of the problem
tauA = 2. #Alfvén time for the Earth ~2years
n2pi = 2*np.pi
tauA2pi = tauA*n2pi
d = 2258.5 #Size of the earth's core to physically redimensionalise
d2pi = d*n2pi
y2s = 3600.*24*365 #seconds in 1 year

#------------------------------------------------------------------------------
#%% Load data

#
#-- Yet an other atempt at finding the dispersion relation #-- NOTE: Values below imposed pulsation 1024pi are spurious (box "too small") and above 100000pi are also spurious (observation impossible)
pulse_visc=np.array([1024, 2048,  4096,   5000,   10000,    20000,   30000,  32768,    50000,    70000,    85000,   100000,   160000,   200000])*np.pi#/tredim #--> To get pulsation in Alfven time
wtrue=np.array([0.29245, 0.5849, 1.168, 1.4280, 2.85599, 5.711986, 8.56798, 9.3585, 14.27997, 19.99195, 24.27594, 28.55993, 45.69589, 57.11987])
l_disp_path = False
l_disp_norm = True
le = 1.1e-3
#-- PathBase Observations
#(, 2.07*1.34545,         41.5) --> it is the same point as (2.8, 41)
kspb = np.array([ 22,    27,   33,   34,   35,   41,   47,   56,   80,   94,  115,  125])#,  127])#,   104, 157, 190])#,         51.5])
wtpb = np.array([0.3,  0.61,  1.1,  1.3,  1.4,  2.8,  3.8,  5.7,  9.8, 14.3,  24., 26.8])#, 28.6])#,  20., 43., 55.])#, 2.78*1.34545])
lepb = 1.48e-3 #NOTE: Lehnert ratio lepb/le = 1.34545
#-- Simple (B0-Y33) Observations
ks = np.array([ 28,   32,  37,   38, 47.1, 65.4,  78, 79.6,  102, 131, 141])#, 198, 276, 450])#
wt = np.array([0.3, 0.59, 1.2, 1.41,  2.8,  5.7, 8.6,  9.4, 14.9,  21,  25])#, 28.7,  46,  55])#
#pulse_visc_add=np.array([2048, 4096, 20000, 50000, 85000, 100000])*np.pi#/tredim #--> To get pulsation in Alfven time
lehe = 1.57e-3
kshe = np.array([  25,  26,  36.,  47,   72,   95,  136])#,   70])#
wthe = np.array([0.81, 1.0, 1.98, 4.2, 10.1, 16.5, 19.9])#, 31.9])#
lele = 7.84e-4
ksle = np.array([  36,   44,  77,  145])#,  130,   95,  106])#
wtle = np.array([0.78, 1.57, 7.9, 20.0])#, 30.8, 33.8, 40.5])#
#-- B0-Y66 Observations
le66 = 1.45e-3#1.1e-3 #NOTE: BY66 change the Elsasser number
ks66 = np.array([  16,   17,   19,  38,   66,   96,  132])#, 138])#
wt66 = np.array([0.58, 1.19, 1.43, 5.7, 14.6, 24.5, 28.3])#, 48.])#
llin = np.linspace(30,150,50)
law2 = 2e-3*llin**2

#
#-- Or simply load csv files
#Header: #Label     Nr         lmax       l_H        q_h        Ek         Pm         Lundquist  Lehnert    Elssasser  w_input    w_output   ks_output  Waves-at-CMB
cols2read = np.linspace(1,12,12,dtype=np.int16)
dataS = np.loadtxt('/Users/obarrois/Desktop/Show/2025_Barrois_Aubert_Waves-QGA-Path/ascii_table_Simple-B0-Data.csv',skiprows=1,usecols=cols2read)
dataS = np.loadtxt('/Users/obarrois/Desktop/Show/2025_Barrois_Aubert_Waves-QGA-Path/ascii_table_Complex-B0-Data.csv',skiprows=1,usecols=cols2read)

#
#------------------------------------------------------------------------------
#%% Paper plot section

from matplotlib.ticker import LogFormatter, AutoMinorLocator, FormatStrFormatter
#
# plt.rcParams['text.usetex'] = True
plt.rcParams['text.usetex'] = True
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

#-- Theoretical dispersion relations ([Barrois and Aubert, 2025] from [Gillet et al., 2021])
l_theory = True # physically redimensionning quantities?
#-- Constants
ks0 = 44.8#45#
vath = 2.e-1#1.90e-1#
Lehnert = 1.48e-3
mth=3; hth = 0.34; Omegath = 1./Lehnert#1.e3#
#mth=6; hth = 1.; Omegath =1.e4#
ks0th = (mth*Omegath/(vath*hth**2))**(1/3.)
vathth = mth*Omegath/(ks0**3*hth**2)
#-- Full relation: Eq.2 [Barrois and Aubert, 2025]
ksth = np.linspace(1,401,201)
oksth= ks0/ksth#ks0th/ksth#
omthp = vath*ksth*(oksth**3 + np.sqrt(1 + oksth**6))
omthm = vath*ksth*(oksth**3 - np.sqrt(1 + oksth**6))

#-- Aflvén relation: Eq.5
omthAp = vath*ksth + mth*Omegath/(ksth*hth)**2
omthAm =-vath*ksth + mth*Omegath/(ksth*hth)**2
#-- Rossby relation: Eq.6
omthR = 2*mth*Omegath/(ksth*hth)**2
#-- QG-MC relation: Eq.7
omthMC =-(vath*hth)**2*ksth**4/(2*mth*Omegath)

#
#-- Fig.8 Barrois and Aubert 2025
#-- Plot Omega-vs-ks with or without units
#
fig = plt.figure(figsize=(21, 12.1))
ax = plt.subplot(111)
if ( l_phydim ):
    if ( l_theory ):
        ax.loglog(ksth/d2pi, abs(omthm)/tauA2pi, ls='-', color='k', lw=2.1, alpha=0.9, label=r'$\vert \omega_\mathrm{theoretical}^- \vert$ (Eq.2)')
        ax.loglog(ksth/d2pi, omthp/tauA2pi, ls='-.', color='k', lw=2.1, alpha=0.9, label=r'$\omega_\mathrm{theoretical}^+$')#  (Eq.2)')#
        #ax.loglog(ksth/d2pi, omthp/tauA2pi, ls='-', color='#1e1e1e', lw=1.9, alpha=0.5, label=r'$\omega_\mathrm{theoretical}^+$')#  (Eq.2)')#
        #ax.loglog(ksth[22:]/d2pi, abs(omthAm[22:])/tauA2pi, ls=':', color='#1e1e1e', lw=1.9, alpha=0.5, label=r'$\vert \omega_\mathrm{\cal A}^- \vert \rightarrow k_s^1$ (Eq.5)')
        ax.loglog(ksth[41:]/d2pi, abs(omthAm[41:])/tauA2pi-0.08, ls=':', color='#1e1e1e', lw=1.9, alpha=0.5, label=r'$\vert \omega_\mathrm{\cal A}^- \vert \propto k_s^1$ (Eq.5)')
        #ax.loglog(ksth[22:]/d2pi, omthAp[22:]/tauA2pi, ls=':', color='#1e1e1e', lw=1.9, alpha=0.2, label=r'$\omega_\mathrm{\cal A}^+ \sim k_s^1$')# (Eq.5)')#
        ax.loglog(ksth[:31]/d2pi, omthR[:31]/tauA2pi, ls='-.', color='#1e1e1e', lw=1.9, alpha=0.5, label=r'$\omega_\mathrm{R} \propto k_s^{-2}$ (Eq.6)')
        ax.loglog(ksth[:31]/d2pi, abs(omthMC[:31])/tauA2pi, ls='--', color='#1e1e1e', lw=1.9, alpha=0.5, label=r'$\vert \omega_\mathrm{MC} \vert \propto k_s^4$ (Eq.7)')
    else:
        #-- Revision 1:
        #ax.loglog(np.linspace(60,400,50)/d2pi, 2e-1*np.linspace(60,400,50)/tauA2pi, ls=':', color='#1e1e1e', lw=1.9, alpha=0.5, label=r'$k_s^1$')
        #ax.loglog(llin/d2pi, law2/tauA2pi, ls='-.', color='#1e1e1e', lw=1.9, alpha=0.5, label=r'$k_s^2$')
        #ax.loglog(np.linspace(10,50,50)/d2pi, 1e-6*np.linspace(10,50,50)**4/tauA2pi, ls='--', color='#1e1e1e', lw=1.9, alpha=0.5, label=r'$k_s^4$')
        #-- Revision 2 (non-teoretical scaling ks^2 confusing):
        ax.loglog(np.linspace(50,400,50)/d2pi, 2e-1*np.linspace(50,400,50)/tauA2pi, ls=':', color='#1e1e1e', lw=1.9, alpha=0.5, label=r'$k_s^1$')
        ax.loglog(np.linspace(10,60,50)/d2pi, 1e-6*np.linspace(10,60,50)**4/tauA2pi, ls='--', color='#1e1e1e', lw=1.9, alpha=0.5, label=r'$k_s^4$')
    #
    ax.loglog(ks/d2pi, wt/tauA2pi, 'b-o', lw=3.1, ms=13, alpha=0.9, label=r'S Cases')#r'Simple $B_0$')
    ax.loglog(ks[4]/d2pi, wt[4]/tauA2pi, color='b', marker='o', mfc='None', ms=16.2, ls='', alpha=0.9, label=r'SB-1 Case')#
    #ax.loglog(kshe/d2pi, wthe/tauA2pi, 'k->', lw=3.1, ms=13, alpha=0.8, label=r'stronger $B_0$ S Cases')#r'Simple stronger $B_0$')
    #ax.loglog(ksle/d2pi, wtle/tauA2pi, 'c-<', lw=3.1, ms=13, alpha=0.7, label=r'weaker $B_0$ S Cases')#r'Simple weaker $B_0$')
    ax.loglog(kspb/d2pi, wtpb/tauA2pi/1.34545, 'r-*', lw=3.1, ms=13, label=r'C Cases')#r'Complex $B_0$')
    ax.loglog(kspb[6]/d2pi, wtpb[6]/tauA2pi/1.34545, color='r', marker='s', mfc='None', ms=16, ls='', label=r'CB-1 Case')#
    ax.loglog(kspb[11]/d2pi, wtpb[11]/tauA2pi/1.34545, color='r', marker='d', mfc='None', ms=16, ls='', label=r'CB-2 Case')#
else:
    ax.loglog(np.linspace(60,400,50), 2e-1*np.linspace(60,400,50)/n2pi/le, ls=':', color='#1e1e1e', lw=1.9, alpha=0.5, label=r'$k_s^1$')
    ax.loglog(llin, law2/n2pi/le, ls='-.', color='#1e1e1e', lw=1.9, alpha=0.5, label=r'$k_s^2$')
    ax.loglog(np.linspace(10,50,50), 1e-6*np.linspace(10,50,50)**4/n2pi/le, ls='--', color='#1e1e1e', lw=1.9, alpha=0.5, label=r'$k_s^4$')
    #
    ax.loglog(ks, wt/n2pi/le, 'b-o', lw=3.1, ms=13, alpha=0.9, label=r'S Cases')#r'Simple $B_0$')
    ax.loglog(ks[4], wt[4]/n2pi/le, color='b', marker='o', mfc='None', ms=16, ls='', alpha=0.9, label=r'SB-1 Case')#
    #ax.loglog(kshe, wthe/n2pi/lehe, 'k->', lw=3.1, ms=13, alpha=0.8, label=r'stronger $B_0$ S Cases')#r'Simple stronger $B_0$')
    #ax.loglog(ksle, wtle/n2pi/lele, 'c-<', lw=3.1, ms=13, alpha=0.7, label=r'weaker $B_0$ S Cases')#r'Simple weaker $B_0$')
    ax.loglog(kspb, wtpb/n2pi/1.34545/lepb, 'r-*', lw=3.1, ms=13, label=r'C Cases')#r'Complex $B_0$')
    ax.loglog(kspb[6], wtpb[6]/n2pi/1.34545/lepb, color='r', marker='s', mfc='None', ms=16, ls='', label=r'CB-1 Case')#
    ax.loglog(kspb[11], wtpb[11]/n2pi/1.34545/lepb, color='r', marker='d', mfc='None', ms=16, ls='', label=r'CB-2 Case')#
if ( l_phydim ):
    plt.xlabel(r'Cylindrical radial linear wavenumber $\vert k_s \vert / (2\pi), (km^{-1})$', fontsize=36)
    plt.ylabel(r'Wave frequency $\vert \omega_o \vert / (2\pi), (y^{-1})$', fontsize=36)
else:
    plt.xlabel(r'Cylindrical radial wavenumber $k_s$', fontsize=36)
    plt.ylabel(r'Normalised Wave frequency $\omega_o / (2\pi\,\Omega)$', fontsize=36)
formatter = LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.5))
ax.get_xaxis().set_minor_formatter(formatter)
##ax.xaxis.set_minor_locator(LogFormatter())
##ax.xaxis.set_minor_formatter(FormatStrFormatter("%.0e"))
#yformatter = LogFormatter(labelOnlyBase=False, minor_thresholds=(10, 0.01))
#ax.get_yaxis().set_minor_formatter(yformatter)
ax.tick_params(which='both', width=1.5)
ax.tick_params(which='major', length=12)
ax.tick_params(which='minor', length=6)
plt.gca().xaxis.set_tick_params(which='major', labelsize=32)
plt.gca().xaxis.set_tick_params(which='minor', labelsize=26)
plt.gca().yaxis.set_tick_params(which='major', labelsize=32)
plt.gca().yaxis.set_tick_params(which='minor', labelsize=26)
xlevels=(2e-3, 6e-3, 1e-2)
plt.xticks(xlevels)
ylevels=(1e-2, 2e-2, 6e-2, 1e-1, 2e-1, 6e-1, 1, 2)
plt.yticks(ylevels)
if ( l_phydim ):
    plt.xlim(18/d2pi,265/d2pi)
    plt.ylim(100*le/tauA2pi,49000*le/tauA2pi)
else:
    plt.xlim(18,265)
    plt.ylim(98/n2pi,49000/n2pi)
plt.grid(ls=':')#which='both',
plt.legend(fontsize=26)#32)#
plt.tight_layout()
if ( l_save ):
    #plt.savefig(savePlot+'Scaling_Dispersion-Relation_QG-MC-All_phydim_ther.pdf')
    plt.savefig('/home/ovrbr/Olivier/Post-Doc_2025-2027/'+'Scaling_Dispersion-Relation_QG-MC-All_phydim_ther.pdf')
    plt.close('all')
else:
    plt.show()

#-- End Script
