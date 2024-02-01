"""
Created on Mon Feb 09 2023
@author: obarrois

Python version of PARODY-JA4.56-Base/Matlab Matlab file 'makebasefield.m'.
Compute a background magneitic field B0 and its curl j0 using spherical bessel
functions of the first kind.
"""

#import os
#import sys
import numpy as np
from parobpy.load_parody import parodyload, load_basefield
from parobpy.parob_lib import rderavg, curl_sph#, get_curl#
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import cmocean.cm as cmo
import shtns

#matplotlib.use('Agg')  # backend for no display if needed

#-- Functions
def get_lm_map(l_max, m_max):
	"""
	This routine determines the look-up tables to convert the indices
	(l, m) to the single index lm.
	WARNING:: The limit of numpy.int16 = 2**16 = 65536 which can be lower than
	lm_max for high l_max truncations! --> use numpy.int32 instead

	Inputs::
		-- l_max: maximum degree of the Spherical Harmonic (SH) grid
		type: numpy.integer
		-- m_max: maximum order of the SH grid
		type: numpy.integer
	Outputs::
		-- lm_map: look-up tables to convert (l, m) into lm
		type: numpy.ndarray; numpy.int32
		size: l_max+1 x m_max+1
	Example::
		>>> llmm2lm = get_lm_map(8,8)
	"""
	lm_map = np.zeros((l_max+1,m_max+1), dtype=np.int32)
	lm=0
	for m in range(0,m_max+1):
		for l in range(m,l_max+1):
			lm=lm+1
			lm_map[l,m]=lm #-- llmm2lm[l+1-1,m+1-1]=
	#--
	return lm_map

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

#----------------------------------------------------------------------------%%
#-- INPUT PARAMETERS

l_resym = True # re-symmetrizing the data if minc !=1
l_curl_b0 = True # computing curl of Background field B0

l_check_all = True # display all check plot figures
l_save = 0 # save figures?
l_binary = True # in a binary or in a hd5f format?
saveDir = '/Users/obarrois/Desktop/Base_Fields/1e7/' # path to save files
savePlot = '/Users/obarrois/Desktop/Parodys/Outputs/' # path to save main figures

#------------------------------------------------------------------------------
#%% Load data

run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '1e7', 'Pm1o44e-1', 1.1e-3, 1.0, 'b4', '0.000040320' # 1e-7
#run_Ek, run_Pm, Lehnert, amp_B, run_ID, timestamp = '3e10', '', 2.6e-4, 1.0, 'b5', '0.000001456' # 3e-10

#directory = '/gpfs/users/obarrois/Work/Waves1e7/' # if you use python directly from cluster
#directory = '/Users/obarrois/Desktop/Base_Fields/1e7/' # if you want to use basefield from the computer
directory = '/Users/obarrois/Desktop/stella/Work/Waves'+run_Ek+'/'+run_Pm+'/' # if you mount stella on own desk

Gt_file = 'Gt={}.{}'.format(timestamp,run_ID)
filename = directory + Gt_file

(_, _, _, _, _, _, _, _, _, _, _, Ek, Ra, Pm, Pr,
        nr, ntheta, nphi, minc, radius, theta, phi,
        _, _, _, _, _, _, _) = parodyload(filename)

#-- for double check
phir = phi.copy()
nphir = nphi
if ( l_check_all ):
    basename ='/Users/obarrois/Desktop/Base_Fields/1e7/basefield.mat'
    B0rR, B0tR, B0pR, _, _, _ = load_basefield(basename,nr,ntheta,nphi)
    if ( l_resym and minc !=1 ):
        from parobpy.parob_lib import symmetrize
        nphir = nphi*minc##+1 NOTE: last point is spurious in MagIC.symmetrize
        phir = np.linspace(phi[0],3*phi[-1], nphir)
        B0rR = symmetrize(B0rR,ms=minc)
        B0tR = symmetrize(B0tR,ms=minc)
        B0pR = symmetrize(B0pR,ms=minc)

#-- Parameters
nrtot=449
ng=50
#ntheta=200
#nphi=nphir*minc##+1 #+1--> 145 --> removed last point in symmetrize

rcmb = radius[nr-1]#radius[0]#
#phi = np.linspace(phir[0],3*phir[-1], nphi) #phi = np.linspace(phir[0],2*np.pi-phir[0], nphir)
sint = np.sin(theta)
cost = np.cos(theta)

# NOTE: WARNING!!: l_max/m_max values are somewhat hardcoded (probably with the end of path in mind)
#nphir=288
l_max = 133# nphir//3
m_max = 48#l_max#
lmmax=9045 #??
sh = shtns.sht(l_max, m_max)
nlat, nlon = sh.set_grid(nphi=nphir, nlat=ntheta)
lmmax = sh.nlm #--> 1225 with l_max = nphi/3 = 48

llmm2lm = get_lm_map(l_max, m_max)

#-- Define Bpol/tor
bpol = np.zeros((lmmax,nr), dtype=np.complex128)
btor = np.zeros_like(bpol)

#------------------------------------------------------------------------------
#%% Compute base field in spectral space

#-- define Spherical bessel functions
from scipy.special import jv as besselj1

def jnu(x, nu): #-- has to give arguments after grid for root_scalar purposes
    """
    Evaluates the spherical bessel function of the first kind of real order nu.
    NOTE: This is equivalent to scipy.special.spherical_jn

    Inputs::
        -- x: grid points where the function will be evaluated
        type: np.array; np.float
        -- nu: order of the bessel function
        type: np.int
    Outputs::
        -- jnu: spherical bessel function of the first kind of order nu
    Example::
        >>> grid = np.linspace(0,10,200); order = 1
        >>> j1 = jnu(grid, order)
    """
    #return scipy.special.spherical_jn(nu, x)
    return np.sqrt(np.pi/(2.*x)) * besselj1(0.5+nu, x)

#-- extract roots of the Spherical bessel functions
from scipy.optimize import root_scalar

#beta11=fzero(j0,[3 4])/r(nr);
sol1 = root_scalar(jnu, args=(0), method = 'toms748', bracket = [3, 4])
beta11 = sol1.root/rcmb
#beta12=fzero(j0,[5 7])/r(nr);
sol2 = root_scalar(jnu, args=(0), method = 'toms748', bracket = [5, 7])
beta12 = sol2.root/rcmb
#beta31=fzero(j2,[5 6])/r(nr);
sol3 = root_scalar(jnu, args=(2), method = 'toms748', bracket = [5, 6])
beta31 = sol3.root/rcmb
# NOTE: Get the same values from matlab!

#-- build poloidal field, so that: B0 = VxVx(P,r) with P = following...
lm10 = llmm2lm[1,0] #llmm2lm(1+1,0+1)
bpol[lm10-1, :] = jnu(beta11*radius, 1) - 0.3*jnu(beta12*radius, 1) #yBpr(lm,:)=j1(beta11*r)-0.3*j1(beta12*r);

lm30 = llmm2lm[3,0] #llmm2lm(3+1,0+1)
bpol[lm30-1, :] = - 0.2*jnu(beta31*radius, 3) #yBpr(lm,:)=-0.2*j3(beta31*r);

lm33 = llmm2lm[3,3] #llmm2lm(3+1,3+1)
bpol[lm33-1, :] = -0.3*jnu(beta31*radius, 3) #yBpr(lm,:)=0.3*j3(beta31*r);

#------------------------------------------------------------------------------
#%% Bring base field in physical space

#-- prepare spec bpol/btor to spat B0, j0
dbpol = np.zeros_like(bpol)
ddbpol = np.zeros_like(bpol)
dbtor = np.zeros_like(btor)
for lm in range(lmmax):
    dbpol[lm, :] = rderavg(bpol[lm, :], rad=radius, exclude=False)
    ddbpol[lm, :] = rderavg(dbpol[lm, :], rad=radius, exclude=False)
#-- because toroidal field = zeros, no need to compute dbtor

# Renormalisation? --> Seem to be needed
for i in range(nr):
    bpol[1:, i] = bpol[1:, i]*(sh.l[1:]*(sh.l[1:]+1))/radius[i]**2

#-- bringing Bpol/tor into physical space: TP_spec_spat_B_CE
B0r = np.zeros((nphir,ntheta,nr),)
B0t = np.zeros_like(B0r)
B0p = np.zeros_like(B0r)
for i in range(nr):
    #br, bt, bp = sh.synth(bpol[:, i], dbpol[:, i], btor[:, i]) # NOTE: theta will be given as first dimension
    br = sh.synth(bpol[:, i])
    bt, bp = sh.synth(dbpol[:, i], btor[:, i])/radius[i] #-- to get the right units
    B0r[:,:, i] = br.T; B0t[:,:, i] = bt.T; B0p[:,:, i] = bp.T

#-- bringing Bpol/tor into curled physical space: TP_spec_spat_rotB_CE
j0r = np.zeros((nphir,ntheta,nr),)
j0t = np.zeros_like(B0r)
j0p = np.zeros_like(B0r)
if ( l_curl_b0 ):
    #-- grid mesh helpers
    grid = [radius, theta, phir]
    r3D = np.zeros((nphir, ntheta, nr),)
    for i in range(nr):
        r3D[:,:,i] = radius[i]
    #
    th3D = np.zeros_like(r3D)
    for j in range(ntheta):
        th3D[:,j,:] = theta[j]
    #
    sint3D = np.sin(th3D)
    cost3D = np.cos(th3D)
    s3D = r3D*sint3D
    grid_help = [r3D, sint3D, s3D]
    #j0r, j0t, j0p = get_curl(grid, B0r, B0t, B0p, l_spectral=True, sh_struct=sh)
    j0r, j0t, j0p = curl_sph(grid_help, B0r, B0t, B0p, l_mpi=False, l_spectral=True, sh_struct=sh)
    if ( np.sum(np.isnan(j0r)) + np.sum(np.isnan(j0t)) + np.sum(np.isnan(j0p)) != 0 ): print('Warning!:: nan in curl B0 computations!') # to check if there are any problems
else:
    for i in range(nr):
        jr, jt, jp = sh.synth(bpol[:, i], ddbpol[:, i], btor[:, i], dbtor[:, i]) # NOTE: theta will be given as first dimension
        j0r[:,:, i] = jr.T; j0t[:,:, i] = jt.T; j0p[:,:, i] = jp.T

#-- Base field as symmetry of minc=3
Brbase = B0r[:48,:,:]
Btbase = B0t[:48,:,:]
Bpbase = B0p[:48,:,:]
Jrbase = j0r[:48,:,:]
Jtbase = j0t[:48,:,:]
Jpbase = j0p[:48,:,:]

#------------------------------------------------------------------------------
#%% Plot and save

#-- Plot base for paper

#-- Equat B0r minc
eps = 1.e-3
rr, pphi = np.meshgrid(radius, phi)
xx = rr*np.cos(pphi)
yy = rr*np.sin(pphi)
#-- Adjusting colorsfor trajs #CB Barrois and Aubert 2024
Atrajcolor=WA_BRoc_Dark#'#32CD32' # Lime # '#99CC32' # Jaune-Vert # '#FF7F00' # Orange
Ftrajcolor= Personal_Cyan#'#FF7F00' # Orange # '#32CD32' # Lime # '#99CC32' # Jaune-Vert # 
Strajcolor= 'blue'#FF7F00' # Orange # '#32CD32' # Lime # '#99CC32' # Jaune-Vert # 

fieldplot = B0r[:,ntheta//2,:]
#fig = plt.figure(figsize=(26, 3.6))
fig = plt.figure(figsize=(11, 6.9))
cmax = 1.25#abs(fieldplot).max()*0.95
llevels=np.linspace(-cmax,cmax,64)
clevels=np.linspace(-cmax,cmax,5)
ax = plt.subplot(111)
cf = ax.contourf(xx,yy,fieldplot[:nphi,:],levels=llevels,extend='both',cmap='PuOr_r')
#cf = ax.pcolormesh(xx,yy,fieldplot[:nphi,:],vmin=-cmax,vmax=cmax,antialiased=True,shading='gouraud',rasterized=True,cmap='PuOr_r')
ax.plot(radius[0]*np.cos(phi), radius[0]*np.sin(phi), 'k-', lw=1.9)
ax.plot(radius[-1]*np.cos(phi), radius[-1]*np.sin(phi), 'k-', lw=1.9)
xa = radius[-1]*np.cos(phi[0])
ya = radius[-1]*np.sin(phi[0])
xb = radius[0]*np.cos(phi[0])
x0 = np.linspace(xa, xb, 32)
y0 = np.tan(phi[0])*(x0-xa)+ya
ax.plot(x0, y0, color=Ftrajcolor, ls='-', lw=6.4)#, alpha=0.8)
xc = radius[-1]*np.cos(phi[-1])
yc = radius[-1]*np.sin(phi[-1])
xd = radius[0]*np.cos(phi[-1])
x1 = np.linspace(xc, xd, 32)
y1 = np.tan(phi[-1])*(x1-xc)+yc
ax.plot(x1, y1, color=Ftrajcolor, ls='-', lw=6.4)#, alpha=0.8)
xe = radius[-1]*np.cos(phi[24])
ye = radius[-1]*np.sin(phi[24])
xf = radius[0]*np.cos(phi[24])
x2 = np.linspace(xe, xf, 32)
y2 = np.tan(phi[24])*(x2-xe)+ye
ax.plot(x2, y2, color=Ftrajcolor, ls='-', lw=6.4)#, alpha=0.8)
xg = radius[-1]*np.cos(phi[12])
yg = radius[-1]*np.sin(phi[12])
xh = radius[0]*np.cos(phi[12])
x3 = np.linspace(xg, xh, 32)
y3 = np.tan(phi[12])*(x3-xg)+yg
#ax.plot(x3, y3, color=Strajcolor, ls='-', lw=2.6, alpha=0.8)
xi = radius[-1]*np.cos(phi[36])
yi = radius[-1]*np.sin(phi[36])
xj = radius[0]*np.cos(phi[36])
x4 = np.linspace(xi, xj, 32)
y4 = np.tan(phi[36])*(x4-xi)+yi
#ax.plot(x4, y4, color=Strajcolor, ls='-', lw=2.6)
xk = radius[-1]*np.cos(phi[3])
yk = radius[-1]*np.sin(phi[3])
xl = radius[0]*np.cos(phi[3])
x5 = np.linspace(xk, xl, 32)
y5 = np.tan(phi[3])*(x5-xk)+yk
#ax.plot(x5, y5, color=Atrajcolor, ls='-', lw=3.2)
#ax.plot(x0, y0, 'k-', lw=1.5)
#ax.plot(x1, y1, 'k-', lw=1.5)
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
cb = fig.colorbar(cf, ax=ax, fraction=0.05, pad=0.035, orientation='vertical', ticks=clevels, format=r'${x:.1f}$')
cb.ax.tick_params(labelsize=32)
transAx = mtransforms.ScaledTranslation(8.35+10/72, -45/72, fig.dpi_scale_trans)
ax.text(0.0, 1.0, PLabels[3], transform=ax.transAxes + transAx,
        fontsize=36, va='bottom', fontfamily='serif')
plt.tight_layout()
#if ( not l_save ):  plt.show()

#-- Surf B0r de-minc
phip = np.linspace(-np.pi, np.pi, nphir)
thetap = np.linspace(np.pi/2, -np.pi/2, ntheta)
pphi, ttheta = np.mgrid[-np.pi:np.pi:nphir*1j, np.pi/2.:-np.pi/2.:ntheta*1j]
lon2 = pphi*180./np.pi
lat2 = ttheta*180./np.pi

circles = np.r_[-60., -30., 0., 30., 60.]
delon = 60.
meridians = np.arange(-180+delon, 180, delon)

from vizuals import hammer2cart
xx, yy = hammer2cart(ttheta, pphi)

fieldplot = B0r[:,:,-2]
#fig = plt.figure(figsize=(26, 3.6))
fig = plt.figure(figsize=(13.5, 6.9))
#cmax = 1.25#abs(fieldplot[n_t,:,:]).max()*0.92
llevels=np.linspace(-cmax,cmax,64)
clevels=np.linspace(-cmax,cmax,5)
ax = plt.subplot(111)
cf = ax.contourf(xx,yy,fieldplot,levels=llevels,extend='both',cmap='PuOr_r')#'seismic')#cmo.balance)#
#cf = radialContour(fieldplot,rad=1,cm='PuOr_r')#levels=llevels,extend='both',cmap='seismic')#'PuOr_r')#cmo.balance)#
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
ax.set_xlim(xx.min()*1.01, 1.01*xx.max())
ax.set_ylim(yy.min()*1.01, 1.01*yy.max())
ax.axis('off')
cb = fig.colorbar(cf, ax=ax, orientation='vertical', fraction=0.05, pad=0.022, ticks=clevels, format=r'${x:.1f}$')#1e}')
cb.ax.tick_params(labelsize=32)
transAx = mtransforms.ScaledTranslation(10.61+10/72, -45/72, fig.dpi_scale_trans)
ax.text(0.0, 1.0, PLabels[0], transform=ax.transAxes + transAx,
        fontsize=36, va='bottom', fontfamily='serif')
plt.tight_layout()
#plt.show()
if ( not l_save ):  plt.show()

#-- Save B0 frames
if ( l_save ):
    plt.figure(1)
    plt.savefig(savePlot+'Breq_deminc_VaLong.png')
    plt.figure(2)
    plt.savefig(savePlot+'Brsurf_SameCB.png')
    plt.close('all')

#-- Save using either a binary or a hd5f format (the latter is closer to what makebasefield.m produces)
if ( l_save ):
    if ( l_binary ): #-- save a binary file using native python open and tofile routines
        f = open('{}/basefield.bi'.format(saveDir), 'wb')
        B0r.tofile(f)
        B0t.tofile(f)
        B0p.tofile(f)
        j0r.tofile(f)
        j0t.tofile(f)
        j0p.tofile(f)
        f.close()
    else: #-- save as a matlab file, using h5py
        #save -v7.3 basefield.mat Brbase Btbase Bpbase Jrbase Jtbase Jpbase
        import h5py
        with h5py.File('{}/basefield.hdf5'.format(saveDir), 'a') as f:
            f.create_dataset('Brbase', data=B0r)
            f.create_dataset('Btbase', data=B0t)
            f.create_dataset('Bpbase', data=B0p)
            f.create_dataset('Jrbase', data=j0r)
            f.create_dataset('Jtbase', data=j0t)
            f.create_dataset('Jpbase', data=j0p)
        
    #np.tofile(B0r)

#-- End Script
