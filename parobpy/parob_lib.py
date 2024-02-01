# -*- coding: utf-8 -*-
#-- extracted from libmagic and libpizza
"""
Created on Fri Feb 03 2023
@author: obarrois

useful tools from pizza and magic python packages
to compute various derivatives and reconstruct fields.
"""

import scipy.interpolate as scint
import numpy as np
import glob, os, re, sys
from scipy.fftpack import dct
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d


def cc2real(f):
    """
    Translate a list of complex into a list of reals
    """
    return 2.*np.sum(abs(f[1:, :])**2, axis=0) + f[0, :].real*f[0, :].real


def scanDir(pattern, tfix=None):
    """
    This function sorts the files which match a given input pattern from the
    oldest to the most recent one (in the current working directory)

    >>> dat = scanDir('log.*')
    >>> print(log)

    :param pattern: a classical regexp pattern
    :type pattern: str
    :param tfix: in case you want to add only the files that are more recent
                 than   a certain date, use tfix (computer 1970 format!!)
    :type tfix: float
    :returns: a list of files that match the input pattern
    :rtype: list
    """
    dat = [(os.stat(i).st_mtime, i) for i in glob.glob(pattern)]
    dat.sort()
    if tfix is not None:
        out = []
        for i in dat:
            if i[0] > tfix:
                out.append(i[1])
    else:
        out = [i[1] for i in dat]
        
    return out


def symmetrize(data, ms, reversed=False):
    """
    Imported from pizza/MagIC python package:
    Symmetrise an array which is defined only with an azimuthal symmetry
    minc=ms
     --> modified by @obarrois to remove the last point that is spurious

    :param data: the input array
    :type data: numpy.ndarray
    :param ms: the azimuthal symmetry
    :type ms: int
    :param reversed: set to True, in case the array is reversed (i.e. n_phi
                     is the last column)
    :type reversed: bool
    :returns: an output array of dimension (data.shape[0]*ms) ##+1)
    :rtype: numpy.ndarray
    """
    if reversed:
        nphi = data.shape[-1]*ms+1
        size = [nphi]
        size.insert(0, data.shape[-2])
        if len(data.shape) == 3:
            size.insert(0, data.shape[-3])
        out = np.zeros(size, dtype=data.dtype)
        for i in range(ms):
            out[..., i*data.shape[-1]:(i+1)*data.shape[-1]] = data
        out[..., -1] = out[..., 0]
    else:
        nphi = data.shape[0]*ms + 1
        size = [nphi]
        if len(data.shape) >= 2:
            size.append(data.shape[1])
        if len(data.shape) == 3:
            size.append(data.shape[2])
        out = np.zeros(size, dtype=data.dtype)
        for i in range(ms):
            out[i*data.shape[0]:(i+1)*data.shape[0], ...] = data
        out[-1, ...] = out[0, ...]

    return out[:-1, ...]


def avgField(time, field, tstart=None, tstop=None, std=False):
    """
    This subroutine computes the time-average (and the std) of a time series

    :param time: time
    :type time: numpy.ndarray
    :param field: the time series of a given field
    :type field: numpy.ndarray
    :param tstart: the starting time of the averaging
    :type tstart: float
    :param tstart: the stopping time of the averaging
    :type tstart: float
    :param std: when set to True, the standard deviation is also calculated
    :type std: bool
    :returns: the time-averaged quantity
    :rtype: float
    """
    if tstart is not None:
        mask = np.where(abs(time-tstart) == min(abs(time-tstart)), 1, 0)
        ind = np.nonzero(mask)[0][0]
    else:  # the whole input array is taken!
        ind = 0
    if tstop is not None:
        mask = np.where(abs(time-tstop) == min(abs(time-tstop)), 1, 0)
        ind1 = np.nonzero(mask)[0][0]+1
    else:  # the whole input array is taken!
        ind1 = len(time)
    fac = 1./(time[ind1-1]-time[ind])
    avgField = fac*np.trapz(field[ind:ind1], time[ind:ind1])

    if std:
        stdField = np.sqrt(fac*np.trapz((field[ind:ind1]-avgField)**2,
                           time[ind:ind1]))
        return avgField, stdField
    else:
        return avgField


def chebgrid(nr, a, b):
    """
    This function defines a Gauss-Lobatto grid from a to b.

    >>> r_icb = 0.5 ; r_cmb = 1.5; n_r_max=65
    >>> rr = chebgrid(n_r_max, r_icb, r_cmb)

    :param nr: number of radial grid points
    :type nr: int
    :param a: lower limit of the Gauss-Lobatto grid
    :type a: float
    :param b: upper limit of the Gauss-Lobatto grid
    :type b: float
    :returns: the Gauss-Lobatto grid
    :rtype: numpy.ndarray
    """
    rst = (a+b)/(b-a)
    rr = 0.5*(rst+np.cos(np.pi*(1.-np.arange(nr+1.)/nr)))*(b-a)

    return rr


def costf(f, fac=True):
    """
    This routine transform an input array from real to Chebyshev space

    :param f: the input array
    :type f: numpy.ndarray
    :param fac: normalisation factor is used
    :type f: bool
    :returns: a transformed array
    :rtype: numpy.ndarray
    """
    # nr = f.shape[-1]
    if fac:
        norm = np.sqrt(0.5/(f.shape[-1]-1))
    else:
        norm = 1.
    # fbig = np.hstack((f[..., :], f[..., -2:0:-1]))
    # fbig = fbig.astype('complex256')
    # fhat = norm*np.fft.fft(fbig, axis=-1)[..., :nr]

    fhat = norm*dct(f, type=1, axis=-1)

    return fhat


def get_dr(f):
    """
    This routine calculates the first radial derivative of a input array using
    Chebyshev recurrence relation.

    :param f: the input array
    :type f: numpy.ndarray
    :returns: the radial derivative of f
    :rtype: numpy.ndarray
    """
    Nr = f.shape[-1]
    fhat = costf(f)

    # eps = np.finfo(1.0e0).eps
    # valmin = 500. * eps*abs(fhat).max()

    df = np.zeros_like(fhat)
    df[..., -1] = 0.
    df[..., -2] = (Nr-1)*fhat[..., -1]

    for i in range(Nr-3, -1, -1):
        df[..., i] = df[..., i+2]+2.*(i+1)*fhat[..., i+1]

    df[..., :] = 2.*df[..., :]

    df = costf(df)

    return df


def intcheb(f):
    """
    This routine computes an integration of a function along radius using
    Chebyshev recurrence relation.

    :param f: the input array
    :type f: numpy.ndarray
    :returns: the integral of f between z1 and z2
    :rtype: float
    """
    nr = f.shape[-1]-1
    w2 = costf(f)
    w2[..., 0] *= 0.5
    w2[..., -1] *= 0.5

    if len(f.shape) == 1:
        int = 0.
        for i in range(0, nr+1, 2):
            int = int-1./(i**2-1)*w2[i]
    elif len(f.shape) == 2:
        int = np.zeros(f.shape[0], dtype=f.dtype)
        for m in range(f.shape[0]):
            int[m] = 0.
            for i in range(0, nr+1, 2):
                int[m] = int[m]-1./(i**2-1)*w2[m, i]

    # Be careful if a mapping is used this would be modified
    int *= np.sqrt(2./(f.shape[-1]-1))

    return int


def spat_spec(arr_grid, n_m_max):
    """
    This routine computes a spectral transform from a spatial represenation
    to a spectral representation.

    :param f: the input array
    :type f: numpy.ndarray
    :param n_m_max: the number of modes
    :type n_m_max: int
    :returns: an array in the spectral space
    :rtype: numpy.ndarray
    """
    n_phi = arr_grid.shape[0]

    return np.fft.fft(arr_grid, axis=0)[:n_m_max]/n_phi


def spec_spat(arr_M, n_phi_max):
    """
    This routine computes a spectral transform from a spectral represenation
    to a spatial representation.

    :param f: the input array
    :type f: numpy.ndarray
    :param n_phi_max: the number of azimuthal grid points
    :type n_phi_max: int
    :returns: an array in the physical space
    :rtype: numpy.ndarray
    """
    n_m = arr_M.shape[0]
    tmp = np.zeros((int(n_phi_max/2)+1, arr_M.shape[-1]), 'complex64')
    tmp[:n_m, :] = arr_M

    return np.fft.irfft(tmp, n=n_phi_max, axis=0)*n_phi_max


def phideravg(data, minc=1, order=4):
    """
    phi-derivative of an input array

    >>> gr = MagicGraph()
    >>> dvphidp = phideravg(gr.vphi, minc=gr.minc)

    :param data: input array
    :type data: numpy.ndarray
    :param minc: azimuthal symmetry
    :type minc: int
    :param order: order of the finite-difference scheme (possible values are 2 or 4)
    :type order: int
    :returns: the phi-derivative of the input array
    :rtype: numpy.ndarray
    """
    nphi = data.shape[0]
    dphi = 2.*np.pi/minc/(nphi-1.)
    if order == 2:
        der = (np.roll(data, -1,  axis=0)-np.roll(data, 1, axis=0))/(2.*dphi)
        der[0, ...] = (data[1, ...]-data[-2, ...])/(2.*dphi)
        der[-1, ...] = der[0, ...]
    elif order == 4:
        der = (   -np.roll(data,-2,axis=0) \
               +8.*np.roll(data,-1,axis=0) \
               -8.*np.roll(data, 1,axis=0)  \
                  +np.roll(data, 2,axis=0)   )/(12.*dphi)
        der[1, ...] = (-data[3, ...]+8.*data[2, ...]-\
                       8.*data[0, ...] +data[-2, ...])/(12.*dphi)
        der[-2, ...] = (-data[0, ...]+8.*data[-1, ...]-\
                       8.*data[-3, ...]+data[-4, ...])/(12.*dphi)
        der[0, ...] = (-data[2, ...]+8.*data[1, ...]-\
                       8.*data[-2, ...] +data[-3, ...])/(12.*dphi)
        der[-1, ...] = der[0, ...]

    return der


def rderavg(data, rad, exclude=False):
    """
    Radial derivative of an input array

    >>> gr = MagiGraph()
    >>> dvrdr = rderavg(gr.vr, gr.radius)

    :param data: input array
    :type data: numpy.ndarray
    :param rad: radial grid
    :type rad: numpy.ndarray
    :param exclude: when set to True, exclude the first and last radial grid points
                    and replace them by a spline extrapolation (default is False)
    :type exclude: bool
    :returns: the radial derivative of the input array
    :rtype: numpy.ndarray
    """
    r1 = rad[0]
    r2 = rad[-1]
    nr = data.shape[-1]
    grid = chebgrid(nr-1, r1, r2)
    tol = 1e-6 # This is to determine whether Cheb der will be used
    diff = abs(grid-rad).max()
    if diff > tol:
        spectral = False
        grid = rad
    else:
        spectral = True

    if exclude:
        g = grid[::-1]
        gnew = np.linspace(r2, r1, 1000)
        if len(data.shape) == 2:
            for i in range(data.shape[0]):
                val = data[i, ::-1]
                tckp = scint.splrep(g[1:-1], val[1:-1])
                fnew = scint.splev(gnew, tckp)
                data[i, 0] = fnew[-1]
                data[i, -1] = fnew[0]
        else:
            for j in range(data.shape[0]):
                for i in range(data.shape[1]):
                    val = data[j, i, ::-1]
                    tckp = scint.splrep(g[1:-1], val[1:-1])
                    fnew = scint.splev(gnew, tckp)
                    data[j, i, 0] = fnew[-1]
                    data[j, i, -1] = fnew[0]
    if spectral:
        d1 = matder(nr-1, r1, r2)
        if len(data.shape) == 2:
            der = np.tensordot(data, d1, axes=[1, 1])
        else:
            der = np.tensordot(data, d1, axes=[2, 1])
    else:
        denom = np.roll(grid, -1) - np.roll(grid, 1)
        denom[0] = grid[1]-grid[0]
        denom[-1] = grid[-1]-grid[-2]
        der = (np.roll(data, -1,  axis=-1)-np.roll(data, 1, axis=-1))/denom
        der[..., 0] = (data[..., 1]-data[..., 0])/(grid[1]-grid[0])
        der[..., -1] = (data[..., -1]-data[..., -2])/(grid[-1]-grid[-2])

    return der


def thetaderavg(data, order=4):
    """
    Theta-derivative of an input array (finite differences)

    >>> gr = MagiGraph()
    >>> dvtdt = thetaderavg(gr.vtheta)

    :param data: input array
    :type data: numpy.ndarray
    :param order: order of the finite-difference scheme (possible values are 2 or 4)
    :type order: int
    :returns: the theta-derivative of the input array
    :rtype: numpy.ndarray
    """
    if len(data.shape) == 3: # 3-D
        ntheta = data.shape[1]
        dtheta = np.pi/(ntheta-1.)
        if order == 2:
            der = (np.roll(data, -1,  axis=1)-np.roll(data, 1, axis=1))/(2.*dtheta)
            der[:, 0, :] = (data[:, 1, :]-data[:, 0, :])/dtheta
            der[:, -1, :] = (data[:, -1, :]-data[:, -2, :])/dtheta
        elif order == 4:
            der = (   -np.roll(data,-2,axis=1) \
                   +8.*np.roll(data,-1,axis=1) \
                   -8.*np.roll(data, 1,axis=1)  \
                      +np.roll(data, 2,axis=1)   )/(12.*dtheta)
            der[:, 1, :] = (data[:, 2, :]-data[:, 0, :])/(2.*dtheta)
            der[:, -2, :] = (data[:, -1, :]-data[:, -3, :])/(2.*dtheta)
            der[:, 0, :] = (data[:, 1, :]-data[:, 0, :])/dtheta
            der[:, -1, :] = (data[:, -1, :]-data[:, -2, :])/dtheta

    elif len(data.shape) == 2: #2-D
        ntheta = data.shape[0]
        dtheta = np.pi/(ntheta-1.)
        if order == 2:
            der = (np.roll(data, -1,  axis=0)-np.roll(data, 1, axis=0))/(2.*dtheta)
            der[0, :] = (data[1, :]-data[0, :])/dtheta
            der[-1, :] = (data[-1, :]-data[-2, :])/dtheta
        elif order == 4:
            der = (-np.roll(data,-2,axis=0)+8.*np.roll(data,-1,axis=0)-\
                  8.*np.roll(data,1,axis=0)+np.roll(data,2,axis=0))/(12.*dtheta)
            der[1, :] = (data[2, :]-data[0, :])/(2.*dtheta)
            der[-2, :] = (data[-1, :]-data[-3, :])/(2.*dtheta)
            der[0, :] = (data[1, :]-data[0, :])/dtheta
            der[-1, :] = (data[-1, :]-data[-2, :])/dtheta

    return der


def zderavg(data, rad, colat=None, exclude=False):
    """
    z derivative of an input array

    >>> gr = MagiGraph()
    >>> dvrdz = zderavg(gr.vr, eta=gr.radratio, colat=gr.colatitude)

    :param data: input array
    :type data: numpy.ndarray
    :param rad: radial grid
    :type rad: numpy.ndarray
    :param exclude: when set to True, exclude the first and last radial grid points
                    and replace them by a spline extrapolation (default is False)
    :type exclude: bool
    :param colat: colatitudes (when not specified a regular grid is assumed)
    :type colat: numpy.ndarray
    :returns: the z derivative of the input array
    :rtype: numpy.ndarray
    """
    if len(data.shape) == 3:  # 3-D
        ntheta = data.shape[1]
    elif len(data.shape) == 2:  # 2-D
        ntheta = data.shape[0]
    nr = data.shape[-1]
    if colat is not None:
        th = colat
    else:
        th = np.linspace(0., np.pi, ntheta)

    if len(data.shape) == 3:  # 3-D
        thmD = np.zeros_like(data)
        for i in range(ntheta):
            thmD[:,i,:] = th[i]
    elif len(data.shape) == 2:  # 2-D
        thmD = np.zeros((ntheta, nr), np.float64)
        for i in range(ntheta):
            thmD[i, :] = th[i]

    dtheta = thetaderavg(data)
    dr = rderavg(data, rad, exclude)
    dz = np.cos(thmD)*dr - np.sin(thmD)/rad*dtheta

    return dz


def sderavg(data, rad, colat=None, exclude=False):
    """
    s derivative of an input array

    >>> gr = MagiGraph()
    >>> dvpds = sderavg(gr.vphi, eta=gr.radratio, colat=gr.colatitude)

    :param data: input array
    :type data: numpy.ndarray
    :param rad: radial grid
    :type rad: numpy.ndarray
    :param exclude: when set to True, exclude the first and last radial grid points
                    and replace them by a spline extrapolation (default is False)
    :type exclude: bool
    :param colat: colatitudes (when not specified a regular grid is assumed)
    :type colat: numpy.ndarray
    :returns: the s derivative of the input array
    :rtype: numpy.ndarray
    """
    ntheta = data.shape[0]
    nr = data.shape[-1]
    if colat is not None:
        th = colat
    else:
        th = np.linspace(0., np.pi, ntheta)
    rr2D, th2D = np.meshgrid(rad, th)
    dtheta = thetaderavg(data)
    dr = rderavg(data, rad, exclude)
    ds = np.sin(th2D)*dr + np.cos(th2D)/rr2D*dtheta

    return ds


def matder(nr, z1, z2):
    """
    This function calculates the derivative in Chebyshev space.

    >>> r_icb = 0.5 ; r_cmb = 1.5; n_r_max=65
    >>> d1 = matder(n_r_max, r_icb, r_cmb)
    >>> # Chebyshev grid and data
    >>> rr = chebgrid(n_r_max, r_icb, r_cmb)
    >>> f = sin(rr)
    >>> # Radial derivative
    >>> df = dot(d1, f)

    :param nr: number of radial grid points
    :type nr: int
    :param z1: lower limit of the Gauss-Lobatto grid
    :type z1: float
    :param z2: upper limit of the Gauss-Lobatto grid
    :type z2: float
    :returns: a matrix of dimension (nr,nr) to calculate the derivatives
    :rtype: numpy.ndarray
    """
    nrp = nr+1
    w1 = np.zeros((nrp, nrp), dtype=np.float64)
    zl = z2-z1
    for i in range(nrp):
        for j in range(nrp):
            w1[i, j] = spdel(i, j, nr, zl)

    return w1


def spdel(kr, jr, nr, zl):
    if kr != nr :
        fac = 1.
        k = kr
        j = jr
    else:
        fac = -1.
        k = 0.
        j = nr-jr

    spdel = fac*dnum(k, j, nr)/den(k, j, nr)
    return -spdel*(2./zl)


def dnum(k, j, nr):
    if k == 0:
        if (j == 0 or j == nr):
            dnum = 0.5
            a = nr % 2
            if a == 1:
                dnum = -dnum
            if j == 0:
                dnum = 1./3.*float(nr*nr)+1./6.
            return dnum

        dnum = 0.5*(float(nr)+0.5)*((float(nr)+0.5)+(1./np.tan(np.pi*float(j) \
               /float(2.*nr)))**2)+1./8.-0.25/(np.sin(np.pi*float(j)/ \
               float(2*nr))**2) - 0.5*float(nr*nr)
        return dnum

    dnum = ff(k+j, nr)+ff(k-j, nr)
    return dnum


def ff(i, nr):
    if i == 0:
        return 0
    ff = float(nr)*0.5/np.tan(np.pi*float(i)/float(2.*nr))

    a = i % 2
    if a == 0:
        ff = -ff
    return ff


def den(k, j, nr):
    if k == 0:
        den = 0.5*float(nr)
        a = j % 2
        if a == 1:
            den = -den
        if (j == 0 or j == nr):
            den = 1.
        return den

    den = float(nr)*np.sin(np.pi*float(k)/float(nr))
    if (j == 0 or j == nr):
        den = 2.*den
    return den


def cylSder(radius, data, order=4):
    """
    This function computes the s derivative of an input array defined on
    a regularly-spaced cylindrical grid.

    >>> s = linspace(0., 1., 129 ; dat = cos(s)
    >>> ddatds = cylSder(s, dat)

    :param radius: cylindrical radius
    :type radius: numpy.ndarray
    :param data: input data
    :type data: numpy.ndarray
    :param order: order of the finite-difference scheme (possible values are 2 or 4)
    :type order: int
    :returns: s derivative
    :rtype: numpy.ndarray
    """
    ns = data.shape[-1]
    ds = (radius.max()-radius.min())/(ns-1.)
    if order == 2:
        der = (np.roll(data, -1,  axis=-1)-np.roll(data, 1, axis=-1))/(2.*ds)
        der[..., 0] = (data[..., 1]-data[..., 0])/ds
        der[..., -1] = (data[..., -1]-data[..., -2])/ds
    elif order == 4:
        der = (   -np.roll(data,-2,axis=-1) \
               +8.*np.roll(data,-1,axis=-1) \
               -8.*np.roll(data, 1,axis=-1) \
                  +np.roll(data, 2,axis=-1)   )/(12.*ds)
        der[..., 1] = (data[..., 2]-data[..., 0])/(2.*ds)
        der[..., -2] = (data[..., -1]-data[..., -3])/(2.*ds)
        der[..., 0] = (data[..., 1]-data[..., 0])/ds
        der[..., -1] = (data[..., -1]-data[..., -2])/ds

    return der


def cylZder(z, data):
    """
    This function computes the z derivative of an input array defined on
    a regularly-spaced cylindrical grid.

    >>> z = linspace(-1., 1., 129 ; dat = cos(z)
    >>> ddatdz = cylZder(z, dat)

    :param z: height of the cylinder
    :type z: numpy.ndarray
    :param data: input data
    :type data: numpy.ndarray
    :returns: z derivative
    :rtype: numpy.ndarray
    """
    nz = data.shape[1]
    dz = (z.max()-z.min())/(nz-1.)
    der = (np.roll(data, -1,  axis=1)-np.roll(data, 1, axis=1))/(2.*dz)
    der[:, 0, :] = (data[:, 1, :]-data[:, 0, :])/dz
    der[:, -1, :] = (data[:, -1, :]-data[:, -2, :])/dz

    return der


def zavgpy(input, radius, ns, minc, normed=True):
    """
    This function computes a z-integration of a list of input arrays 
    (on the spherical grid). This works well for 2-D (phi-slice) 
    arrays. In case of 3-D arrays, only one element is allowed
    (too demanding otherwise).

    :param input: a list of 2-D or 3-D arrays
    :type input: list(numpy.ndarray)
    :param radius: spherical radius
    :type radius: numpy.ndarray
    :param ns: radial resolution of the cylindrical grid (nz=2*ns)
    :type ns: int
    :param minc: azimuthal symmetry
    :type minc: int
    :param normed: a boolean to specify if ones wants to simply integrate
                    over z or compute a z-average (default is True: average)
    :type normed: bool
    :returns: a python tuple that contains two numpy.ndarray and a
                list (height,cylRad,output) height[ns] is the height of the
                spherical shell for all radii. cylRad[ns] is the cylindrical
                radius. output=[arr1[ns], ..., arrN[ns]] contains
                the z-integrated output arrays.
    :rtype: tuple
    """
    nz = 2*ns
    ro = radius[0]
    ri = radius[-1]
    z = np.linspace(-ro, ro, nz)
    cylRad = np.linspace(0., ro, ns)
    cylRad = cylRad[1:-1]

    height = np.zeros_like(cylRad)
    height[cylRad >= ri] = 2.*np.sqrt(ro**2-cylRad[cylRad >= ri]**2)
    height[cylRad < ri] = 2.*(np.sqrt(ro**2-cylRad[cylRad < ri]**2)
                                -np.sqrt(ri**2-cylRad[cylRad < ri]**2))

    if len(input[0].shape) == 3:
        nphi = input[0].shape[0]
        phi = np.linspace(0., 2.*np.pi/minc, nphi)
        output = np.zeros((nphi, ns-2), dtype=input[0].dtype)
        for iphi in progressbar(range(nphi)):
            Z, S, out2D = sph2cyl_plane([input[0][iphi, ...]], radius, ns)
            S = S[:, 1:-1]
            Z = Z[:, 1:-1]
            output[iphi, :] = np.trapz(out2D[0][:, 1:-1], z, axis=0)
            if normed:
                output[iphi, :] /= height

        return height, cylRad, phi, np.array(output)
    elif len(input[0].shape) == 2:
        Z, S, out2D = sph2cyl_plane(input, radius, ns)
        S = S[:, 1:-1]
        Z = Z[:, 1:-1]
        output = []
        outIntZ = np.zeros((ns-2), dtype=input[0].dtype)
        for k,out in enumerate(out2D):
            outIntZ = np.trapz(out[:, 1:-1], z, axis=0)
            if normed:
                outIntZ /= height
            output.append(outIntZ)

        return height, cylRad, np.array(output)


def sph2cyl(g, ns=None, nz=None):
    """
    This function interpolates the three flow (or magnetic field)
    component of a :ref:`G_#.TAG <secGraphFile>` file
    on a cylindrical grid of size (ns, nz).

    .. warning:: This might be really slow!

    :param g: input graphic output file
    :type g: :py:class:`magic.MagicGraph`
    :param ns: number of grid points in the radial direction
    :type ns: int
    :param nz: number of grid points in the vertical direction
    :type nz: int
    :returns: a python tuple of five numpy.ndarray (S,Z,vs,vp_cyl,vz).
              S[nz,ns] is a meshgrid that contains the radial coordinate.
              Z[nz,ns] is a meshgrid that contains the vertical coordinate.
              vs[nz,ns] is the radial component of the velocity (or magnetic
              field), vp_cyl[nz,ns] the azimuthal component and vz[nz,ns] the
              vertical component.
    :rtype: tuple
    """
    if ns is None or nz is None:
        ns = g.nr ; nz = 2*ns

    theta = np.linspace(0., np.pi, g.ntheta)
    radius = g.radius[::-1]

    Z, S = np.mgrid[-radius.max():radius.max():nz*1j,0:radius.max():ns*1j]

    new_r = np.sqrt(S**2+Z**2).ravel()
    new_theta = np.arctan2(S, Z).ravel()
    ir = interp1d(radius, np.arange(len(radius)), bounds_error=False)
    it = interp1d(theta, np.arange(len(theta)), bounds_error=False)

    new_ir = ir(new_r)
    new_it = it(new_theta)
    new_ir[new_r > radius.max()] = len(radius)-1.
    new_ir[new_r < radius.min()] = 0.

    coords = np.array([new_it, new_ir])

    vr_cyl = np.zeros((g.npI, nz, ns), dtype=g.vr.dtype)
    vp_cyl = np.zeros_like(vr_cyl)
    vt_cyl = np.zeros_like(vr_cyl)
    for k in progressbar(range(g.npI)):
        dat = map_coordinates(g.vphi[k, :, ::-1], coords, order=3)
        dat[new_r > radius.max()] = 0.
        dat[new_r < radius.min()] = 0.
        vp_cyl[k, ...] = dat.reshape((nz, ns))
        dat = map_coordinates(g.vtheta[k, :, ::-1], coords, order=3)
        dat[new_r > radius.max()] = 0.
        dat[new_r < radius.min()] = 0.
        vt_cyl[k, ...] = dat.reshape((nz, ns))
        dat = map_coordinates(g.vr[k, :, ::-1], coords, order=3)
        dat[new_r > radius.max()] = 0.
        dat[new_r < radius.min()] = 0.
        vr_cyl[k, ...] = dat.reshape((nz, ns))

    th3D = np.zeros((g.npI, nz, ns), dtype=g.vr.dtype)
    for i in range(g.npI):
        th3D[i, ...] = np.arctan2(S, Z)
    vs = vr_cyl * np.sin(th3D) + vt_cyl * np.cos(th3D)
    vz = vr_cyl * np.cos(th3D) - vt_cyl * np.sin(th3D)

    return S, Z, vs, vp_cyl, vz


def sph2cyl_plane(data, rad, ns):
    """
    This function extrapolates a phi-slice of a spherical shell on
    a cylindrical grid

    >>> # Read G_1.test
    >>> gr = MagicGraph(ivar=1, tag='test') # example with a Magic file
    >>> # but can be done with any graphic file containing fields(phi,theta,r)
    >>> # phi-average v_\phi and s
    >>> vpm = gr.vphi.mean(axis=0)
    >>> sm = gr.entropy.mean(axis=0)
    >>> # Interpolate on a cylindrical grid
    >>> Z, S, outputs = sph2cyl_plane([vpm, sm], gr.radius, 512, 1024)
    >>> vpm_cyl, sm_cyl = outputs

    :param data: a list of 2-D arrays [(ntheta, nr), (ntheta, nr), ...]
    :type data: list(numpy.ndarray)
    :param rad: radius
    :type rad: numpy.ndarray
    :param ns: number of grid points in s direction
    :type ns: int
    :returns: a python tuple that contains two numpy.ndarray and a list (S,Z,output).
              S[nz,ns] is a meshgrid that contains the radial coordinate.
              Z[nz,ns] is a meshgrid that contains the vertical coordinate.
              output=[arr1[nz,ns], ..., arrN[nz,ns]] is a list of the interpolated
              array on the cylindrical grid.
    :rtype: tuple
    """
    ntheta, nr = data[0].shape
    theta = np.linspace(0., np.pi, ntheta)
    nz = 2*ns

    radius = rad[::-1]

    theta = np.linspace(0., np.pi, ntheta)

    Z, S = np.mgrid[-radius.max():radius.max():nz*1j,0:radius.max():ns*1j]

    new_r = np.sqrt(S**2+Z**2).ravel()
    new_theta = np.arctan2(S, Z).ravel()
    ir = interp1d(radius, np.arange(len(radius)), bounds_error=False)
    it = interp1d(theta, np.arange(len(theta)), bounds_error=False)

    new_ir = ir(new_r)
    new_it = it(new_theta)
    new_ir[new_r > radius.max()] = len(radius)-1.
    new_ir[new_r < radius.min()] = 0.

    coords = np.array([new_it, new_ir])

    output = []
    for dat in data:
        dat_cyl = map_coordinates(dat[:, ::-1], coords, order=3)
        dat_cyl[new_r > radius.max()] = 0.
        dat_cyl[new_r < radius.min()] = 0.
        dat_cyl = dat_cyl.reshape((nz, ns))
        output.append(dat_cyl)

    return S, Z, output


def progressbar(it, prefix="", size=60):
    """
    Fancy progress-bar for loops

    .. code-block:: python

           for i in progressbar(range(1000000)):
               x = i

    :type it: iterator
    :param prefix: prefix string before progress bar
    :type prefix: str
    :param size: width of the progress bar (in points of xterm width)
    :type size: int
    :type size: int
    """
    count = len(it)
    def _show(_i):
        x = int(size*_i/count)
        sys.stdout.write("{}[{}{}] {}/{}\r".format(prefix, "#"*x, "."*(size-x),
                                                   _i, count))
        sys.stdout.flush()

    _show(0)
    for i, item in enumerate(it):
        yield item
        _show(i+1)
    sys.stdout.write("\n")
    sys.stdout.flush()


def get_curl(grid, fr, ft, fp, l_mpi=False, l_spectral=False, sh_struct=None):
    """
    This function computes the spherical curl of an input 3D vector field
    (with its 3 components in spherical coordinates) using either finite
    differences or spectral derivatives in theta and phi.
    NOTE: this function is almost a useless wrapper only building the "grid helpers"!
          --> better to simply and directly call:
          >>> curl_sph(grid_help, fr, ft, fp, l_mpi, l_spectral, sh_struct)

    Inputs::
        -- grid: radial, theta and phi grid points respectively
        type: list of 3 np.array; np.float
        size: nrmax, ntmax, nphimax
        -- fr: radial component of a 3D vector field
        type: numpy.ndarray; np.float
        size: nphimax x ntmax x nrmax
        -- ft: theta component of a 3D vector field
        type: numpy.ndarray; np.float; size_like(fr)
        -- fp: phi component of a 3D vector field
        type: numpy.ndarray; np.float; size_like(fr)
        -- l_mpi: boolean to make use of the multiple threads of the computer to speed up the horizontal grad computations (only if l_spectral)
        type: logical
        -- l_spectral: boolean to decide if the horizontal derivatives should be computed in the spectral or physical space
        type: logical
        -- sh_struct: shtns Spectral Harmonics object with given l_max and m_max (orthonormalized)
        type: python object; shtns.sht
    Outputs::
        -- cr: radial component of the curl of the input 3D field
        type: numpy.ndarray; np.float
        size: nphimax x ntmax x nrmax
        -- ct: theta component of the curl of the input 3D field
        type: numpy.ndarray; np.float; size_like(cr)
        -- cp: phi component of the curl of the input 3D field
        type: numpy.ndarray; np.float; size_like(cr)
    Example::
        >>> gr = Pizza3DFields() # load the graphic file
        >>> grid = [gr.radius, gr.theta, gr.phi]
        >>> cbr, cbt, cbp = get_curl(grid, gr.br, gr.bt, gr.bp, gr.minc, l_spectral=False)
    """
    rad = grid[0]; theta = grid[1]#; phi = grid[2] # extract grid
    nrmax = len(rad); ntmax = len(theta)#; npmax = len(phi) # extract grid sizes
    #-- coordinates mesh-grid helper
    r3D = np.zeros_like(fr)
    for i in range(nrmax):
        r3D[:,:,i] = rad[i]
    th3D = np.zeros_like(ft)
    for j in range(ntmax):
        th3D[:,j,:] = theta[j]
    sint3D = np.sin(th3D) # NOTE: theta grid should NOT start at 0 to avoid problem in 1./s3D
    s3D = r3D*sint3D
    grid_help = [r3D, sint3D, s3D] # grid meshes
    #-- curl computation
    if ( l_spectral ):
        from shtns import sht
        if ( type(sh_struct) != sht ):
            print("SHTNotFoundError: If l_spectral is True, a shtns.sht structure is required! Zeros will be returned!")
            cr = np.zeros_like(fr)
            ct = np.zeros_like(ft)
            cp = np.zeros_like(fp)
            cr, ct, cp = curl_sph(grid_help, fr, ft, fp, l_mpi, True, sh_struct)
    else:
        cr, ct, cp = curl_sph(grid_help, fr, ft, fp, l_mpi, False) # equivalent to:
        #cr = 1./s3D*(thetaderavg(sint3D*fp, order=4) - phideravg(ft, minc=minc))
        #ct = 1./r3D*(phideravg(fr, minc=minc)/sint3D - rderavg(r3D*fp, rad=rad, exclude=False))
        #cp = 1./r3D*(rderavg(r3D*ft, rad=rad, exclude=False) - thetaderavg(fr, order=4))

    return cr, ct, cp


def curl_sph(grid_help, fr, ft, fp, l_mpi=False, l_spectral=False, sh_struct=None):
    """
    This function computes the 3 components of the spherical curl of an input 3D vector field
    (with its 3 components in spherical coordinates) using either finite differences or
    spectral derivatives in theta and phi.

    Inputs::
        -- grid_help: radial, sin(theta) and cylindrical radius grid points respectively
        type: list of 3 np.array; np.float
        size: nrmax x ntmax x nphimax
        -- fr: radial component of a 3D vector field
        type: numpy.ndarray; np.float
        size: nphimax x ntmax x nrmax
        -- ft: theta component of a 3D vector field
        type: numpy.ndarray; np.float; size_like(fr)
        -- fp: phi component of a 3D vector field
        type: numpy.ndarray; np.float; size_like(fr)
        -- l_mpi: boolean to make use of the multiple threads of the computer to speed up the horizontal grad computations (only if l_spectral)
        type: logical
        -- l_spectral: boolean to decide if the horizontal derivatives should be computed in the spectral or physical space
        type: logical
        -- sh_struct: shtns Spectral Harmonics object with given l_max and m_max (orthonormalized)
        type: python object; shtns.sht
    Outputs::
        -- cr: radial component of the curl of the input 3D field
        type: numpy.ndarray; np.float
        size: nphimax x ntmax x nrmax
        -- ct: theta component of the curl of the input 3D field
        type: numpy.ndarray; np.float; size_like(cr)
        -- cp: phi component of the curl of the input 3D field
        type: numpy.ndarray; np.float; size_like(cr)
    Example::
        >>> gr = Pizza3DFields() # load the graphic file
        >>> sh = shtns.sht(gr.l_max, gr.m_max)
        >>> nlat, nlon = sh.set_grid(nphi=gr.n_phi_max, nlat=gr.n_theta_max)
        >>> r3D = np.zeros_like(gr.br)
        >>> for i in range(nrmax):
        >>>     r3D[:,:,i] = gr.radius[i]
        >>> th3D = np.zeros_like(gr.br)
        >>> for j in range(ntmax):
        >>>     th3D[:,j,:] = gr.theta[j]
        >>> sint3D = np.sin(th3D)
        >>> s3D = r3D*sint3D
        >>> grid = [r3D, sint3D, s3D]
        >>> cbr, cbt, cbp = get_curl(grid, gr.br, gr.bt, gr.bp, True, sh)
    """
    r3D = grid_help[0]; sint3D = grid_help[1]; s3D = grid_help[2] # extract grid helper; s3D = r3D * sint3D
    #-- get radients; spectral precision on theta and phi if l_spectral = True
    if ( l_spectral ):
        from shtns import sht
        if ( type(sh_struct) != sht ): print("SHTNotFoundError: If l_spectral is True, a shtns.sht structure is required!")
    grp, gtp, _ = get_grad(r3D[0,0,:], s3D*fp, l_mpi, l_spectral, sh_struct) # d (r*sint * f_phi)/dr   ; d (r*sint * f_phi)/dtheta
    grt, _, gpt = get_grad(r3D[0,0,:], r3D*ft, l_mpi, l_spectral, sh_struct) # d (     r * f_theta)/dr ; d (     r * f_theta)/dphi
    _, gtr, gpr = get_grad(r3D[0,0,:],     fr, l_mpi, l_spectral, sh_struct) # d (         f_r)/dtheta ; d (         f_r)/dphi
    #-- curl computation from gradients
    cr = (gtp - gpt)/(sint3D*r3D**2) # ( r * d (sint * f_phi)/dtheta - r * d f_theta/dphi ) / (r*sint * r) = 1/s*( d (sint * f_phi)/dtheta   - d      f_theta/dphi )
    ct = (gpr - grp)/s3D # ( d f_r/dphi - sint * d (r * f_phi)/dr ) / (r*sint)                             = 1/r*( d         f_r/dphi / sint - d (r * f_phi)/dr )
    cp = (grt - gtr)/r3D # ( d (r * f_theta)/dr - d f_r/dtheta ) / r                                       = 1/r*( d (   r * f_theta)/dr     - d      f_r/dtheta )

    return cr, ct, cp


def get_grad(r, f, l_mpi=False, l_spectral=False, sh_struct=None):
    """
    This function computes the 3 components of the gradient of an input 3D field
    (in spherical coordinates) using either finite differences or spectral 
    derivatives in theta and phi.

    Inputs::
        -- r: radial grid points
        type: np.array; np.float
        size: nrmax
        -- f: 3D field
        type: numpy.ndarray; np.float
        size: nphimax x ntmax x nrmax
        -- l_mpi: boolean to make use of the multiple threads of the computer to speed up the horizontal grad computations (only if l_spectral)
        type: logical
        -- l_spectral: boolean to decide if the horizontal derivatives should be computed in the spectral or physical space
        type: logical
        -- sh_struct: shtns Spectral Harmonics object with given l_max and m_max (orthonormalized)
        type: python object; shtns.sht
    Outputs::
        -- gr: radial gradient of the input 3D field
        type: numpy.ndarray; np.float
        size: nphimax x ntmax x nrmax
        -- gt: theta gradient of the input 3D field
        type: numpy.ndarray; np.float; size_like(gr)
        -- gt: phi gradient of the input 3D field
        type: numpy.ndarray; np.float; size_like(gr)
    Example::
        >>> gr = Pizza3DFields() # load the graphic file
        >>> sh = shtns.sht(gr.l_max, gr.m_max)
        >>> nlat, nlon = sh.set_grid(nphi=gr.n_phi_max, nlat=gr.n_theta_max)
        >>> grbr, gtbr, gpbr = get_vec_grad(gr.radius, gr.br, l_spectral=True, sh)
    """
    nrmax = len(r) # extract radial grid size
    ff = f.astype(np.float64) # NOTE: simply better to use dtype='float64' and not 'float32'
    if ( l_mpi ): #-- NOTE: will need to call the script using it with $ mpiexec -n X python3 your_script.py
        import mpi4py.MPI as mpi
        comm = mpi.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        rsize = nrmax//size
        rStart = rank*rsize
        gt = np.zeros_like(ff); gp = np.zeros_like(ff)
        gtc = np.zeros((ff.shape[0], ff.shape[1], rsize), dtype=np.float64); gpc = np.zeros_like(gtc)
        gtrec = np.zeros((size, ff.shape[0], ff.shape[1], rsize), dtype=np.float64); gprec = np.zeros_like(gtrec)
    else:
        size = 1; rsize = nrmax; rStart = 0
        gtc = np.zeros_like(ff); gpc = np.zeros_like(ff)
    if ( l_spectral ):
        from shtns import sht
        if ( type(sh_struct) != sht ):
            print("SHTNotFoundError: If l_spectral is True, a shtns.sht structure is required! Zeros will be returned for theta and phi components!")
        else:
            flm = np.zeros((sh_struct.nlm,rsize), dtype=np.complex128)
            for i in range(rsize):
                #ff = f.astype(np.float64) # NOTE: array has to be dtype='float64' and not 'float32'
                flm[:,i] = sh_struct.analys(ff[:,:,rStart+i].T) # NOTE: need to give theta dimension first
                gtr, _ = sh_struct.synth_grad(flm[:,i])#*(sh_struct.l*(sh_struct.l + 1))) # theta- -derivatives
                gpr = sh_struct.synth(flm[:,i]*sh_struct.m*complex(0.,1.))# NOTE: phi-der spurious in synth_grad, better to simply do i.m*f
                gtc[:,:,i] = gtr.T; gpc[:,:,i] = gpr.T
    else:
        #print("MPINotUsedWarning: If l_spectral is False, mpi can not be used to speed up horizontal gradient computations")
        for i in range(rsize):
            gtr = thetaderavg(ff[:,:,rStart+i].T, order=4) # theta-derivative # NOTE: if fields are 2D, need to give theta dimension first (only for theta...)
            gpr = phideravg(ff[:,:,rStart+i], minc=1) # phi-derivative # NOTE: need to resymmetrize the date beforehand!
            gtc[:,:,i] = gtr.T; gpc[:,:,i] = gpr
    if ( l_mpi ):
        comm.Barrier()
        comm.Gather(gtc, gtrec, root=0)
        comm.Gather(gpc, gprec, root=0)
        for n in range(size):
            gt[:,:,n*rsize:(n+1)*rsize] = gtrec[n,:,:,:]
            gp[:,:,n*rsize:(n+1)*rsize] = gprec[n,:,:,:]
    else:
        gt = gtc; gp = gpc
    gr = rderavg(ff, rad=r, exclude=False) # r-derivative

    return gr, gt, gp
