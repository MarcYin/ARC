import numpy as np
import math
from scipy.stats import poisson
from numba import njit

LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')

# @njit
def fast_factorial(n):
    if n > 20:
        raise ValueError
    return LOOKUP_TABLE[n]

# @njit
def tav(alpha, nr):
    # from https://github.com/Christiaanvandertol/SCOPE/blob/master/src/RTMs/BSM.m
    #Stern's formula in Lekner & Dorf (1988) gives reflectance for alfa = 90 degrees
    # y1 = (3*n2+2*nr+1)./(3*(nr+1).^2);
    # y2 = 2*nr.^3.*(nr.^2+2*nr-1)./(np.^2.*nm);
    # y3 = n2.*np.*log(nr)./nm.^2;
    # y4 = n2.*nm.^2./np.^3.*log((nr.*(nr+1)./(nr-1)));
    # st = y1-y2+y3-y4;


    n2 =   nr**2
    np_ =   n2 + 1
    nm_ =   n2 - 1
    a  = (nr + 1)**2 / 2
    k = -(n2 - 1)**2 / 4

    sin_a = np.sin(np.deg2rad(alpha))
    if alpha != 0:
        B2 = sin_a**2 - np_ / 2
        if alpha == 90:
            B1 = B2 * 0
        else:
            B1 = np.sqrt(B2**2 + k)
        b = B1 - B2
        b3 = b**3
        a3 = a**3
        ts = (k**2 / (6 * b3)+ k / b - b / 2) - (k**2 / (6 * a3)+ k / a - a / 2)
        tp1 = -2 * n2 * (b - a) / np_**2
        tp2 = -2 * n2 * np_ * (np.log(b / a)) / nm_**2
        tp3 = n2 * (1 / b - 1 / a) / 2
        tp4 = 16 * n2**2 * (n2**2 + 1) * (np.log((2 * np_ * b - nm_**2) / (2 * np_*a - nm_**2))) / (np_**3 * nm_**2)
        tp5 = 16 * n2**2 * n2          * (1 / (2 * np_ * b - nm_**2) - 1 / (2 * np_ * a - nm_**2)) / np_**3
        tp = tp1 + tp2 + tp3 + tp4 + tp5
        Tav = (ts + tp) / (2 * sin_a**2)
    else:
        Tav = 4 * nr / (nr + 1)**2
    return Tav

# @njit
def soilwat(rdry,nw,kw,SMp,SMC,deleff):
    # from https://github.com/Christiaanvandertol/SCOPE/blob/master/src/RTMs/BSM.m
    k = np.arange(7)
    nk = len(k)
    mu = (SMp - 5) / SMC
    if mu <= 0:
        rwet = rdry
    else:
        # Lekner & Dorf (1988) modified soil background reflectance
        # for soil refraction index = 2.0; uses the tav-function of PROSPECT
        rbac = 1 - (1-rdry) * (rdry * tav(90, 2.0 / nw) / tav(90, 2.0) + 1 - rdry) # Rbac
        # total reflectance at bottom of water film surface
        p    = 1 - tav(90, nw) / nw**2   # rho21, water to air, diffuse

        # reflectance of water film top surface, use 40 degrees incidence angle,
        # like in PROSPECT
        Rw  = 1 - tav(40,nw)             # rho12, air to water, direct


        # fraction of areas
        # P(0)   = dry soil area            fmul(1)
        # P(1)   = single water film area   fmul(2)
        # P(2)   = double water film area   fmul(3)
        # without loop
#         fmul    =   poisson.pmf(k, mu)                        # Pobability 
        fmul = np.zeros((nk, 1))
        for i in range(len(k)):
            ret = np.exp(-mu) * mu**i / fast_factorial(i)
            fmul[i] = ret
        tw      =   np.exp(-2 * kw * deleff * k);                   # two-way transmittance,exp(-2*kw*k Delta)
        Rwet_k  =   Rw + (1 - Rw) * (1 - p) * tw * rbac / (1 - p * tw * rbac)
        rwet    =   rdry * fmul[0] + Rwet_k[:, 1:].dot(fmul[1:]) 
    return rwet

def BSM(B, lat, lon, SMp, BSM_paras):
    SMC     = 25  # empirical parameter (fixed) for BSM
    film    = 0.015 #empirical parameter (fixed) for BMS
    GSV, nw, kw = BSM_paras
    
#     GSV = BSM_paras.GSV
#     nw  = BSM_paras.nw
#     kw  = BSM_paras.kw
    
    latRad = np.deg2rad(lat)
    lonRad = np.deg2rad(lon)
    f1 = B * np.sin(latRad)
    f2 = B * np.cos(latRad) * np.sin(lonRad)
    f3 = B * np.cos(latRad) * np.cos(lonRad)
#     rdry = f1 * GSV[:, [0]] + f2 * GSV[:, [1]] + f3 * GSV[:, [2]]
    rdry = f1 * GSV[:, 0] + f2 * GSV[:, 1] + f3 * GSV[:, 2]
    rdry = np.expand_dims(rdry, axis=1)
    
    rwet = soilwat(rdry,nw,kw,SMp,SMC,film)
    
    return rwet
    