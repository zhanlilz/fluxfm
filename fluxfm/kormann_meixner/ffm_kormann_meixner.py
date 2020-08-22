# Kormann & Meixner footprint model (Kormann and Meixner, 2001). The
# implementation is based on a MATLAB script originally by Jakob Sievers
# (05/2013) but revised and annotated by Christian Wille.
# 
# Zhan Li, zhanli@gfz-potsdam.de
# Created: Sat Aug 22 18:41:23 CEST 2020

import numpy as np
from scipy import ndimage as spndimage

def estimateZ0(zm, ws, wd, ustar, mo_len):
    """Estimate roughness lengths based on Kormann-Meixner footprint model. 

    Estimate roughness length (z0) by relating power-law wind profile in
    Kormann & Meixner's footprint model (Kormann and Meixner, 2001) to log wind
    profile in Monin-Obukhov similarity theory. 

    The number of input measurement intervals should be large enough to cover
    all possible wind directions because this function outputs roughness
    lengths that are smoothed within a 45-degree window of wind directions at
    1-degree step. 

    The implementation is based on a MATLAB script originally by Jakob Sievers
    (05/2013) but revised and annotated by Christian Wille.

    Parameters
    ----------
    zm : ndarray of shape (n_intervals,)
        The list of measurement height (meter) per measurement intervals. 

    ws : ndarray of shape (n_intervals,)
        The list of wind speed (m*s^-1) per measurement interval.

    wd : ndarray of shape (n_intervals,)
        The list of wind direction (degree) per measurement interval.

    ustar : ndarray of shape (n_intervals,)
        The list of friction velocity (m*s^-1) per measurement interval.

    mo_len : ndarray of shape (n_intervals,)
        The list of Monin-Obukhov length (meter) per measurement interval. 

    Returns
    -------
    z0med : ndarray of shape (n_intervals,)
        The list of estimated roughness length, z0 (meter) per measurement
        interval.

    Refernces
    ---------
    Kormann, R., Meixner, F.X., 2001. An Analytical Footprint Model For
    Non-Neutral Stratification. Boundary-Layer Meteorology 99, 207–224.
    https://doi.org/10.1023/A:1018991015119
    """
    # von Karman constant
    k = 0.4

    n_intervals = len(zm)
    # Check if inputs are of the same length
    if (n_intervals != len(ws) \
            or n_intervals != len(wd) \
            or n_intervals != len(ustar) \
            or n_intervals != len(mo_len)):
        raise RuntimeError("Input parameters must be of the same length!")

    psi_m = _psiM(zm, mo_len)
    # roughness length z0, a solution for z0 to the Eq. (31) in (Kormann and
    # Meixner, 2001)
    z0 = zm * np.exp(psi_m - (k * ws / ustar))

    # remove outliers
    z0[z0>1000] = np.nan;

    # NOTE: here z0, raw values of roughness length from solving Eq. (31) in
    # (Kormann and Meixner, 2001) is smoothed using median in a 45-deg window
    # of wind directions at 1-degree step. This is where the size of input
    # measurement intervals matter.
    halfbinwidth = 22
    z0med = np.zeros_like(z0) + np.nan;
    for kk in range(0, 360):
        wd_wrapped = wd;
        if kk<90:
            wd_wrapped[wd>270] = wd[wd>270] - 360
        elif kk>270:
            wd_wrapped[wd<90] = wd[wd<90] + 360
        idx1 = np.logical_and(wd >= kk, wd < (kk+1))        
        idx2 = np.logical_and(wd_wrapped >= (kk-halfbinwidth), wd_wrapped < (kk+1+halfbinwidth))
        z0med[idx1] = np.nanmedian(z0[idx2])
    return z0med

def _phiM(zm, mo_len):
    """Calculate phi_m using Eq. (33), the stability function in (Kormann and
    Meixner, 2001)

    Parameters
    ----------

    Returns
    -------
        phi_m : ndarray of shape (n_intervals,)
            Values of phi_m in (Kormann and Meixner, 2001).
    """
    phi_m = np.zeros_like(zm)
    sflag = mo_len < 0
    phi_m[sflag] = (1 - 16 * zm[sflag] / mo_len[sflag])**(-0.25)
    sflag = mo_len >= 0
    phi_m[sflag] = 1 + 5 * zm[sflag] / mo_len[sflag]
    return phi_m

def _phiC(zm, mo_len):
    """Calculate phi_c using Eq. (34), the stability function in (Kormann and
    Meixner, 2001)

    Parameters
    ----------

    Returns
    -------
        phi_c : ndarray of shape (n_intervals,)
            Values of phi_c in (Kormann and Meixner, 2001).
    """
    phi_c = np.zeros_like(zm)
    sflag = mo_len < 0
    phi_c[sflag] = (1 - 16 * zm[sflag] / mo_len[sflag])**(-0.5)
    sflag = mo_len >= 0
    phi_c[sflag] = 1 + 5 * zm[sflag] / mo_len[sflag]
    return phi_c

def _psiM(zm, mo_len):
    """Calculate psi_m using Eq. (35), the diabatic integration of the wind
    profile (using 1/phi_m as zeta), in (Kormann and Meixner, 2001)

    Parameters
    ----------

    Returns
    -------
        psi_m : ndarray of shape (n_intervals,)
            Values of psi_m in (Kormann and Meixner, 2001).
    """
    psi_m = np.zeros_like(zm)
    sflag = mo_len < 0
    inv_phi_m = (1 - 16 * zm[sflag] / mo_len[sflag])**(0.25)
    psi_m[sflag] = -2 * np.log(0.5 * (1 + inv_phi_m)) \
            - np.log(0.5 * (1 + inv_phi_m**2)) \
            + 2 * np.arctan(inv_phi_m) - np.pi * 0.5
    sflag = mo_len >= 0
    psi_m[sflag] = 5 * zm[sflag] / mo_len[sflag]
    return psi_m
