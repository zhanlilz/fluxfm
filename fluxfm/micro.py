"""
Collection of common constants and equations used in micrometeorology, e.g.,
constants and equations in various similarity theories. 
"""

# Author: Zhan Li <zhanli@gfz-potsdam.de>

import numpy as np

VON_KARMAN = 0.4
# constant in the universal function phi_m under unstable condition (with
# suffix _U) and stabel condition (with suffix _S).
# See Table 2.8 in (Foken, 2008). Also refer to Eq. (33) in (Kormann and
# Meixner, 2001), Eq. (4) & (5) in (Graf et al., 2014).
PHI_M_CONST_U = 19.3
PHI_M_CONST_S = 6

def _phi_m(zm, mo_len):
    """Calculate phi_m using Eq. (33), the stability function in (Kormann and
    Meixner, 2001)

    Parameters
    ----------
    zm : array-like of shape (n_obs,)
        List of effective/aerodynamic height (meter) per observation at one
        time step. 

    mo_len : arry-like of shape (n_obs,)
        List of Monin-Obukhov length (meter) per observation at one time step. 

    Returns
    -------
        phi_m : ndarray of shape (n_obs,)
            Values of phi_m in (Kormann and Meixner, 2001).
    """
    phi_m = np.zeros_like(zm)
    sflag = mo_len < 0
    phi_m[sflag] = (1 - PHI_M_CONST_U * zm[sflag] / mo_len[sflag])**(-0.25)
    sflag = mo_len > 0
    phi_m[sflag] = 1 + PHI_M_CONST_S * zm[sflag] / mo_len[sflag]
    return phi_m

def _phi_c(zm, mo_len):
    """Calculate phi_c using Eq. (34), the stability function in (Kormann and
    Meixner, 2001)

    Parameters
    ----------
    zm : array-like of shape (n_obs,)
        List of effective/aerodynamic measurement height (meter) per
        observation at one time step. 

    mo_len : arry-like of shape (n_obs,)
        List of Monin-Obukhov length (meter) per observation at one time step. 

    Returns
    -------
        phi_c : ndarray of shape (n_obs,)
            Values of phi_c in (Kormann and Meixner, 2001).
    """
    phi_c = np.zeros_like(zm)
    sflag = mo_len < 0
    phi_c[sflag] = (1 - PHI_M_CONST_U * zm[sflag] / mo_len[sflag])**(-0.5)
    sflag = mo_len > 0
    phi_c[sflag] = 1 + PHI_M_CONST_S * zm[sflag] / mo_len[sflag]
    return phi_c

def _psi_m(zm, mo_len):
    """Calculate psi_m using Eq. (35), the diabatic integration of the wind
    profile (using 1/phi_m as zeta), in (Kormann and Meixner, 2001)

    Parameters
    ----------
    zm : array-like of shape (n_obs,)
        List of effective/aerodynamic measurement height (meter) per
        observation at one time step. 

    mo_len : arry-like of shape (n_obs,)
        List of Monin-Obukhov length (meter) per observation at one time step. 

    Returns
    -------
        psi_m : ndarray of shape (n_obs,)
            Values of psi_m in (Kormann and Meixner, 2001).
    """
    psi_m = np.zeros_like(zm)
    sflag = mo_len < 0
    inv_phi_m = (1 - PHI_M_CONST_U * zm[sflag] / mo_len[sflag])**(0.25)
    psi_m[sflag] = -2 * np.log(0.5 * (1 + inv_phi_m)) \
            - np.log(0.5 * (1 + inv_phi_m**2)) \
            + 2 * np.arctan(inv_phi_m) - np.pi * 0.5
    sflag = mo_len > 0
    psi_m[sflag] = PHI_M_CONST_S * zm[sflag] / mo_len[sflag]
    return psi_m

def _m_param(zm, ws, ustar, mo_len):
    """Estimate the parameter m, in Eq. (11) of (Kormann and Meixner, 2001),
    exponent of wind velocity, in u(z)=U*z^m under the assumption of a
    power-law wind profile.

    Parameters
    ----------
    zm : array-like of shape (n_obs,)
        List of effective/aerodynamic measurement height (meter) per
        observation at one time step. 

    ws : array-like of shape (n_obs,)
        List of wind speed (m*s^-1) per observation at one time step.

    ustar : array-like of shape (n_obs,)
        List of friction velocity (m*s^-1) per observation at one time step.

    mo_len : array-like of shape (n_obs,)
        List of Monin-Obukhov length (meter) per observation at one time step. 

    Returns
    -------
        m : ndarray of shape (n_obs,)
            Values of m in the Eq. (11) of (Kormann and Meixner, 2001).
    """
    # Estimate m using the analytical approach in (Kormann and Meixner, 2001),
    # following the Eq. (36). 
    phi_m = _phiM(zm, mo_len)
    m = ustar * phi_m / (VON_KARMAN * ws)
    return m

def _n_param(zm, mo_len):
    """TO FIX CONSTANTS
    Estimate the parameter n, in Eq. (11) of (Kormann and Meixner, 2001),
    constant of eddy diffusivity, in K(z)=kappa*z^n under the assumption of a
    power-law wind profile.

    Parameters
    ----------
    zm : array-like of shape (n_obs,)
        List of effective/aerodynamic measurement height (meter) per
        observation at one time step. 

    mo_len : array-like of shape (n_obs,)
        List of Monin-Obukhov length (meter) per observation at one time step. 

    Returns
    -------
        n : ndarray of shape (n_obs,)
            Values of n in the Eq. (11) of (Kormann and Meixner, 2001).
    """
    # Estimate m using the analytical approach in (Kormann and Meixner, 2001),
    # following the Eq. (36).
    n = np.zeros_like(zm)
    sflag = mo_len < 0
    n[sflag] = (1 - 24 * zm[sflag] / mo_len[sflag]) \
            / (1 - 16 * zm[sflag] / mo_len[sflag])
    sflag = mo_len > 0
    n[sflag] = 1 / (1 + 5 * zm[sflag] / mo_len[sflag])
    return n

def _monin_obukhov_z0(zm, ws, ustar, mo_len):
    """Estimate z0 (aerodynamic roughness length) using Monin-Obukhov
    similarity theory under the assumption of a power-law wind profile. See Eq.
    (31) in [1]_, Eq. (6) in [2]_, or Section 2.3.2 in [3]_ for a detailed
    overview.

    Parameters
    ----------
    zm : array-like of shape (n_obs,)
        List of effective/aerodynamic measurement height (meter) per
        observation at one time step. 

    ws : array-like of shape (n_obs,)
        List of wind speed (m*s^-1) per observation at one time step.

    ustar : array-like of shape (n_obs,)
        List of friction velocity (m*s^-1) per observation at one time step.

    mo_len : array-like of shape (n_obs,)
        List of Monin-Obukhov length (meter) per observation at one time step. 

    Returns
    -------
    z0 : ndarray of shape (n_obs,)
        Estimated z0 (aerodynamic roughness length).

    References
    ----------
    .. [1] Kormann, R., Meixner, F.X., 2001. An Analytical Footprint Model For
    Non-Neutral Stratification. Boundary-Layer Meteorology 99, 207–224.
    https://doi.org/10.1023/A:1018991015119
    
    .. [2] Graf, A., van de Boer, A., Moene, A., Vereecken, H., 2014.
    Intercomparison of Methods for the Simultaneous Estimation of Zero-Plane
    Displacement and Aerodynamic Roughness Length from Single-Level
    Eddy-Covariance Data. Boundary-Layer Meteorol 151, 373–387.
    https://doi.org/10.1007/s10546-013-9905-z

    .. [3] Foken, T., 2008. Micrometeorology. Springer Berlin Heidelberg,
    Berlin, Heidelberg. https://doi.org/10.1007/978-3-540-74666-9

    """
    psi_m = _psi_m(zm, mo_len)
    # roughness length z0, a solution for z0 to the Eq. (31) in (Kormann and
    # Meixner, 2001)
    z0 = zm * np.exp(psi_m - (VON_KARMAN * ws / ustar))

    return z0

def _monin_obukhov_u(zm, z0, ustar, mo_len):
    """Estimate horizontal wind speed (u) using Monin-Obukhov similarity theory
    under the assumption of a power-law wind profile. See Eq.  (31) in [1]_,
    Eq. (6) in [2]_, or Section 2.3.2 in [3]_ for a detailed overview.

    Parameters
    ----------
    zm : array-like of shape (n_obs,)
        List of effective/aerodynamic measurement height (meter) per
        observation at one time step. 

    z0 : array-like of shape (n_obs,)
        List of roughness length (meter) per observation at one time step.

    ustar : array-like of shape (n_obs,)
        List of friction velocity (m*s^-1) per observation at one time step.

    mo_len : array-like of shape (n_obs,)
        List of Monin-Obukhov length (meter) per observation at one time step. 

    Returns
    -------
    u : ndarray of shape (n_obs,)
        Estimated u (horizontal wind sped).

    References
    ----------
    .. [1] Kormann, R., Meixner, F.X., 2001. An Analytical Footprint Model For
    Non-Neutral Stratification. Boundary-Layer Meteorology 99, 207–224.
    https://doi.org/10.1023/A:1018991015119
    
    .. [2] Graf, A., van de Boer, A., Moene, A., Vereecken, H., 2014.
    Intercomparison of Methods for the Simultaneous Estimation of Zero-Plane
    Displacement and Aerodynamic Roughness Length from Single-Level
    Eddy-Covariance Data. Boundary-Layer Meteorol 151, 373–387.
    https://doi.org/10.1007/s10546-013-9905-z

    .. [3] Foken, T., 2008. Micrometeorology. Springer Berlin Heidelberg,
    Berlin, Heidelberg. https://doi.org/10.1007/978-3-540-74666-9

    """
    psi_m = _psi_m(zm, mo_len)
    # roughness length z0, a solution for z0 to the Eq. (31) in (Kormann and
    # Meixner, 2001)
    u = ustar / VON_KARMAN * (np.log(zm / z0) + psi_m) 

    return u

