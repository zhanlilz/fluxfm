#!/usr/bin/env python
#
# Estimate z0 and z/d simultaneously from single-level micrometeorological
# measurements using the approaches given in Graf et al., 2011.
# 
# Zhan Li, zhanli@gfz-potsdam.de
# Created: Tue Oct  6 17:31:52 CEST 2020
# 
# Graf, A., van de Boer, A., Moene, A., Vereecken, H., 2014. Intercomparison of
# Methods for the Simultaneous Estimation of Zero-Plane Displacement and
# Aerodynamic Roughness Length from Single-Level Eddy-Covariance Data.
# Boundary-Layer Meteorol 151, 373–387.
# https://doi.org/10.1007/s10546-013-9905-z

import warnings

import numpy as np
from sklearn.linear_model import LinearRegression

from micro import VON_KARMAN, PHI_M_CONST_U, PHI_M_CONST_S
from micro import _psi_m, _monin_obukhov_z0

from utils import check_is_fitted

class SurfaceAerodynamicFPRE():
    """Flux-Profile-based Regression estimator of surface aerodynamic
    parameters. 

    This estimator takes single-level micrometeorological measurements to
    simultaneously estimate two surface aerodynamic parameters including
    effective/aerodynamic measurement height z and aerodynamic roughness length
    z0 using the regression method based on flux-profile similarity theory
    [1]_. 

    Parameters
    ----------
    solver : str {'univariate', 'bivariate'}
        If 'univariate':
          Use the Eq. (9) and (10) in [1]_ to solve a univariate linear
          regression.
        If 'bivariate':
          Use the Eq. (9) and (11) in [1]_ to solve a bivariate linear
          regression. 

    min_nobs : float
        Minimum number of observations that meet the applicability criteria
        given by Table 1 in [1]_ to carry out the estimates.

    Attributes
    ----------
    N : integer
        Number of valid observations that meet the applicability criteria given
        by Table 1 in [1]_.

    Nin : integer
        Number of input observations after excluding NaN and Inf.

    References
    ----------
    .. [1] Graf, A., van de Boer, A., Moene, A., Vereecken, H., 2014.
    Intercomparison of Methods for the Simultaneous Estimation of Zero-Plane
    Displacement and Aerodynamic Roughness Length from Single-Level
    Eddy-Covariance Data.  Boundary-Layer Meteorol 151, 373–387.
    https://doi.org/10.1007/s10546-013-9905-z
    """

    def __init__(
            self, 
            solver='univariate', 
            min_nobs=10):
        self.solver = solver
        self.min_nobs = min_nobs

    def _filter(self, data, z0max, zmax):
        """Filter data according to the applicabilit criteria in Table 1 in
        [1]_.

        Parameters
        ----------
        data : ndarray of shape (n_obs, 3)
            Input ``n_obs`` observations of 3 variables from micrometeorology
            measurements in 3 columns of ``data``: alongwind speed, friction
            velocity, and Monin-Obukohv length, in that order. 
 
        z0max : float
            Upper bound to the expected z0 values (aerodynamic surface
            roughness length, e.g. 10% of canopy height for vegetated surface),
            in meters.

        zmax : float
            Upper bound to the expected z values (effective/aerodynamic
            measurement height, in meters.
       
        Returns
        -------
        sdata : ndarray of shape (N_, 3)
            Selected ``N_`` observations of 3 variables after filtering. 
        """
        data = data.copy()

        self.z0max_ = z0max
        self.zmax_ = zmax

        sflag = np.logical_or(np.isnan(data), np.isinf(data))
        sflag = np.logical_not(sflag)
        sflag = np.all(sflag, axis=1)
        data = data[sflag, :]
        self.Nin_ = data.shape[0]

        sflag_arr = [ 
                1 / data[:, 2] < -0.103 / zmax, 
                1 / data[:, 2] < -0.084 / z0max, 
                1 / data[:, 2] >  0.037 / z0max, 
                1 / data[:, 2] >      1 / zmax, 
                ]
        sflag = np.logical_not(np.any(np.vstack(sflag_arr).T, axis=1))
        data = data[sflag, :]
        self.N_ = data.shape[0]
        return data
    
    def _calc_xy(self, data):
        """Calculate response variable y and explanatory variable X for the
        linear regression approach. 

        Returns
        -------
        X : ndarray of shape (N_, 1 or 2)
            If the solver is 'univariate', X has the shape of (N_, 1). Refer to
            Eq.  (10) in [1]_. If the solver is 'bivariate', X has the shape
            (N_, 2). Refer to Eq. (11) in [1]_, with the 1st column of X being
            :math:`u_*/\kappa`, and the 2nd column of X being :math:`(u_*
            \\beta)/(\kappa L)`

        y : ndarray of shape (N_, 1)
            Refer to Eq. (10) or (11) in [1]_.

        References
        ----------
        .. [1] Graf, A., van de Boer, A., Moene, A., Vereecken, H., 2014.
        Intercomparison of Methods for the Simultaneous Estimation of
        Zero-Plane Displacement and Aerodynamic Roughness Length from
        Single-Level Eddy-Covariance Data.  Boundary-Layer Meteorol 151,
        373–387.  https://doi.org/10.1007/s10546-013-9905-z
        """
        if self.solver == 'univariate':
            X = PHI_M_CONST_S / data[:, 2] 
            y = data[:, 0] * VON_KARMAN / data[:, 1] 
            X, y = X[:, np.newaxis], y[:, np.newaxis]
        elif self.solver == 'bivariate':
            X = [
                    data[:, 1] / VON_KARMAN, 
                    data[:, 1]*PHI_M_CONST_S / (VON_KARMAN*data[:, 2])]
            X = np.vstack(X).T
            y = data[:, 0][:, np.newaxis]
        else:
            raise ValueError('Unrecognized solver={0:s}'.format(self.solver))
        return X, y

    def fit_transform(self, data, z0max, zmax):
        """Build the estimator from single-level micrometeorolgoical data and
        apply it to estimate surface aerodynamic parameters.

        Parameters
        ----------
        data : ndarray of shape (n_obs, 3)
            Input ``n_obs`` observations of 3 variables from micrometeorology
            measurements in 3 columns of ``data``: alongwind speed, friction
            velocity, and Monin-Obukohv length, in that order. 

        z0max : float
            Upper bound to the expected z0 values (aerodynamic surface
            roughness length, e.g. 10% of canopy height for vegetated surface),
            in meters.

        zmax : float
            Upper bound to the expected z values (effective/aerodynamic
            measurement height, in meters.

        Returns
        -------
        z : float 
            Estimated z (effective/aerodynamic measurement height). 

        z0 : float 
            Estimated z0 (surface aerodynamic roughness length). 
        """
        data = self._filter(data, z0max, zmax)
        if self.N_ < self.min_nobs: 
            warnings.warn(
                    'too few data points left after validation.'
                    ' return nan')
            z, z0 = np.nan, np.nan
            self.coef_, self.intercept_ = None, None
        else:
            X, y = self._calc_xy(data)
            if self.solver == 'univariate':
                # 1st regression model (univariate linear regression)
                # Eq. 10 in Graf et al. 2014
                reg = LinearRegression().fit(X, y)
                z = reg.coef_[0, 0]
                z0 = z / np.exp(reg.intercept_[0])
            elif self.solver == 'bivariate':
                # 2nd regression model (bivariate linear regression)
                # Eq. 11 in Graf et al. 2014
                reg = LinearRegression(fit_intercept=False).fit(X, y)
                z = reg.coef_[0, 1]
                z0 = z / np.exp(reg.coef_[0, 0])
            else:
                raise ValueError(
                        'Unrecognized solver={0:s}'.format(self.solver))
            self.coef_ = reg.coef_
            self.intercept_ = reg.intercept_
        return z, z0

class SurfaceAerodynamicFPIT():
    """Flux-Profile-based Iterative estimator of surface aerodynamic
    parameters. 

    This estimator takes single-level micrometeorological measurements to
    simultaneously estimate two surface aerodynamic parameters including
    effective/aerodynamic measurement height z and aerodynamic roughness length
    z0 using the iterative method based on flux-profile similarity theory
    [1]_, [2]_. 

    Parameters
    ----------
    solver : str {'sigma-s', 'sigma-s-approx'}
        If 'sigma-s':
          Use the Eq. (6) and (7) in [1]_ to solve the minimization of the
          exact version of :math:`\Sigma_{S}`.
        If 'sigma-s-approx':
          Use the Eq. (6) and (8) in [1]_ to solve the minimization of the
          approximate version of :math:`\Sigma_{S}`.

    min_nobs : float
        Minimum number of observations that meet the applicability criteria
        given by Table 1 in [1]_ to carry out the estimates.

    Attributes
    ----------
    N : integer
        Number of valid observations that meet the applicability criteria given
        by Table 1 in [1]_.

    Nin : integer
        Number of input observations after excluding NaN and Inf.

    References
    ----------
    .. [1] Graf, A., van de Boer, A., Moene, A., Vereecken, H., 2014.
    Intercomparison of Methods for the Simultaneous Estimation of Zero-Plane
    Displacement and Aerodynamic Roughness Length from Single-Level
    Eddy-Covariance Data.  Boundary-Layer Meteorol 151, 373–387.
    https://doi.org/10.1007/s10546-013-9905-z

    .. [2] Martano, P., 2000. Estimation of Surface Roughness Length and
    Displacement Height from Single-Level Sonic Anemometer Data. J. Appl.
    Meteor. 39, 708–715.
    https://doi.org/10.1175/1520-0450(2000)039<0708:EOSRLA>2.0.CO;2

    """
    def __init__(
            self, 
            solver='sigma-s', 
            min_nobs=10):
        self.solver = solver
        self.min_nobs = min_nobs

    def _filter(self, data, z0max, zmax):
        """Filter data according to the applicabilit criteria in Table 1 in
        [1]_.

        Parameters
        ----------
        data : ndarray of shape (n_obs, 3)
            Input ``n_obs`` observations of 3 variables from micrometeorology
            measurements in 3 columns of ``data``: alongwind speed, friction
            velocity, and Monin-Obukohv length, in that order. 

        z0max : float
            Upper bound to the expected z0 values (aerodynamic surface
            roughness length, e.g. 10% of canopy height for vegetated surface),
            in meters.

        zmax : float
            Upper bound to the expected z values (effective/aerodynamic
            measurement height, in meters.

        Returns
        -------
        sdata : ndarray of shape (N_, 3)
            Selected ``N_`` observations of 3 variables after filtering. 
        """
        data = data.copy()

        self.z0max_ = z0max
        self.zmax_ = zmax

        sflag = np.logical_or(np.isnan(data), np.isinf(data))
        sflag = np.logical_not(sflag)
        sflag = np.all(sflag, axis=1)
        data = data[sflag, :]
        self.Nin_ = data.shape[0]

        sflag_arr = [ 
                data[:, 0] < 1.5, 
                1 / data[:, 2] < -0.084 / z0max, 
                1 / data[:, 2] >  0.037 / z0max, 
                1 / data[:, 2] >      1 / zmax, 
                ]
        sflag = np.logical_not(np.any(np.vstack(sflag_arr).T, axis=1))
        data = data[sflag, :]
        self.N_ = data.shape[0]
        return data

    def fit_transform(self, data, z0max, zmax, zv):
        """Build the estimator from single-level micrometeorolgoical data and
        apply it to estimate surface aerodynamic parameters.

        Parameters
        ----------
        data : ndarray of shape (n_obs, 3)
            Input ``n_obs`` observations of 3 variables from micrometeorology
            measurements in 3 columns of ``data``: alongwind speed, friction
            velocity, and Monin-Obukohv length, in that order. 

        z0max : float
            Upper bound to the expected z0 values (aerodynamic surface
            roughness length, e.g. 10% of canopy height for vegetated surface),
            in meters.

        zmax : float
            Upper bound to the expected z values (effective/aerodynamic
            measurement height, in meters.

        zv : ndarray of shape (n_possible, )
            List of ``n_possible`` possible z values in meters for numerical
            search of optimal z and z0.
       
        Returns
        -------
        z : float 
            Estimated z (effective/aerodynamic measurement height). 

        z0 : float 
            Estimated z0 (surface aerodynamic roughness length). 
        """
        data = self._filter(data, z0max, zmax)
        if self.N_ < self.min_nobs: 
            warnings.warn(
                    'too few data points left after validation.'
                    ' return nan')
            z, z0 = np.nan, np.nan
        else:
            zv_mat, lm_mat = np.meshgrid(zv, data[:, 2])
            _, u_mat = np.meshgrid(zv, data[:, 0])
            _, ustar_mat = np.meshgrid(zv, data[:, 1])
            z0v = _monin_obukhov_z0(zv_mat, u_mat, ustar_mat, lm_mat)

            z0avg = np.nanmean(z0v, axis=0)

            if self.solver == 'sigma-s':
                psim = _psi_m(zv_mat, lm_mat)
                sv = VON_KARMAN * u_mat / ustar_mat - psim
                svstd = np.nanstd(sv, axis=0)
                ix = np.nanargmin(svstd)
            elif self.solver == 'sigma-s-approx':
                z0cv = np.nanstd(z0v, axis=0)/ z0avg
                ix = np.nanargmin(z0cv)
            else:
                raise ValueError(
                        'Unrecognized solver={0:s}'.format(self.solver))
                
            if ix == 0 or ix == len(zv)-1:
                z, z0 = np.nan, np.nan
            else:
                z, z0 = zv[ix], z0avg[ix]

        return z, z0

class SurfaceAerodynamicFVIT():
    pass
