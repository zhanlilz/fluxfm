import sys

import numpy as np

sys.path.insert(0, '../fluxfm')
import z0d_graf_etal as z0d
from micro import VON_KARMAN

test_file = 'data/ECdata_J1lower.txt'

header=1 # number of header lines to omit
miss=-9999 # entry that indicates "missing data" (apart from NaN)

# which column contains what? with first column being 0
col_u=6 # the mean wind speed (m/s)
col_ustar=8 # friction velocity (m/s)
col_Hv=5 # the buoyancy flux (W/m^2) (-1 if not separately available)
col_Tv=3 # the virtual temperature (K) (-1 if not separately available)
col_rho=4 # the mean air density (kg/m^3) (-1 if not separately available)
col_L=-1 # Obukhov length L (Default -1, only give if Hv,Tv,rho missing)
col_stdw=2 # needed for the flux-variance similarity method
col_stdT=9 # needed for the flux-variance similarity method

g=9.81 # gravity acceleration (9.81 m/s^2)
k=VON_KARMAN # von Karman constant (0.4)
cp=1005 # specific heat capacity of air @ const. pressure (1005 J/kg/K)

test_input = np.loadtxt(test_file,  skiprows=header)
test_input = np.where(test_input!=miss, test_input, np.nan)

u = test_input[:, col_u]
rho = test_input[:, col_rho]
ustar = test_input[:, col_ustar]
Hv = test_input[:, col_Hv]
Tv = test_input[:, col_Tv]
stdw = test_input[:, col_stdw]
stdT = test_input[:, col_stdT]

# Compute Obukhov length L from given rho, ustar, Hv, and Tv.
# You can also provide pre-estimated L directly.
L = -rho * ustar**3 / ( k * g * Hv / (cp * Tv) );
# Compute scaling temperature T*
Tstar = - Hv / ( rho * cp * ustar );

z0max = 0.2
zmax = 2.5

# FP-RE approach
# ``data`` is a 3-column array with each column being, alongwind speed,
# friction velocity, and Monin-Obukohv length, in that order. 
data = np.array([u, ustar, L]).T
# FP-RE-1, using Eq. (9) and (10) in Graf et al., 2014
estimator = z0d.SurfaceAerodynamicFPRE()
z, z0 = estimator.fit_transform(data, z0max, zmax)
msg_str = '''FP-RE-1, using Eq. (9) and (10) in Graf et al., 2014:
# of obs./rows in your input data array = {2:d}
# of obs./rows after removing NaN and Inf = {3:d}
# of obs./rows that meets selection criteria = {4:d}
z = {0:.3f}, z0 = {1:.3f}
'''.format(
        z, z0, 
        data.shape[0], 
        estimator.Nin_, 
        estimator.N_)
sys.stdout.write(msg_str)
# FP-RE-2, using Eq. (9) and (11) in Graf et al., 2014
estimator = z0d.SurfaceAerodynamicFPRE(solver='bivariate')
z, z0 = estimator.fit_transform(data, z0max, zmax)
msg_str = '''FP-RE-2, using Eq. (9) and (11) in Graf et al., 2014:
# of obs./rows in your input data array = {2:d}
# of obs./rows after removing NaN and Inf = {3:d}
# of obs./rows that meets selection criteria = {4:d}
z = {0:.3f}, z0 = {1:.3f}
'''.format(
        z, z0, 
        data.shape[0], 
        estimator.Nin_, 
        estimator.N_)
sys.stdout.write(msg_str)

sys.stdout.write('\n')

# FP-IT approach
# ``data`` is a 3-column array with each column being, alongwind speed,
# friction velocity, and Monin-Obukohv length, in that order. 
data = np.array([u, ustar, L]).T
zv = np.arange(0.1, zmax+1, 0.1)
# FP-IT-1
estimator = z0d.SurfaceAerodynamicFPIT()
z, z0 = estimator.fit_transform(data, z0max, zmax, zv)
msg_str = '''FP-IT-1, using Eq. (6) and (7) in Graf et al., 2014:
# of obs./rows in your input data array = {2:d}
# of obs./rows after removing NaN and Inf = {3:d}
# of obs./rows that meets selection criteria = {4:d}
z = {0:.3f}, z0 = {1:.3f}
'''.format(
        z, z0, 
        data.shape[0], 
        estimator.Nin_, 
        estimator.N_)
sys.stdout.write(msg_str)
# FP-IT-2, using Eq. (6) and (8) in Graf et al., 2014
estimator = z0d.SurfaceAerodynamicFPIT(solver='sigma-s-approx')
z, z0 = estimator.fit_transform(data, z0max, zmax, zv)
msg_str = '''FP-IT-2, using Eq. (6) and (8) in Graf et al., 2014:
# of obs./rows in your input data array = {2:d}
# of obs./rows after removing NaN and Inf = {3:d}
# of obs./rows that meets selection criteria = {4:d}
z = {0:.3f}, z0 = {1:.3f}
'''.format(
        z, z0, 
        data.shape[0], 
        estimator.Nin_, 
        estimator.N_)
sys.stdout.write(msg_str)

sys.stdout.write('\n')

# FV-IT approach
# ``data`` is a 6-column array with each column being, alongwind speed,
# friction velocity, Monin-Obukohv length, standard deviation of vertical wind,
# standard deviation of sonic temperature, and friction temperature, in that
# order.
data = np.array([u, ustar, L, stdw, stdT, Tstar]).T
zv = np.arange(0.1, zmax+1, 0.1)
# FV-IT-1, using Eq. (6) and (12) in Graf et al., 2014
estimator = z0d.SurfaceAerodynamicFVIT()
z, z0 = estimator.fit_transform(data, z0max, zmax, zv)
msg_str = '''FV-IT-1, using Eq. (6) and (12) in Graf et al., 2014:
# of obs./rows in your input data array = {2:d}
# of obs./rows after removing NaN and Inf = {3:d}
# of obs./rows that meets selection criteria = {4:d}
z = {0:.3f}, z0 = {1:.3f}
'''.format(
        z, z0, 
        data.shape[0], 
        estimator.Nin_, 
        estimator.N_)
sys.stdout.write(msg_str)
# FV-IT-2, using Eq. (6) and (13) in Graf et al., 2014
estimator = z0d.SurfaceAerodynamicFVIT(solver='sigma-t')
z, z0 = estimator.fit_transform(data, z0max, zmax, zv)
msg_str = '''FV-IT-2, using Eq. (6) and (13) in Graf et al., 2014:
# of obs./rows in your input data array = {2:d}
# of obs./rows after removing NaN and Inf = {3:d}
# of obs./rows that meets selection criteria = {4:d}
z = {0:.3f}, z0 = {1:.3f}
'''.format(
        z, z0, 
        data.shape[0], 
        estimator.Nin_, 
        estimator.N_)
sys.stdout.write(msg_str)
# FV-IT-3, using Eq. (14) and (13) in Graf et al., 2014
estimator = z0d.SurfaceAerodynamicFVIT(solver='sigma-w-reg')
z, z0 = estimator.fit_transform(data, z0max, zmax, zv)
msg_str = '''FV-IT-3, using Eq. (14) and (13) in Graf et al., 2014:
# of obs./rows in your input data array = {2:d}
# of obs./rows after removing NaN and Inf = {3:d}
# of obs./rows that meets selection criteria = {4:d}
z = {0:.3f}, z0 = {1:.3f}
'''.format(
        z, z0, 
        data.shape[0], 
        estimator.Nin_, 
        estimator.N_)
sys.stdout.write(msg_str)
