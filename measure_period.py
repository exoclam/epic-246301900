import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from lightkurve import search
import theano
import exoplanet as xo
import pymc3 as pm
import theano.tensor as tt
import pymc3_ext as pmx
from celerite2.theano import terms, GaussianProcess
from scipy.signal import savgol_filter
from pymc3.step_methods.hmc import quadpotential
import pandas as pd
from astropy.io import fits

#################################################################################
# Read data
#################################################################################
filename = '/blue/sarahballard/c.lam/superflare/ktwo246301900-c12_llc.fits'
hdul = fits.open(filename)
data = fits.getdata(filename,header=True,ext=1)
lc = pd.DataFrame(np.array(data[0]).byteswap().newbyteorder()) # to avoid big-endian/little-endian mixup: https://stackoverflow.com/questions/30283836/creating-pandas-dataframe-from-numpy-array-leads-to-strange-errors
lc = lc.dropna(subset=['TIME','PDCSAP_FLUX','PDCSAP_FLUX_ERR'])

#################################################################################
# Bring in flare times and posteriors from each of three pieces
#################################################################################
npz1 = np.load('/blue/sarahballard/c.lam/superflare/flares/pre_flare.npz')['arr_0']
npz2 = np.load('/blue/sarahballard/c.lam/superflare/flares/post_flare.npz')['arr_0']
npz3 = np.load('/blue/sarahballard/c.lam/superflare/flares/post_gap.npz')['arr_0']
# concatenate flare arrays
npz = npz1[1]+npz2[1]+npz3[1]
print(len(npz), len(lc))
quit()

#################################################################################
# Get normalized light curve for whole times series
#################################################################################
norm_y = normalize_flux(lc.PDCSAP_FLUX)
smooth = savgol_filter(norm_y, 13, polyorder=3)
resid = norm_y - smooth
mask = resid < 1.5*np.sqrt(np.mean(resid**2))

ref_time = calculate_ref_time(lc)
x = lc.TIME - ref_time # subtract time by average time
y = resid

gp_mean = norm_y-npz1 # mean for GP for total light curve fit is observed light curve - flare light curve

# periodogram
results = xo.estimators.lomb_scargle_estimator(
    x, gp_mean, max_peaks=1, min_period=1.0, max_period=8.0, samples_per_peak=50 # replaced y with gp_mean
)

peak = results["peaks"][0]
