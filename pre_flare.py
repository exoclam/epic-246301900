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

# calculate reference (average) time
def calculate_ref_time(lc):
    return 0.5 * (np.min(lc.TIME) + np.max(lc.TIME))

# calculate normalized flux
def normalize_flux(lc):
    return (lc/np.median(lc)) - 1

def find_flares(lc):

    # find some flares
    mflare = resid < 1.5 * np.sqrt(np.mean(resid**2))
    lc.CADENCENO = lc.CADENCENO
    cads = lc.CADENCENO - list(lc.CADENCENO)[0] # index at 0
    ref_time = calculate_ref_time(lc)
    x = lc.TIME - ref_time # subtract time by average time
    y = resid # data with savgol subtracted

    fidx = cads.loc[~mflare][:-1]#[np.diff(cads[~mflare]) < 1]
    fidx = list(fidx) # address indexing errors
    npts = np.array([])
    i = 0
    n = 1
    while True:
        try:
            if fidx[i + 1] == fidx[i] + n:
                # measure flare: keep track of number of points (fatness) of a flare
                fidx = np.delete(fidx, i + 1)
                n += 1
            else:
                # count flare: concatenate to array, advance, and reset count
                npts = np.r_[npts, n+1] 
                i += 1
                n = 1
        except IndexError:
            npts = np.r_[npts, n+1]
            break
    tpeaks = np.array([x.loc[(lc.CADENCENO - list(lc.CADENCENO)[0]) == fidx[i]] for i in range(len(fidx))]).T[0]
    ampls = np.array([resid.loc[(lc.CADENCENO - list(lc.CADENCENO)[0]) == fidx[i]] for i in range(len(fidx))]).T[0]
    fwhms = np.asarray(npts) * (2/1440)
    npeaks = np.shape(tpeaks)[0]
    print(tpeaks, ampls, fwhms, npeaks)
    return x, y, tpeaks, ampls, fwhms, npeaks

# this is the flare model implemented in theano
def get_light_curve(time, tpeaks, fwhms, ampls, texp=None, oversample=7): 
    time = time.astype('float64')
    time = tt.as_tensor_variable(time)
    
    if texp is None:
        tgrid = time
    if texp is not None:
        # taking this oversample code from
        # https://github.com/dfm/exoplanet
        # and https://github.com/lkreidberg/batman
        oversample = int(oversample)
        oversample += 1 - oversample % 2
        dt = np.linspace(-texp / 2., texp / 2.,
                         oversample)
        #print(dt)
        tgrid = tt.shape_padright(time) + dt
    multiflare_lc = multiflaremodel(tgrid, tpeaks, fwhms, ampls)
    
    if texp is not None:
        multiflare_lc = tt.mean(tt.reshape(multiflare_lc, (-1, oversample)),
            axis=1)

    return multiflare_lc

def multiflaremodel(t, tpeaks, fwhms, ampls):
    t = t.astype('float64')
    t = tt.as_tensor_variable(t)
    multiflare_lc = tt.zeros_like(t)
    flare_lc = tt.zeros_like(t)
    
    def scan_func(tpeak, fwhm, ampl):
        zeropad_flare_lc = tt.zeros_like(t)
        tcut = (((t - tpeak)/fwhm > -1.) * ((t - tpeak)/fwhm < 20.)).nonzero()
        flare_lc = _flaremodel(t[tcut], tpeak, fwhm, ampl)
        zeropad_flare_lc = tt.set_subtensor(zeropad_flare_lc[tcut],  flare_lc)
        return zeropad_flare_lc
    
    components, updates = theano.scan(fn=scan_func,
                                      sequences=[tpeaks, fwhms, ampls],
                                      )
    multiflare_lc = tt.sum(components, axis=0)
    
#     for i in range(npeaks):
#         tcut = (((t - tpeaks[i])/fwhms[i] > -1.) * ((t - tpeaks[i])/fwhms[i] < 20.)).nonzero()
#         flare_lc = _flaremodel(t[tcut], tpeaks[i], fwhms[i], ampls[i])
#         multiflare_lc = tt.set_subtensor(multiflare_lc[tcut],  multiflare_lc[tcut] + flare_lc)
    return multiflare_lc

def _flaremodel(t, tpeak, fwhm, ampl):
    # reuses some code from AltaiPony and Apaloosa
    t = tt.as_tensor_variable(t)
    flare_lc = tt.zeros_like(t)
    flare_lc = tt.where((t <= tpeak) * ((t - tpeak)/fwhm > -1.),
                        before_flare(t, tpeak, fwhm, ampl),
                        flare_lc
                       )
    flare_lc = tt.where((t > tpeak) * ((t - tpeak)/fwhm < 20.),
                        after_flare(t, tpeak, fwhm, ampl),
                        flare_lc
                       )
    return flare_lc

def before_flare(t, tpeak, fwhm, ampl):
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    fout = ((_fr[0] + _fr[1] * ((t - tpeak) / fwhm) +
                        _fr[2] * ((t - tpeak) / fwhm)**2. +
                        _fr[3] * ((t - tpeak) / fwhm)**3. +
                        _fr[4] * ((t - tpeak) / fwhm)**4. ) *
                        ampl)
    return fout

def after_flare(t, tpeak, fwhm, ampl):
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]
    fout = ((_fd[0] * tt.exp( ((t - tpeak) / fwhm) * _fd[1]) +
             _fd[2] * tt.exp( ((t - tpeak) / fwhm) * _fd[3] )) *
            ampl)
    return fout

def calculate_energy(kp, zeropoint, ed_model):
    l_sun = 3.0128e28 * 1e7 # Watts * 1e7 erg/s per Watt
    parallax = 21.6345 # in milliarcsecs (mas)
    distance = 22.74 # 46.22
    distance = 1./(parallax*0.001) # parsecs
    print(distance)
    pre_lkp = l_sun * np.power(10, 0.4*(zeropoint-kp)) # https://iopscience.iop.org/article/10.1088/0004-637X/797/2/121/pdf
    print("log pre LKp: ", np.log10(pre_lkp)) # log erg/(s*cm^2)
    lkp = (4*np.pi*distance**2) * pre_lkp * 4000
    print("log LKp: ", np.log10(lkp))
    print("ED: ", ed_model)
    try:
        return np.log10(lkp*ed_model)
    except:
        return np.log10([lkp*ed for ed in ed_model]) # show order of energy in ergs

#################################################################################
# Read data
#################################################################################
filename = '/blue/sarahballard/c.lam/superflare/ktwo246301900-c12_llc.fits'
hdul = fits.open(filename)
data = fits.getdata(filename,header=True,ext=1)
lc = pd.DataFrame(np.array(data[0]).byteswap().newbyteorder()) # to avoid big-endian/little-endian mixup: https://stackoverflow.com/questions/30283836/creating-pandas-dataframe-from-numpy-array-leads-to-strange-errors
print(lc.head())

#################################################################################
# Divide data into pre-flare, post-flare/pre-gap, and post-flare/post-gap
# Get smoothed version of light curve and identify flare params w/find_flares()
#################################################################################
lc = lc.dropna(subset=['TIME','PDCSAP_FLUX','PDCSAP_FLUX_ERR']) # toss NaNs since they'll create matrices of NaNs
pre = lc.loc[:1611] # next record is the beginning of the flare, so stop here
post = lc.loc[1614:2336] # start from after the flare until beginning of the gap in time series data
postgap = lc.loc[2337:] # start from end of the gap in data until end of time series data
superflare = lc.loc[1569:1655] # area around the superflare

yerr = np.abs(pre.PDCSAP_FLUX_ERR/np.median(pre.PDCSAP_FLUX_ERR) - 1)
# get a smoothed version of the light curve
norm_y = normalize_flux(pre.PDCSAP_FLUX)
smooth = savgol_filter(norm_y, window_length=9, polyorder=3) 
resid = norm_y - smooth
#mask = resid < 1.5*np.sqrt(np.mean(resid**2))
x, y, tpeaks, ampls, fwhms, npeaks = find_flares(pre)

#################################################################################
# Create multiflare theano object
# Create pyMC flare model
#################################################################################
xx = tt.dvector('xx')
tpeaksx = tt.dvector('tpeaksx')
fwhmsx = tt.dvector('fwhmsx')
amplsx = tt.dvector('amplsx')
#texp = 0.001388888888888889
#multiflare = theano.function([xx, tpeaksx, fwhmsx, amplsx],
#                    get_light_curve(xx, tpeaksx, fwhmsx, amplsx, texp=texp))
multiflare = theano.function([xx, tpeaksx, fwhmsx, amplsx],
                    get_light_curve(xx, tpeaksx, fwhmsx, amplsx))

# make a fine grid that spans the observation window for plotting purposes
t_plot = np.linspace(x.min(), x.max(), 10000)

with pm.Model() as model:
    npeaks = np.shape(tpeaks)[0]
    mean = pm.Normal("mean", mu=0, sd=3.0, testval=0)
    logs2 = pm.Normal("logs2", mu=np.log(np.var(y)), sd=2.0)
    
    tpeak = pm.Normal("tpeak", mu=tpeaks, sd=np.ones_like(tpeaks) * 0.003,
                      testval=tpeaks, shape=npeaks)
    logFwhm = pm.Normal("logFwhm", mu=np.log(fwhms), sd=1.0, testval=np.log(fwhms),
                        shape=npeaks)
    logAmpl = pm.Normal("logAmpl", mu=np.log(ampls), sd=1.0, testval=np.log(ampls),
                        shape=npeaks)
    
    fwhm = pm.Deterministic("fwhm", tt.exp(logFwhm))
    ampl = pm.Deterministic("ampl", tt.exp(logAmpl))
    
    #flare_lc = get_light_curve(x, tpeak, fwhm, ampl, texp=texp, oversample=15)
    flare_lc = get_light_curve(x, tpeak, fwhm, ampl, oversample=7)
    flare_lc_pred = get_light_curve(t_plot, tpeak, fwhm, ampl, oversample=7)
    pm_determ = pm.Deterministic("flare_lc", flare_lc)
    pm_determ_pred = pm.Deterministic("flare_lc_pred", flare_lc_pred) # this one's for plotting only

    pm.Normal("obs", mu=flare_lc + mean, sd=tt.exp(2*logs2), observed=y)

    map_soln = xo.optimize(start=model.test_point, vars=[1/fwhm, mean])
    map_soln = xo.optimize(start=map_soln, vars=[1/ampl, mean])
    # per Tom's advice, sample 1/A & 1/FWHM for better performance
    map_soln = xo.optimize(start=map_soln, vars=[1/fwhm, 1/ampl]) 
    map_soln = xo.optimize(start=map_soln, vars=[tpeak])

    approx = pm.MeanField(start=map_soln)
    pm.fit(method=pm.KLqp(approx), n=100000,
            obj_optimizer=pm.adagrad_window
        )
    start = approx.sample(draws=2,) # formerly draws=4
    start = list(start)
    stds = approx.bij.rmap(approx.std.eval())
    cov = model.dict_to_array(stds) ** 2
    potential = quadpotential.QuadPotentialDiag(cov)

#################################################################################
# Sample and plot posteriors from multiflare model
#################################################################################
with model:
    step = pm.NUTS(potential=potential, target_accept=0.9)
    trace = pm.sample(draws=2400, tune=200, step=step, start=start, cores=2) #formerly draws=2400, tune=200

# make a fine grid that spans the observation window for plotting purposes
# from https://docs.exoplanet.codes/en/latest/tutorials/rv/
t_plot = np.linspace(x.min(), x.max(), 10000)

flc_plot =  np.percentile(trace['flare_lc_pred'], [5, 50, 95], axis=0)
flc = np.percentile(trace['flare_lc'], [5, 50, 95], axis=0)
np.savez('flares/pre_flare.npz', flc)

fig, axes = plt.subplots(ncols=2, nrows=np.ceil(npeaks/2).astype(int), figsize=[12,npeaks*2])
axes = axes.flatten()
for i in range(npeaks):
    timemask_t_plot = (t_plot >= tpeaks[i]-0.2) * (t_plot < tpeaks[i]+0.2)
    timemask = (x >= tpeaks[i]-0.2) * (x < tpeaks[i]+0.2)
    ax = axes[i]
    ax.plot(x[timemask], y[timemask], '.k')
    ax.plot(t_plot[timemask_t_plot], flc_plot[1][timemask_t_plot], label="model_pred", zorder=1001)
    art = ax.fill_between(t_plot[timemask_t_plot], 
                          flc_plot[0][timemask_t_plot],
                          flc_plot[2][timemask_t_plot], color="C1", alpha=0.3,
                               zorder=1000)
    art.set_edgecolor("none")
    ax.set_xlim(tpeaks[i]-0.2, tpeaks[i]+0.2)
    #ax.set_ylim(np.min(y[timemask]) - 1, 1 + np.max(y[timemask]))
    ax.legend()
    
fig.tight_layout()
fig.savefig('plots/flare-model-pregap.png')

#################################################################################
# Flare energies
#################################################################################
# get unnormalized times
tpeaks_unnormalized = [tpeak+calculate_ref_time(pre) for tpeak in tpeaks]

# get equivalent durations
ed_model = []
for i in range(npeaks):
    timemask = (x >= tpeaks[i]-0.2) & (x < tpeaks[i]+0.2)
    ed_model.append(np.trapz(flc[1][timemask]*86400, x[timemask]))

# use kepmag value from exofop: https://exofop.ipac.caltech.edu/k2/edit_target.php?id=246301900
flare_energies = calculate_energy(kp=10.807, zeropoint=-20.24, ed_model=ed_model)
print("log flare energy: ", flare_energies)
fe_dict = {'time': tpeaks_unnormalized, 'flare_energies': flare_energies, 'ed': ed_model}
fe = pd.DataFrame(fe_dict)
fe.to_csv('flares/flare_energies_pre.csv')

"""
#################################################################################
# Subtract flare light curve and get initial period guess
#################################################################################
gp_mean = norm_y-flc[1] # mean for GP for total light curve fit is observed light curve - flare light curve
# periodogram
results = xo.estimators.lomb_scargle_estimator(
    x, gp_mean, max_peaks=1, min_period=1.0, max_period=8.0, samples_per_peak=50 # replaced y with gp_mean
)
peak = results["peaks"][0]
#freq, power = results["periodogram"]

##########################################################################################
# Model flare-subtracted light curve with pyMC3 and GP likelihood, using entire timeseries
##########################################################################################
norm_y = normalize_flux(lc.PDCSAP_FLUX)
norm_yerr = normalize_flux(lc.PDCSAP_FLUX_ERR)

with pm.Model() as model:

    # The mean flux of the time series
    mean = pm.Normal("mean", mu=np.mean(gp_mean), sd=np.std(gp_mean)) # replaced 0.0 and 10.0
    # A jitter term describing excess white noise
    jitter = pm.Lognormal("jitter", mu=np.log(np.mean(norm_yerr)), sd=np.log(np.std(norm_yerr))) # try not logging
    #jitter = pm.Lognormal("jitter", mu=0., sd=np.std(norm_yerr))
    
    # A term to describe the non-periodic variability
    #sigma = pm.InverseGamma(
    #    "sigma", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0))
    #rho = pm.InverseGamma(
    #    "rho", **pmx.estimate_inverse_gamma_parameters(0.5, 2.0))
    sigma = pm.InverseGamma(
        "sigma", **pmx.estimate_inverse_gamma_parameters(1.0, 2.5))
    rho = pm.InverseGamma(
        "rho", **pmx.estimate_inverse_gamma_parameters(0.5, 1.5))
    
    # The parameters of the RotationTerm kernel
    sigma_rot = pm.InverseGamma(
        "sigma_rot", **pmx.estimate_inverse_gamma_parameters(1.0, 2.0)) # formerly Beta param was 5.0, not 2.0
    
    #period = pm.Lognormal("period", mu=np.log(peak["period"]), sd=1.0) 
    period = pm.Normal("period", mu=peak["period"], sd=1.0)  

    #Q0 = pm.Lognormal("Q0", mu=0, sd=2) # or np.log(np.mean(gp_mean))?  
    #dQ = pm.Lognormal("dQ", mu=0, sd=2) # or np.log(np.mean(gp_mean))?
    Q0 = pm.Lognormal("Q0", mu=0, sd=2) # or np.log(np.mean(gp_mean))?  
    dQ = pm.Lognormal("dQ", mu=0, sd=2) # or np.log(np.mean(gp_mean))?
    f = pm.Uniform("f", lower=0.1, upper=1.0)
    
    # Set up the Gaussian Process model
    kernel = terms.SHOTerm(sigma=sigma, rho=rho, Q=1 / 3.0) # tweaking rho and sigma affects whether I'm touch-tight
    kernel += terms.RotationTerm(
        sigma=sigma_rot, period=period, Q0=Q0, dQ=dQ, f=f)
    gp = GaussianProcess(
        kernel, t=x, mean=mean, 
        diag=np.asarray(norm_yerr**2)+jitter**2, quiet=True) # diag = norm_yerr**2 + jitter**2

    # Compute the Gaussian Process likelihood and add it into the
    # the PyMC3 model as a "potential"
    gp.marginal("gp", observed=norm_y)

    # Compute the mean model prediction for plotting purposes
    pm.Deterministic("pred", gp.predict(norm_y))

    # Optimize to find the maximum a posteriori parameters
    map_soln = pmx.optimize()

# sample from posterior
np.random.seed(10863087)
with model:
    trace = pmx.sample(
        tune=2500,
        draws=2500,
        start=map_soln,
        cores=2,
        chains=2,
        target_accept=0.95,
    )

#################################################################################
# Identify tpeaks, ie. flare times
# Bring in tpeaks from post_flare and post_gap
#################################################################################
tpeaks_norm1 = [tpeak+calculate_ref_time(pre) for tpeak in tpeaks]
tpeaks2 = [-7.13061199, -3.82069349, -2.30875702,  6.64025654,  6.92629784]
tpeaks3 = [-2.44161407,  0.68441788]
tpeaks_norm2 = [tpeak+calculate_ref_time(post) for tpeak in tpeaks2]
tpeaks_norm3 = [tpeak+calculate_ref_time(postgap) for tpeak in tpeaks3]
tpeaks_total = tpeaks_norm1+tpeaks_norm2+tpeaks_norm3
tpeaks_total.append(2938.333899) # don't forget about the superflare!
"""



