import numpy as np
from sklearn.neighbors import KernelDensity
from scipy import stats
from joblib import Parallel, delayed

#### density estimation
def sklearn_kde(a,xplot,ktype,h):
    """estimate pdf(a) over xplot using sklearn (has more kernel options)
        Params:
            ktype: kernel type
            h: bandwidth"""
    kde = KernelDensity(kernel=ktype,bandwidth=h).fit(a.reshape(-1, 1))
    log_dens = kde.score_samples(xplot.reshape(-1, 1))
    dens = np.exp(log_dens)
    return dens

def scp_kde(a, aplot=None, h=1, return_func=False):
    """estimte pdf(a) over aplot using Gaussian KDE from scipy.stats
        Params:
         h = scaling factor of the bandwidth
         return_func: option to return a callable or the values over aplot"""
    def my_kde_bandwidth(obj, fac=h):
        """calculate width defined by Scott's Rule, multiplied by a constant factor."""
        return np.power(obj.n, -1./(obj.d+4)) * fac
    kde = stats.gaussian_kde(a,bw_method=my_kde_bandwidth)
    if return_func:
        return kde
    else:
        dens = kde(aplot)
        return dens
    
#### generate random sample, not accounting for selection effects 
#### and measure rate by bootstrap random sampling the predictions
def observe_af(obsN,af_pop,ap,nboot=100):
    np.random.seed(0)
    """observe af with size obsN, bootstrap that sample to get the best-estimate p(af_obs)"""
    # observe subsample of af and approx pdf
    af_obs_full = np.random.choice(af_pop,obsN)
    # bootstrap
    def boot(j):
        af_obs = np.random.choice(af_obs_full,obsN,replace=True)
        p_af_obs = scp_kde(af_obs,ap)
        return p_af_obs
    boot_p_af_obs = Parallel(n_jobs=-1)(delayed(boot)(j) for j in range(nboot))
    med_p_af_obs = np.median(boot_p_af_obs,axis=0)
    std_p_af_obs = np.std(boot_p_af_obs,axis=0)
    return boot_p_af_obs,med_p_af_obs, std_p_af_obs

def sample_pred(rate,p_af_preds,ap,obsN=100, nboot=100):
    """bootstrap sample from p_af_pred for one rate, return pdfs"""
    # normalize p_af_pred
    p = np.nan_to_num(p_af_preds[rate], nan=0.0)
    pnorm = p/np.sum(p)
    # sample from pred paf + bootstrap
    samp = np.random.choice(ap, size=(nboot, obsN), p=pnorm)
    # construct bootstrapped pdfs of p_af_pred_sampled
    samp_pdf = np.array([scp_kde(s,ap) for s in samp])
    return samp_pdf # shape [nboot, ap]

def get_pa_per_obsN(obsN,p_af_preds,af_pop,ap,rates):
    """get median p_af_obs and bootstrap+sampled p_af_pred"""
    # get the best approximate p_af_obs from bootstrap
    _, med_p_af_obs, std_p_af_obs = observe_af(obsN,af_pop,ap)
    # sample predicted af 
    boot = lambda rate: sample_pred(rate,p_af_preds,ap,obsN,nboot=100)
    res_p_pred = Parallel(n_jobs=-1)(delayed(boot)(r) for r in rates) # shape [rate, nboot, ap]
    boot_p_af_pred = {r:p for r,p in zip(rates,res_p_pred)}
    return med_p_af_obs, std_p_af_obs, boot_p_af_pred

def cal_dkl_boot_cut(dat, acutmin, acutmax,ap,obs_sampsize):
    """read dat of p(a) and calculate divKL, option to cut a range"""
    dat_dkl = {}
    dat_cut={}
    # define some functions
    cut = lambda p,a: np.where((a>acutmin)&(a<acutmax),p,0)
    normalize = lambda p,a: p / np.trapz(p,a)
    for obsN in obs_sampsize:
        # get data for each obsN
        med_p_af_obs, _, d_p_pred_boot = dat[obsN]
        p_pred_boot = np.array(list(d_p_pred_boot.values()))
        # cut p_obs
        med_p_af_obs_cut = normalize(cut(med_p_af_obs,ap),ap)
        # cut p_pred_boots
        app = ap[np.newaxis,np.newaxis,:]
        p_pred_boot_cut = cut(p_pred_boot,app)
        normfac = np.trapz(p_pred_boot_cut,app,axis=2)
        p_pred_boot_cutnorm = p_pred_boot_cut/normfac[:,:,np.newaxis]
        # calculate divKL
        eps = 1e-10 
        dat_dkl[obsN] = np.sum(med_p_af_obs_cut * np.log((med_p_af_obs_cut + eps) / (p_pred_boot_cutnorm + eps)), axis=2)
        dat_cut[obsN] = [med_p_af_obs_cut,p_pred_boot_cutnorm]
    return dat_dkl, dat_cut