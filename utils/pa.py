import numpy as np
from scipy import stats
from scipy.integrate import quad, simpson
from sklearn.neighbors import KernelDensity
from joblib import Parallel, delayed

# define some constatns
# mbh-sigma constants
epsilon = 0.38
beta = 5.64
alpha = 8.32
# tdf constants
coulomb_log = 3
gamma = 19/ coulomb_log / 25 / 200 * 1e8

def pa_sigma(a,mbh,t,a0,R=1):
    """p(af|mbh) for af = af(mbh, sigma)"""
    # calculate p(af|mbh)
    logmbh = np.log10(mbh)
    prefactor = 1/(np.sqrt(2*np.pi)*epsilon/beta) 
    sigma_star = mbh*t/(gamma/R)/(a0**2-a**2)/200
    exp_term = np.exp(-(np.log10(sigma_star) - 1/beta*logmbh + alpha/beta)**2 / (2*epsilon**2/beta**2))
    jacobian = 2*a/(a0**2-a**2)/np.log(10)
    return prefactor * exp_term * jacobian

def p_a0_uniform(amin,amax):
    """find p(a) over U(amin,amax)"""
    return 1/(amax-amin)

def pa_sigma_a0_unif_integrand(a0, a, mbh, t, R, amin, amax):
    """return the integrand of p(af|mbh), 
        i.e. p(af|mbh,sigma,a0)*p(sigma|mbh)*p(a0|mbh)"""
    return pa_sigma(a, mbh, t, a0, R) * p_a0_uniform(amin, amax)

def pa_sigma_a0_unif(amin,amax,mbh,t,R,a=None):
    """find p(af) for 1 rate by integrating over all a0 for each af(a) value
        using quad int"""
    if a is None:
        a = np.linspace(0,amax+3,200)
    p = []
    for a_ in a:
        alower = np.max([a_,amin])
        p.append(quad(pa_sigma_a0_unif_integrand, alower, amax, args=(a_,mbh,t,R,amin,amax))[0])
    return a, p

def pa_sigma_a0_unif_fast(amin,amax,mbh,t,R,af_grid=None):
    """find p(af) for 1 rate by integrating over all a0 for each af(a) value
        using vectorization"""
    if af_grid is None:
        af_grid = np.linspace(0,amax+3,200)
    a0_grid = np.linspace(0,af_grid.max()+3,10000)

    AF,A0 = np.meshgrid(af_grid,a0_grid,indexing="ij")

    pa_sigma_grid = pa_sigma(AF, mbh, t, A0, R)
    p_a0_grid = np.array([p_a0_uniform(amin, amax)]*len(a0_grid))
    p_a0_grid = np.where((a0_grid>amin)&(a0_grid<amax),p_a0_uniform(amin, amax),0)
    p_a0_grid = p_a0_grid[np.newaxis,:]

    integrand_mat = pa_sigma_grid*p_a0_grid
    integrand_mat = np.where(A0 >= AF, integrand_mat, 0)
    paf = simpson(integrand_mat,a0_grid,axis=1)
    return af_grid, paf

def af_from_a0(a0,mbh,t,R):
    """given mbh, sample sigma and evolve a0 to af under DF"""
    logmbh = np.log10(mbh)
    N = len(a0)
    # sample sigma
    logsigma = np.random.normal(loc=1/beta*logmbh-alpha/beta,scale=epsilon/beta,size=N)
    sigma = 10**logsigma*200
    af = np.sqrt(a0**2 - mbh*t/(gamma/R*sigma))
    return af

def monte_carlo_unif_a0(amin, amax, mbh=1e8, t=1, R=1, N=10000):
    """sample {a0} from U(amin,amax) then find corresponding {af}"""
    # sample a0
    a0 = np.random.uniform(amin,amax,N)
    af = af_from_a0(a0,mbh,t,R)
    return a0,af

def plot_pa_unif(amin,amax,mbh,t,R,ax,clr,checkFast=False):
    """assume a0~U, given a0 range, integrate analyticaly to find p(af) and plot
        default = use quad int. has option for fast integration"""
    # find p(a|mbh)
    a,p = pa_sigma_a0_unif(amin,amax,mbh,t,R)
    # calculate prior over large range for pretty plot
    unif = lambda x,amin,amax: 0 if (x<amin) or (x> amax) else 1/(amax-amin)
    apr = np.linspace(0,20,100)
    pr = [unif(a_,amin,amax) for a_ in apr]
    # plot
    ax.plot(a,p,c=clr,label=rf"$a_0\sim U({{{amin}}}-{{{amax}}}),R={R}$")
    ax.plot(apr,pr,linestyle="--",c=clr,alpha=0.7)
    if checkFast:
        aF,pF = pa_sigma_a0_unif_fast(amin,amax,mbh,t,R)
        ax.plot(aF,pF,c='k',label=rf"fast",linestyle='--')

def sklearn_kde(a,xplot,ktype,h):
    """estimate pdf(a) over xplot using sklearn
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

def div_KL(p,q):
    """calculate KL divergence given pdf p (observed) and q (predicted)"""
    d = p*(np.log10(p/q))
    div = np.sum(d[np.isfinite(d)])
    return div

def paf_analytic_allrates(p_a0,a0_samp,mbh,t,af_grid=None,
                        Naf=200,Na0=10000,uniform=False,rates=np.logspace(-1,1,50)):
    """given a sample of a0 and precalculated p_a0 function, return a dictionary of predicted p_af for different rates"""
    if af_grid is None:
        af_grid = np.linspace(0, np.max(a0_samp) + 3, Naf)
    
    # grid for a0
    a0_grid = np.linspace(np.min(af_grid), np.max(af_grid), Na0)
    # meshgrid for a0, af
    AF, A0 = np.meshgrid(af_grid, a0_grid, indexing='ij')

    # grid for p_a0
    if uniform:
        amax = np.ceil(np.max(a0_samp))
        amin = np.floor(np.min(a0_samp))
        p_a0_vals = np.array([1/(amax-amin)]*len(a0_grid))
        p_a0_vals = np.where((a0_grid<amax) & (a0_grid>amin), 1/(amax-amin), 0)
    else:
        p_a0_vals = p_a0(a0_grid)
    # reshape to mesh grid
    p_a0_grid = p_a0_vals[np.newaxis,:]

    def optimize_inner(R):
        """function to parallelize p_af caculation for different rates"""
        # grid of integrand
        integrand_matrix = pa_sigma(AF, mbh, t, A0, R) * p_a0_grid
        # cut to integration limit: a0 >= af    
        integrand_matrix = np.where(A0 >= AF, integrand_matrix, 0)
        # integrate
        p_af = simpson(integrand_matrix, a0_grid, axis=1)
        return p_af
    
    # calculate p_af and save as dict[rate]:p_af
    res = Parallel(n_jobs=-1)(delayed(optimize_inner)(r) for r in rates)
    p_af_preds = {r:p for r,p in zip(rates,res)}
    return p_af_preds