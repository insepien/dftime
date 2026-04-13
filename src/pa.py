import numpy as np
from scipy.integrate import quad, simpson
from joblib import Parallel, delayed

#### define some constants
# mbh-sigma constants
epsilon = 0.38
beta = 5.64
alpha = 8.32
# tdf constants
coulomb_log = 3
gamma = 19/ coulomb_log / 25 / 200 * 1e8

#### analytical calculation of p(af) for a0~U
def p_af_sigma(a,mbh,t,a0,R=1):
    """calculate int p(af, sigma|mbh) d~sigma"""
    logmbh = np.log10(mbh)
    prefactor = 1/(np.sqrt(2*np.pi)*epsilon/beta) 
    sigma_star = mbh*t/(gamma/R)/(a0**2-a**2)/200
    exp_term = np.exp(-(np.log10(sigma_star) - 1/beta*logmbh + alpha/beta)**2 / (2*epsilon**2/beta**2))
    jacobian = 2*a/(a0**2-a**2)/np.log(10)
    return prefactor * exp_term * jacobian

def p_a0_uniform(amin,amax):
    """return p(a) for a~U(amin,amax)"""
    return 1/(amax-amin)

def p_af_sigma_a0_unif_fast(amin,amax,mbh,t,R,af_grid=None):
    """for a0~U(amin,amax), 
        marginalize over sigma and a0 to find p(af|mbh) for 1 rate """
    if af_grid is None:
        af_grid = np.linspace(0,amax+3,200)
    a0_grid = np.linspace(0,af_grid.max()+3,10000)

    AF,A0 = np.meshgrid(af_grid,a0_grid,indexing="ij")

    pa_sigma_grid = p_af_sigma(AF, mbh, t, A0, R)
    p_a0_grid = np.array([p_a0_uniform(amin, amax)]*len(a0_grid))
    p_a0_grid = np.where((a0_grid>amin)&(a0_grid<amax),p_a0_uniform(amin, amax),0)
    p_a0_grid = p_a0_grid[np.newaxis,:]

    integrand_mat = pa_sigma_grid*p_a0_grid
    integrand_mat = np.where(A0 >= AF, integrand_mat, 0)
    paf = simpson(integrand_mat,a0_grid,axis=1)
    return af_grid, paf

#### checks w MC
def af_from_a0(a0,mbh,t,R):
    """given mbh, sample sigma and evolve a0 to af under DF individually"""
    logmbh = np.log10(mbh)
    N = len(a0)
    # sample sigma
    logsigma = np.random.normal(loc=1/beta*logmbh-alpha/beta,scale=epsilon/beta,size=N)
    sigma = 10**logsigma*200
    # calculate af
    af = np.sqrt(a0**2 - mbh*t/(gamma/R*sigma))
    return af

def af_mc_unif_a0(amin, amax, mbh=1e8, t=1, R=1, N=10000):
    """assume a0~U(amin,amax), sample {a0} from U(amin,amax) then calculate {af} for some rate"""
    # sample a0
    a0 = np.random.uniform(amin,amax,N)
    af = af_from_a0(a0,mbh,t,R)
    return a0,af

def paf_analytic_allrates(p_a0,a0_samp,mbh,t,af_grid=None,
                        Naf=200,Na0=10000,uniform=False,rates=np.logspace(-1,1,50)):
    """given a sample of a0 and precalculated p_a0 function, 
        return a dictionary of predicted p_af for different rates"""
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
        integrand_matrix = p_af_sigma(AF, mbh, t, A0, R) * p_a0_grid
        # cut to integration limit: a0 >= af    
        integrand_matrix = np.where(A0 >= AF, integrand_matrix, 0)
        # integrate
        p_af = simpson(integrand_matrix, a0_grid, axis=1)
        return p_af
    
    # calculate p_af and save as dict[rate]: array of p_af values for the af array defined above
    res = Parallel(n_jobs=-1)(delayed(optimize_inner)(r) for r in rates)
    p_af_preds = {r:p for r,p in zip(rates,res)}
    return p_af_preds

