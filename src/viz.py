import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from .pa import *

#### some pa construction checks
def plot_pa_unif(amin,amax,mbh,t,R,ax,clr):
    """assuming a0~U, given a0 range, integrate analyticaly to find p(af) and plot
        default = use quad int. has option for fast integration"""
    # find p(a|mbh)
    aF,pF = p_af_sigma_a0_unif_fast(amin,amax,mbh,t,R)
    # calculate prior over large range for pretty plot
    unif = lambda x,amin,amax: 0 if (x<amin) or (x> amax) else 1/(amax-amin)
    apr = np.linspace(0,20,100)
    pr = [unif(a_,amin,amax) for a_ in apr]
    # plot
    ax.plot(aF,pF,c=clr,label=rf"$a_0\sim U({{{amin}}}-{{{amax}}}),R={R}$")
    ax.plot(apr,pr,linestyle="--",c=clr,alpha=0.7)
        
#### big rate measurement plot
def plot_everything(dat,dat_dkl, dat_cut,ap,obs_sampsize,
                        rates=np.logspace(-1,1,50),tR=1):
    norm = mcolors.LogNorm(vmin=rates.min(), vmax=rates.max())
    # custom colormap
    start_color = 'palevioletred'
    mid_color = 'honeydew'
    end_color = 'seagreen'
    cmap = LinearSegmentedColormap.from_list(
        name='custom_diverging',
        colors=[start_color, mid_color, end_color],
    )
    fig,ax = plt.subplots(3,len(obs_sampsize),figsize=(int(3*len(obs_sampsize)),9),)

    for obsN, i in zip(obs_sampsize, range(len(obs_sampsize))):
        # 1. Plot a0
        # ax[0,i].plot(ap, p_a0_obs_val, c='k', ls='--')
        
        # 2. Batch Plot Predicted af
        med_p_preds = np.median(np.array(list(dat[obsN][-1].values())),axis=1)
        af_seg = [np.column_stack([ap, med_p_preds[i,:]]) for i in range(len(rates))]
        ax[0, i].add_collection(LineCollection(af_seg, colors=cmap(norm(rates)), alpha=0.5))

        # 2.1. Plot cut pdfs
        med_p_af_obs_cut,p_pred_boot_cut = dat_cut[obsN]
        med_p_pred_cutnorm = np.median(p_pred_boot_cut,axis=1)
        predcut_segs = LineCollection([np.column_stack([ap,p]) for p in med_p_pred_cutnorm],colors= cmap(norm(rates)),alpha=0.5)
        ax[1,i].add_collection(predcut_segs)
        ax[1,i].plot(ap,med_p_af_obs_cut,c='k',alpha=0.5)#,label=rf"a$\sim$({acutmin}-{acutmax})")
        

        # 3. Plot best-estimate p(af_obs)
        med_pobs,std_pobs = dat[obsN][:2]
        ax[0,i].plot(ap,med_pobs,c='k',alpha=0.5)
        ax[0,i].fill_between(ap,med_pobs-2*std_pobs,med_pobs+2*std_pobs,color='grey',alpha=0.5)

        # 4. Plot D_KL with errors
        flat_dkl = np.concatenate([dat_dkl[obsN][:,j] for j in range(100)])
        flat_rates = np.tile(rates, 100)
        ax[2,i].scatter(flat_rates, flat_dkl, c=cmap(norm(flat_rates)), s=1, alpha=0.5)

        # # 5. Plot Vlines
        min_rates = [rates[np.where(dat_dkl[obsN][:,i]==dat_dkl[obsN][:,i].min())[0][0]] for i in range(100)]
        ax[2,i].vlines(min_rates, 0, np.max(flat_dkl), color='grey', alpha=0.1)
        # true rate
        ax[2,i].axvline(tR,c='k',label='true',linestyle='--')

        # get summary stats from bootstraps
        medR,sR = np.median(min_rates), np.std(min_rates)
        ax[2,i].set_title(rf"R={medR:.2f}$\pm${sR*2:.2f}")

        # cosmetics
        ax[0,i].set_xlabel(r"$a$")
        ax[0,i].set_ylabel(r"$p(a)$")
        ax[0,i].set_title(f"N={obsN}")
        ax[0,i].set_ylim(ymax=0.25)

        ax[1,i].set_xlabel(r"$a$")
        ax[1,i].set_ylabel(r"$p(a)$")
        # ax[1,i].set_ylim(ymax=0.25)

        ax[2,i].set_xlabel("rates")
        ax[2,i].set_ylabel(rf"$D_{{KL}}$")
        ax[2,i].legend(loc='upper right')

    ax[1,-1].legend(loc='upper right',bbox_to_anchor=(1,1.2))
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax[0,2],label='log(rates)')
    fig.tight_layout();