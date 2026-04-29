from src import *
import scipy.stats as st
import yaml

DEFAULTS = {
    "a0_pop_range": [10, 20],
    "q0_pop_range": [1, 90],
    "sa_range": [0.1,1],
    "sq_range": [0.1, 1],
    "reso": 0.1,
    "Nobs": 100,
    "outdir": "/data/events"
}

def main(config):
    # 0. load config params
    a0_pop_min,a0_pop_max = config['a0_pop_range']
    q0_pop_min,q0_pop_max = config['q0_pop_range']
    sa_min,sa_max = config['sa_range']
    sq_min,sq_max = config['sq_range']
    reso = config['reso']
    Nobs = config['Nobs']
    outdir = config['outdir']

    # 1. Define a large batch size (e.g., 10,000)
    batch_size = 10000 
    np.random.seed(0)

    # 2. Generate arrays for all candidates at once
    a = np.random.uniform(a0_pop_min, a0_pop_max, batch_size)
    q = np.random.uniform(q0_pop_min,q0_pop_max, batch_size)
    z = 0.2#np.random.uniform(0.18, 0.2, batch_size)
    pobs = psel(a, z, q, reso) #* 1/10*1/99
    u = np.random.uniform(0, pobs.max(), batch_size)

    # 3. Apply the mask
    mask = u < pobs
    accepted_a = a[mask]
    accepted_q = q[mask]
    a0_samp = accepted_a[:Nobs]
    q_samp = accepted_q[:Nobs]
    # samples_aq = np.stack([a0_samp,q_samp],axis=1)

    # 4. resample posteriors with negative values
    def gen_event_posteriors(true_param,smin,smax,n_post_samps = 101):
        """generate event posterior samples that is positive
            Params:
                n_post_samps: number of points in an event's posterior sample
        """
        # random sample a measurement about true param, then random sample measurement posterior
        single_event_posteriors = [st.norm(st.norm(a, s).rvs(), s).rvs(n_post_samps) 
                            for a, s in zip(true_param, np.random.uniform(smin,smax, size = len(true_param)))]
        # find posterior with negative value
        find_neg = lambda arr: np.array([i for i in range(len(arr)) if np.min(arr[i])<0])
        neg_ind = find_neg(single_event_posteriors)
        while len(neg_ind) != 0:
            for i in neg_ind:
                s = np.random.uniform(smin,smax)
                a = true_param[i]
                single_event_posteriors[i] = st.norm(st.norm(a, s).rvs(), s).rvs(n_post_samps) 
            neg_ind = find_neg(single_event_posteriors)
        return single_event_posteriors

    single_event_posteriors_a = gen_event_posteriors(a0_samp,sa_min,sa_max)
    single_event_posteriors_q = gen_event_posteriors(q_samp,sq_min,sq_max)
    single_event_posteriors_aq = np.stack([single_event_posteriors_a,single_event_posteriors_q],axis=2)

    # write posteriors to txt file for CLI runs
    for i in range(Nobs):
        filepath = os.path.join(outdir,f"event_{i}.txt")
        np.savetxt(filepath,single_event_posteriors_aq[i,:,:])
 

if __name__ == "__main__":
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    parser = ArgumentParser(description=(
        """
        generate event posteriors on a,q
        """), formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", type=str, default=DEFAULTS['outdir'], help="output directory")
    parser.add_argument("--Nobs", type=int, default=DEFAULTS['Nobs'])
    args = parser.parse_args()

    # update and save config dict
    config = DEFAULTS.copy()
    config.update(vars(args))
    os.makedirs(config["outdir"], exist_ok=True)
    with open(os.path.join(config["outdir"], "gen-event-config.yaml"), "w") as f:
        yaml.dump(config, f)

    main(config)